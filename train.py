"""
FIXED Training Script

Key Fixes:
1. Scheduler now steps per EPOCH (not per batch)
2. Gradient clipping increased to 1.0 (from 0.5)
3. Uses ReduceLROnPlateau (simpler, more robust for GNNs)
4. Proper batch size recommendations
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import logger
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

from dataset import StepPredictionDataset, create_split
from losses import compute_hit_at_k, ProofSearchRankingLoss
import sys
sys.path.insert(0, '.')
from model import get_model
import re
from losses import ApplicabilityConstrainedLoss, compute_mrr


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

class CurriculumScheduler:
    """
    Progressive difficulty scheduling (from Phase 1).
    Adapted for Phase 2 dataloader.
    """
    def __init__(self, dataset: StepPredictionDataset, 
                 base_difficulty: float = 0.15, 
                 max_difficulty: float = 0.95):
        self.dataset = dataset
        self.base_difficulty = base_difficulty
        self.max_difficulty = max_difficulty
        
        # Pre-calculate difficulties
        print("Initializing CurriculumScheduler (fast metadata scan)...")
        
        # --- START FIX: Avoid calling dataset[i] ---
        # Old slow way (caused the hang):
        # self.difficulties = [
        #     dataset[i].difficulty.item() for i in range(len(dataset))
        # ]
        
        # New fast way: Read metadata directly from the dataset's samples tuple
        self.difficulties = []
        if not dataset.samples:
            print("Warning: CurriculumScheduler found no samples.")
        
        for sample_tuple in self.dataset.samples:
            # sample_tuple is (inst, step_idx, has_spectral, metadata)
            metadata = sample_tuple[3] 
            
            # Use the *exact same* difficulty logic as dataset.py's _estimate_difficulty
            num_rules = metadata.get('n_rules', 10) / 100.0
            num_facts = metadata.get('n_nodes', 20) / 200.0
            proof_length = metadata.get('proof_length', 5) / 50.0
            difficulty = (0.4 * num_rules + 
                          0.3 * num_facts + 
                          0.3 * proof_length)
            self.difficulties.append(min(max(difficulty, 0.0), 1.0))
        # --- END FIX ---
        
        print(f"Difficulty scores cached for {len(self.difficulties)} samples.")

    def get_batch_indices(self, epoch: int, total_epochs: int) -> list[int]:
        """Return indices for current difficulty level."""
        # Linear progression from base to max
        progress = epoch / max(total_epochs, 1)
        target_difficulty = (self.base_difficulty + 
                            (self.max_difficulty - self.base_difficulty) * progress)
        
        # Select instance indices below the target difficulty
        valid_indices = [
            i for i, diff in enumerate(self.difficulties)
            if diff <= target_difficulty
        ]
        
        if not valid_indices:
            # Fallback: take N easiest samples
            sorted_indices = sorted(enumerate(self.difficulties), key=lambda x: x[1])
            valid_indices = [i for i, _ in sorted_indices[:max(1, len(self.difficulties) // 10)]]
            logger.warning(f"Curriculum fallback: using {len(valid_indices)} easiest samples")
        
        print(f"  [Curriculum] Epoch {epoch}: Target difficulty <= {target_difficulty:.3f}, "
              f"found {len(valid_indices)}/{len(self.difficulties)} samples.")
        return valid_indices


def train_epoch(model, loader, optimizer, criterion, device, epoch,
                grad_accum_steps=1, value_loss_weight=0.1, tactic_loss_weight=0.1):
    """Fixed training loop with proper mask extraction."""
    model.train()

    total_rank_loss = 0
    total_value_loss = 0
    total_tactic_loss = 0
    correct_preds = 0
    n_samples = 0 # Counter for number of GRAPHS processed

    optimizer.zero_grad()

    # Use tqdm for progress bar
    from tqdm import tqdm
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)

    # for batch_idx, batch in enumerate(loader): # OLD
    for batch_idx, batch in enumerate(progress_bar): # NEW: Use tqdm
        batch = batch.to(device)

        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask,
            batch.step_numbers, batch.eigvecs, batch.eigvals,
            batch.eig_mask, batch.edge_attr, batch.batch
        )

        batch_size = batch.num_graphs # Number of graphs in this batch

        # --- START FIX: Calculate Tactic Loss for the whole batch ---
        tactic_targets = batch.tactic_target # Should have shape [batch_size] or [batch_size, 1]

        # Ensure target is the correct shape (likely [batch_size])
        if tactic_targets.dim() > 1:
            tactic_targets = tactic_targets.squeeze()

        # Handle potential empty batch edge case & shape mismatch safety
        # Ensure tactic_logits has rows equal to batch_size
        if batch_size > 0 and tactic_logits.shape[0] == batch_size and tactic_targets.shape[0] == batch_size:
            # Calculate tactic loss ONCE for the whole batch
            tac_loss = F.cross_entropy(tactic_logits, tactic_targets)
            batch_tactic_loss_value = tac_loss.item() # For logging this batch's tactic loss
        else:
             # Avoid error if batch somehow becomes empty or shapes mismatch
             tac_loss = torch.tensor(0.0, device=device, requires_grad=True)
             batch_tactic_loss_value = 0.0
             if batch_size > 0: # Log if there's a mismatch
                  print(f"Warning: Shape mismatch for tactic loss. Logits: {tactic_logits.shape}, Targets: {tactic_targets.shape}, Batch Size: {batch_size}")
        # --- END FIX ---

        batch_rank_val_loss = 0 # Accumulator for ranking + value loss for graphs in this batch
        batch_value_loss_item = 0 # Accumulator for value loss item value for logging
        batch_rank_loss_item = 0  # Accumulator for rank loss item value for logging
        graphs_in_batch_processed = 0 # Count graphs successfully processed in this batch

        # --- Loop through graphs for Ranking and Value loss ---
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            # Handle case where a graph might have zero nodes after processing
            if graph_scores.numel() == 0:
                continue

            graph_embeddings = embeddings[mask]
            target_idx = batch.y[i].item()

            # Get applicable mask for this graph
            graph_applicable = batch.applicable_mask[mask] if hasattr(batch, 'applicable_mask') else None

            # Skip if target is invalid or graph had no nodes
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue

            # Calculate Ranking loss for this graph
            rank_loss = criterion(
                graph_scores,
                graph_embeddings,
                target_idx,
                applicable_mask=graph_applicable
            )

            # Calculate Value loss for this graph
            # Ensure value and target_value tensors have compatible shapes for MSELoss
            graph_value = value[i].unsqueeze(0) # Make it [1]
            target_value = batch.value_target[i].unsqueeze(0) # Make it [1]
            val_loss = F.mse_loss(graph_value, target_value)

            # Accumulate per-graph RANKING + VALUE losses
            if not (torch.isnan(rank_loss) or torch.isinf(rank_loss)):
                graph_rank_val_loss = (
                    rank_loss +
                    value_loss_weight * val_loss
                    # Tactic loss is handled separately for the batch
                )
                batch_rank_val_loss += graph_rank_val_loss # Accumulate Tensor loss for backprop
                batch_value_loss_item += val_loss.item() # Accumulate value loss item for logging
                batch_rank_loss_item += rank_loss.item() # Accumulate rank loss item for logging
                graphs_in_batch_processed += 1 # Increment count of processed graphs

                # Count correct predictions for accuracy
                if graph_scores.argmax().item() == target_idx:
                    correct_preds += 1

        # --- Calculate final batch loss (Avg Rank+Val + Tactic) and backpropagate ---
        if graphs_in_batch_processed > 0:
            # Average the Rank+Val loss over graphs successfully processed in the batch
            avg_rank_val_loss_for_batch = batch_rank_val_loss / graphs_in_batch_processed
            # Add the single batch-wide tactic loss
            combined_batch_loss = avg_rank_val_loss_for_batch + (tactic_loss_weight * tac_loss)

            # Scale loss for gradient accumulation
            final_batch_loss = combined_batch_loss / grad_accum_steps
            final_batch_loss.backward() # Backpropagate the combined loss

            # Accumulate total losses for epoch logging (use item values)
            total_rank_loss += (batch_rank_loss_item / graphs_in_batch_processed) # Add avg rank loss item
            total_value_loss += (batch_value_loss_item / graphs_in_batch_processed) # Add avg value loss item
            total_tactic_loss += batch_tactic_loss_value # Add tactic loss item (already calculated for batch)

            n_samples += graphs_in_batch_processed # Accumulate number of graphs processed

            # --- Gradient accumulation step ---
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Use args.grad_clip
                optimizer.step()
                optimizer.zero_grad()

        # Update tqdm progress bar description (optional)
        progress_bar.set_postfix({
             'RankL': f'{total_rank_loss / max(batch_idx + 1, 1):.4f}',
             'ValL': f'{total_value_loss / max(batch_idx + 1, 1):.4f}',
             'TacL': f'{total_tactic_loss / max(batch_idx + 1, 1):.4f}'
        })

    progress_bar.close() # Close tqdm bar after epoch

    # --- Calculate epoch averages ---
    num_batches_processed = batch_idx + 1
    # Use n_samples for accuracy as it counts graphs, not batches
    accuracy = correct_preds / max(n_samples, 1)
    # Average losses over number of batches where loss was calculated
    avg_loss = total_rank_loss / max(num_batches_processed, 1)
    avg_value_loss = total_value_loss / max(num_batches_processed, 1)
    avg_tactic_loss = total_tactic_loss / max(num_batches_processed, 1)


    return avg_loss, avg_value_loss, avg_tactic_loss, accuracy
    
@torch.no_grad()
def evaluate(model, loader, criterion, device, split='val'):
    """Fixed evaluation with proper mask extraction."""
    model.eval()
    
    losses = []
    hit1, hit3, hit5, hit10 = [], [], [], []
    ranks = []
    value_loss_fn = nn.MSELoss()
    value_losses = []
    tactic_loss_fn = nn.CrossEntropyLoss()
    tactic_losses = []

    for batch in loader:
        batch = batch.to(device)
        
        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask,
            batch.step_numbers, batch.eigvecs, batch.eigvals,
            batch.eig_mask, batch.edge_attr, batch.batch
        )
        batch_size = batch.num_graphs
        
        tactic_targets = batch.tactic_target.squeeze()
        tac_loss = tactic_loss_fn(tactic_logits, tactic_targets)
        tactic_losses.append(tac_loss.item())

        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            target_idx = batch.y[i].item()
            
            # ✅ FIX: Extract applicable_mask for this specific graph
            if hasattr(batch, 'applicable_mask'):
                graph_applicable = batch.applicable_mask[mask]
            else:
                graph_applicable = None
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            # ✅ Pass graph-specific applicable_mask
            loss = criterion(
                graph_scores, 
                graph_embeddings, 
                target_idx,
                applicable_mask=graph_applicable
            )
            losses.append(loss.item())

            val_loss = value_loss_fn(value[i], batch.value_target[i])
            value_losses.append(val_loss.item())

            
            # ✅ Pass graph-specific applicable_mask to all metrics
            hit1.append(compute_hit_at_k(graph_scores, target_idx, 1, 
                                        applicable_mask=graph_applicable))
            hit3.append(compute_hit_at_k(graph_scores, target_idx, 3,
                                        applicable_mask=graph_applicable))
            hit5.append(compute_hit_at_k(graph_scores, target_idx, 5,
                                        applicable_mask=graph_applicable))
            hit10.append(compute_hit_at_k(graph_scores, target_idx, 10,
                                         applicable_mask=graph_applicable))
            
            ranks.append(compute_mrr(graph_scores, target_idx, 
                                    applicable_mask=graph_applicable))
    
    mrr = sum(ranks) / max(len(ranks), 1)
    
    return {
        f'{split}_loss': sum(losses) / max(len(losses), 1),
        f'{split}_value_loss': sum(value_losses) / max(len(value_losses), 1),
        f'{split}_tactic_loss': sum(tactic_losses) / max(len(tactic_losses), 1),
        f'{split}_hit1': sum(hit1) / max(len(hit1), 1),
        f'{split}_hit3': sum(hit3) / max(len(hit3), 1),
        f'{split}_hit5': sum(hit5) / max(len(hit5), 1),
        f'{split}_hit10': sum(hit10) / max(len(hit10), 1),
        f'{split}_mrr': mrr,
        f'{split}_n': len(hit1)
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--exp-dir', default='experiments/fixed_run')
    parser.add_argument('--epochs', type=int, default=100)
    
    # FIXED: Recommended batch size (was 16)
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (effective batch = batch_size * grad_accum_steps)')
    
    # FIXED: Recommended learning rate
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Peak learning rate')
    
    # FIXED: Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Loss function
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--hard-neg-weight', type=float, default=2.0)
    
    # FIXED: Recommended grad accumulation
    parser.add_argument('--grad-accum-steps', type=int, default=2,
                       help='Gradient accumulation steps (effective batch = 32*2 = 64)')
    
    # FIXED: Gradient clipping
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    parser.add_argument('--use-type-aware', action='store_true', default=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spectral-dir', type=str, default=None,
                       help='Directory containing precomputed spectral features')
    parser.add_argument('--value-loss-weight', type=float, default=0.1,
                       help='Weight for the value prediction loss')
    parser.add_argument('--tactic-loss-weight', type=float, default=0.1,
                       help='Weight for the tactic prediction loss')

    parser.add_argument('--k-dim', type=int, default=16,
                   help='Spectral dimension (k) to use. Must match preprocessing.')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("Warning: CUDA and MPS not available. Defaulting to CPU.")
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")
    print("="*70)
    
    print("\n" + "="*70)
    print("FIXED TRAINING - All Corrections Applied")
    print("="*70)
    # print(f"Loss: MarginRankingLoss (FIXED: 70% hard, 30% easy)")
    print(f"Loss: ProofSearchRankingLoss (wrapping ApplicabilityConstrainedLoss)")
    print(f"Batch Size: {args.batch_size} (Effective: {args.batch_size * args.grad_accum_steps})")
    print(f"Gradient Accumulation: {args.grad_accum_steps} steps")
    print(f"Gradient Clipping: {args.grad_clip}")
    print(f"Model: Fixed Spectral-Temporal GNN")
    print(f"  - Parallel pathways (spectral, spatial, temporal)")
    print(f"  - Multi-scale temporal encoding")
    print(f"  - Late fusion of all pathways")
    print(f"Params: LR={args.lr}, Hidden={args.hidden_dim}, Layers={args.num_layers}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
    
    # Load data
    train_files, val_files, test_files = create_split(args.data_dir, seed=args.seed)
    
    train_ds = StepPredictionDataset(train_files, spectral_dir=args.spectral_dir, seed=args.seed, k_dim=args.k_dim)
    val_ds = StepPredictionDataset(val_files, spectral_dir=args.spectral_dir, seed=args.seed+1, k_dim=args.k_dim)
    test_ds = StepPredictionDataset(test_files, spectral_dir=args.spectral_dir, seed=args.seed+2, k_dim=args.k_dim)
    
    print(f"\nDataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    num_tactics = train_ds.num_tactics # Get from dataset
    curriculum_scheduler = CurriculumScheduler(train_ds)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    
    # Model
    # Get in_dim from the first sample, but NOT k_dim
    in_dim = train_ds[0].x.shape[1]
    # Use the EXPLICIT k-dim from the command line
    k_dim = args.k_dim 
    print(f"Input dimension: {in_dim} (Base) + {k_dim} (Spectral) [FORCED via --k-dim]")
    model = get_model(
        in_dim, 
        args.hidden_dim, 
        args.num_layers,
        dropout=args.dropout,
        use_type_aware=args.use_type_aware,
        k=k_dim,
        num_tactics=num_tactics # <-- NEW
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    # Loss (FIXED)
    criterion = ProofSearchRankingLoss(
        margin=args.margin,
        hard_negative_weight=args.hard_neg_weight
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
        betas=(0.9, 0.98)
    )
    
    # FIXED: Use ReduceLROnPlateau (simpler, more robust for GNNs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize Hit@1
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    # Training
    best_val_hit1 = 0
    best_val_mrr = 0
    patience_counter = 0
    
    print("\nTraining...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        current_lr = get_lr(optimizer)

        valid_indices = curriculum_scheduler.get_batch_indices(epoch, args.epochs)
        sampler = SubsetRandomSampler(valid_indices)
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            sampler=sampler,
            num_workers=4 # Add workers for efficiency
        )
        num_curriculum_samples = len(valid_indices)
        total_samples = len(train_ds)
        print(f"  [Curriculum Info] Epoch {epoch}: Using {num_curriculum_samples}/{total_samples} samples.")
        if num_curriculum_samples == 0:
             print("  [Curriculum Warning] No samples selected by curriculum! Check difficulty scores.")
        # --- END NEW ---

        train_loss, train_val_loss, train_tac_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            args.grad_accum_steps,
            value_loss_weight=args.value_loss_weight, # Pass weight
            tactic_loss_weight=args.tactic_loss_weight # Pass weight
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device, 'val')
        
        print(f"[Epoch {epoch:3d}] "
              f"LR: {current_lr:.6f} | "
              f"Rank L: {train_loss:.4f} | "
              f"Val L: {train_val_loss:.4f} | "
              f"Tac L: {train_tac_loss:.4f} | " # <-- NEW
              f"Train Acc: {train_acc:.4f} | "
              f"Val Hit@1: {val_metrics['val_hit1']:.4f}")
        
        # FIXED: Step scheduler ONCE per epoch based on validation metric
        scheduler.step(val_metrics['val_hit1'])
        
        # Save best
        if val_metrics['val_hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit1']
            best_val_mrr = val_metrics['val_mrr']
            patience_counter = 0
            
            os.makedirs(args.exp_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_hit1': best_val_hit1,
                'val_mrr': best_val_mrr,
            }, f"{args.exp_dir}/best.pt")
            print(f"  → New best Hit@1: {best_val_hit1:.4f}, MRR: {best_val_mrr:.4f}")
        else:
            patience_counter += 1
            
            if patience_counter >= 25:
                print("Early stopping")
                break
    
    # Test
    checkpoint = torch.load(f"{args.exp_dir}/best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device, 'test')
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Hit@1:  {test_metrics['test_hit1']:.4f} ({test_metrics['test_hit1']*100:.1f}%)")
    print(f"Hit@3:  {test_metrics['test_hit3']:.4f} ({test_metrics['test_hit3']*100:.1f}%)")
    print(f"Hit@5:  {test_metrics['test_hit5']:.4f} ({test_metrics['test_hit5']*100:.1f}%)")
    print(f"Hit@10: {test_metrics['test_hit10']:.4f} ({test_metrics['test_hit10']*100:.1f}%)")
    print(f"MRR:    {test_metrics['test_mrr']:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_hit1': best_val_hit1,
        'best_val_mrr': best_val_mrr,
        'config': vars(args)
    }
    
    with open(f"{args.exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.exp_dir}/results.json")


if __name__ == '__main__':
    main()



