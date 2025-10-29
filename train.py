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
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler

from dataset import StepPredictionDataset, create_split
from losses import compute_hit_at_k, ProofSearchRankingLoss
import sys
sys.path.insert(0, '.')
from model import get_model
import re



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
                 base_difficulty: float = 0.3, 
                 max_difficulty: float = 1.0):
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


def train_epoch(model, loader, optimizer, criterion, device, epoch, grad_accum_steps=1,
                value_loss_weight=0.1, tactic_loss_weight=0.1): # <-- NEW Args    
    """
    FIXED: Scheduler no longer stepped here (moved to main loop).
    """
    model.train()
    
    total_loss = 0
    correct_preds = 0
    n_samples = 0

    value_loss_fn = nn.MSELoss()
    total_value_loss = 0

    tactic_loss_fn = nn.CrossEntropyLoss() # <-- NEW
    total_tactic_loss = 0 # <-- NEW

    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask,
            batch.step_numbers, batch.eigvecs, batch.eigvals,
            batch.edge_attr, batch.batch
        )

        
        batch_loss = 0
        batch_value_loss = 0
        batch_tactic_loss = 0 # <-- NEW
        batch_size = batch.num_graphs
        
        tactic_targets = batch.tactic_target.squeeze()
        tac_loss = tactic_loss_fn(tactic_logits, tactic_targets)


        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            target_idx = batch.y[i].item()
            

            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            ranking_loss = criterion(graph_scores, graph_embeddings, target_idx) 
            graph_value = value[i]
            target_value = batch.value_target[i]
            val_loss = value_loss_fn(graph_value, target_value)
            # --- END NEW ---

            if not (torch.isnan(ranking_loss) or torch.isinf(ranking_loss)):
                # --- MODIFIED: Combined Loss ---
                loss = (ranking_loss +
                       (value_loss_weight * val_loss))
                graph_loss = (ranking_loss + 
                            value_loss_weight * val_loss + 
                            tactic_loss_weight * tac_loss / batch_size)  # Divide by batch_size
                batch_loss += graph_loss
                batch_value_loss += val_loss.item()
                # --- END MODIFIED ---
                
                if graph_scores.argmax().item() == target_idx:
                    correct_preds += 1

        
        if batch_size > 0:
            avg_node_loss = batch_loss / batch_size
            combined_batch_loss = avg_node_loss + (tactic_loss_weight * tac_loss)   
            final_batch_loss = combined_batch_loss / grad_accum_steps
            final_batch_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # FIXED: Increased to 1.0 (from 0.5)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += avg_node_loss.item()
            total_value_loss += (batch_value_loss / batch_size)
            total_tactic_loss += tac_loss.item() # <-- NEW
            n_samples += batch_size
    
    avg_loss = total_loss / max(len(loader), 1)
    avg_value_loss = total_value_loss / max(len(loader), 1)
    avg_tactic_loss = total_tactic_loss / max(len(loader), 1) # <-- NEW
    accuracy = correct_preds / max(n_samples, 1)
    
    # Return all losses for logging
    return avg_loss, avg_value_loss, avg_tactic_loss, accuracy # <-- MODIFIED

@torch.no_grad()
def evaluate(model, loader, criterion, device, split='val'):
    """Enhanced evaluation with more metrics."""
    model.eval()
    
    losses = []
    hit1, hit3, hit5, hit10 = [], [], [], []
    ranks = []
    value_loss_fn = nn.MSELoss()
    value_losses = []
    tactic_loss_fn = nn.CrossEntropyLoss() # <-- NEW
    tactic_losses = [] # <-- NEW

    
    for batch in loader:
        batch = batch.to(device)
        
        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask,
            batch.step_numbers, batch.eigvecs, batch.eigvals,
            batch.edge_attr, batch.batch
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
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            loss = criterion(graph_scores, graph_embeddings, target_idx)
            losses.append(loss.item())

            val_loss = value_loss_fn(value[i], batch.value_target[i])
            value_losses.append(val_loss.item())
            
            # Metrics
            hit1.append(compute_hit_at_k(graph_scores, target_idx, 1))
            hit3.append(compute_hit_at_k(graph_scores, target_idx, 3))
            hit5.append(compute_hit_at_k(graph_scores, target_idx, 5))
            hit10.append(compute_hit_at_k(graph_scores, target_idx, 10))
            
            # MRR
            sorted_indices = torch.argsort(graph_scores, descending=True)
            rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
    
    mrr = sum(ranks) / max(len(ranks), 1)
    
    return {
        f'{split}_loss': sum(losses) / max(len(losses), 1),
        f'{split}_value_loss': sum(value_losses) / max(len(value_losses), 1), # <-- NEW
        f'{split}_tactic_loss': sum(tactic_losses) / max(len(tactic_losses), 1), # <-- NEW
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
    print(f"Loss: MarginRankingLoss (FIXED: 70% hard, 30% easy)")
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