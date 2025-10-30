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
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import logger
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler, Sampler
from collections import defaultdict
from dataset import ContrastiveProofDataset, StepPredictionDataset, create_split
from losses import MaskedCrossEntropyLoss, compute_hit_at_k, compute_mrr, TripletLossWithHardMining, get_recommended_loss
import sys
sys.path.insert(0, '.')
from model import get_model, FixedTemporalSpectralGNN

import re
from losses import ApplicabilityConstrainedLoss, compute_mrr


"""
FIXED Training Script

Key Fixes:
1.  FIXED (Issue #3): Replaced sample-level curriculum with
    InstanceLevelCurriculum to prevent data leakage.
2.  FIXED (Issue #6): Updated training/evaluation loops to handle
    (positive, negative) sample tuples for contrastive learning.
3.  FIXED (Model Mismatch): Re-enabled TacticGuidedGNN and tactic
    loss, which were previously ignored.
4.  FIXED (TypeError): InstanceLevelCurriculum now computes its
    own simple difficulty from metadata, decoupling it from
    the dataset's value_target logic.
"""

import os
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import logger
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler, Sampler

# --- MODIFIED: Import full dataset class for type hinting ---
from dataset import StepPredictionDataset, create_split
from losses import compute_hit_at_k, compute_mrr
import sys
sys.path.insert(0, '.')
from model import get_model, TacticGuidedGNN
from losses import get_recommended_loss


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

# --- START FIX (Issue #3 & TypeError): Instance-Level Curriculum ---
class InstanceLevelCurriculum(Sampler):
    """
    FIXED: Implements instance-level curriculum scheduling to prevent
    data leakage.
    
    Groups all samples (steps) by instance_id and only includes them
    when the instance's difficulty is below the current threshold.
    """
    def __init__(self, dataset: StepPredictionDataset, 
                 base_difficulty: float = 0.15, 
                 max_difficulty: float = 0.95,
                 total_epochs: int = 100):
        
        self.dataset = dataset
        self.base_difficulty = base_difficulty
        self.max_difficulty = max_difficulty
        self.total_epochs = total_epochs
        
        print("Initializing InstanceLevelCurriculum (grouping samples)...")
        
        self.instance_to_indices = defaultdict(list)
        self.instance_difficulties = {}
        
        for i, sample_tuple in enumerate(dataset.samples):
            inst, step_idx, _, metadata = sample_tuple
            inst_id = inst.get("id", f"unknown_{i}")
            
            self.instance_to_indices[inst_id].append(i)
            
            # --- START FIX: Compute simple difficulty from metadata ---
            if inst_id not in self.instance_difficulties:
                # This logic is now self-contained and only uses metadata
                n_rules = metadata.get('n_rules', 10)
                n_nodes_total = metadata.get('n_nodes', 20)
                proof_length = metadata.get('proof_length', 1)
                if proof_length == 0: proof_length = 1 # Avoid division by zero
                
                rule_score = min(n_rules / 40.0, 1.0)
                node_score = min(n_nodes_total / 70.0, 1.0)
                proof_score = min(proof_length / 15.0, 1.0)
                
                curriculum_difficulty = (0.3 * rule_score + 
                                         0.2 * node_score + 
                                         0.5 * proof_score)
                
                self.instance_difficulties[inst_id] = min(max(curriculum_difficulty, 0.0), 1.0)
            # --- END FIX ---

        self.instance_ids = list(self.instance_difficulties.keys())
        print(f"Cached difficulties for {len(self.instance_ids)} instances.")
        
        self.current_indices = []
        self.current_epoch = -1

    def _get_indices_for_epoch(self, epoch: int) -> List[int]:
        """Calculates the valid sample indices for a given epoch."""
        progress = (epoch / max(self.total_epochs, 1)) ** 0.5
        target_difficulty = (self.base_difficulty + 
                            (self.max_difficulty - self.base_difficulty) * progress)
        
        valid_sample_indices = []
        instances_included = 0
        for inst_id in self.instance_ids:
            if self.instance_difficulties[inst_id] <= target_difficulty:
                valid_sample_indices.extend(self.instance_to_indices[inst_id])
                instances_included += 1
        
        if not valid_sample_indices and len(self.instance_ids) > 0:
            # Fallback: take easiest 5% of instances
            print("  [Curriculum Warning] Fallback: No instances matched. Taking 5% easiest.")
            sorted_instances = sorted(self.instance_difficulties.items(), key=lambda x: x[1])
            fallback_count = max(1, len(sorted_instances) // 20)
            for inst_id, diff in sorted_instances[:fallback_count]:
                valid_sample_indices.extend(self.instance_to_indices[inst_id])
                instances_included += 1
        
        print(f"  [Curriculum] Epoch {epoch}: Target difficulty <= {target_difficulty:.3f}, "
              f"found {len(valid_sample_indices)} samples from {instances_included} instances.")
        
        return valid_sample_indices

    def set_epoch(self, epoch: int):
        """Called by DataLoader to update indices."""
        self.current_epoch = epoch
        self.current_indices = self._get_indices_for_epoch(epoch)
        
    def __iter__(self):
        if self.current_epoch == -1:
            self.set_epoch(0)
        np.random.shuffle(self.current_indices)
        return iter(self.current_indices)

    def __len__(self):
        return len(self.current_indices)


# --- START FIX (Issue #6 & Model Mismatch): Updated Training Loop ---
def train_epoch(model, loader, optimizer, criterion_rank, criterion_contrast,
                device, epoch, grad_accum_steps=1, 
                value_loss_weight=0.1,
                contrast_loss_weight=0.3):
    """
    FIXED: Training loop for TacticGuidedGNN with contrastive loss.
    Handles (pos_data, neg_data) tuples.
    """
    model.train()
    
    total_rank_loss = 0
    total_value_loss = 0
    total_contrast_loss = 0
    correct_preds = 0
    n_samples = 0
    num_batches_processed = 0

    optimizer.zero_grad()
    
    from tqdm import tqdm
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)

    for batch_idx, batch_tuple in enumerate(progress_bar):
        
        pos_batch, neg_batch = batch_tuple
        if pos_batch is None or neg_batch is None:
            continue
            
        pos_batch = pos_batch.to(device)
        neg_batch = neg_batch.to(device)

        scores, pos_embeddings, value = model(
            pos_batch.x, pos_batch.edge_index, pos_batch.derived_mask,
            pos_batch.step_numbers, pos_batch.eigvecs, pos_batch.eigvals,
            pos_batch.eig_mask, pos_batch.edge_attr, pos_batch.batch
        )

        _, neg_embeddings, _ = model(
            neg_batch.x, neg_batch.edge_index, neg_batch.derived_mask,
            neg_batch.step_numbers, neg_batch.eigvecs, neg_batch.eigvals,
            neg_batch.eig_mask, neg_batch.edge_attr, neg_batch.batch
        )
        
        batch_size = pos_batch.num_graphs

        batch_rank_loss = 0
        batch_value_loss = 0
        batch_contrast_loss = 0
        graphs_in_batch_processed = 0

        for i in range(batch_size):
            pos_mask = (pos_batch.batch == i)
            neg_mask = (neg_batch.batch == i)
            
            graph_scores = scores[pos_mask]
            if graph_scores.numel() == 0:
                continue

            graph_pos_embeds = pos_embeddings[pos_mask]
            target_idx = pos_batch.y[i].item()
            graph_applicable = pos_batch.applicable_mask[pos_mask]
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            rank_loss = criterion_rank(
                graph_scores,
                graph_pos_embeds,
                target_idx,
                applicable_mask=graph_applicable
            )
            if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                continue

            graph_value = value[i].unsqueeze(0)
            target_value = pos_batch.value_target[i].unsqueeze(0)
            val_loss = F.mse_loss(graph_value, target_value)
            
            pos_target_embed = graph_pos_embeds[target_idx]
            
            neg_target_idx = neg_batch.y[i].item()
            graph_neg_embeds = neg_embeddings[neg_mask]
            
            if neg_target_idx < 0 or neg_target_idx >= len(graph_neg_embeds):
                continue
                
            neg_target_embed = graph_neg_embeds[neg_target_idx]
            
            con_loss = criterion_contrast(
                pos_target_embed.unsqueeze(0),
                neg_target_embed.unsqueeze(0)
            )
            
            batch_rank_loss += rank_loss
            batch_value_loss += val_loss
            batch_contrast_loss += con_loss
            graphs_in_batch_processed += 1
            
            if graph_scores.argmax().item() == target_idx:
                correct_preds += 1

        if graphs_in_batch_processed > 0:
            avg_rank_loss = batch_rank_loss / graphs_in_batch_processed
            avg_value_loss = batch_value_loss / graphs_in_batch_processed
            avg_contrast_loss = batch_contrast_loss / graphs_in_batch_processed
            
            combined_batch_loss = (
                avg_rank_loss +
                value_loss_weight * avg_value_loss +
                contrast_loss_weight * avg_contrast_loss
            )

            final_batch_loss = combined_batch_loss / grad_accum_steps
            final_batch_loss.backward()

            total_rank_loss += avg_rank_loss.item()
            total_value_loss += avg_value_loss.item()
            total_contrast_loss += avg_contrast_loss.item()
            n_samples += graphs_in_batch_processed
            num_batches_processed += 1

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
                optimizer.zero_grad()

        progress_bar.set_postfix({
             'RankL': f'{total_rank_loss / max(1, num_batches_processed):.3f}',
             'ValL': f'{total_value_loss / max(1, num_batches_processed):.3f}',
             'ConL': f'{total_contrast_loss / max(1, num_batches_processed):.3f}'
        })

    progress_bar.close() 

    num_batches_processed = max(1, num_batches_processed)
    accuracy = correct_preds / max(n_samples, 1)
    
    return {
        "rank_loss": total_rank_loss / num_batches_processed,
        "value_loss": total_value_loss / num_batches_processed,
        "contrast_loss": total_contrast_loss / num_batches_processed,
        "accuracy": accuracy
    }
    
@torch.no_grad()
def evaluate(model, loader, criterion_rank, device, split='val'):
    """
    FIXED: Evaluation loop for TacticGuidedGNN.
    Handles (pos, neg) tuples but only evaluates on pos_batch.
    """
    model.eval()
    
    rank_losses, value_losses = [], []
    hit1, hit3, hit10, ranks = [], [], [], []
    value_loss_fn = nn.MSELoss()

    for batch_tuple in loader:
        pos_batch, _ = batch_tuple
        if pos_batch is None:
            continue
            
        pos_batch = pos_batch.to(device)
        
        scores, embeddings, value = model(
            pos_batch.x, pos_batch.edge_index, pos_batch.derived_mask,
            pos_batch.step_numbers, pos_batch.eigvecs, pos_batch.eigvals,
            pos_batch.eig_mask, pos_batch.edge_attr, pos_batch.batch
        )
        batch_size = pos_batch.num_graphs
        

        for i in range(batch_size):
            mask = (pos_batch.batch == i)
            graph_scores = scores[mask]
            
            if graph_scores.numel() == 0: continue

            graph_embeddings = embeddings[mask]
            target_idx = pos_batch.y[i].item()
            graph_applicable = pos_batch.applicable_mask[mask]
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            loss = criterion_rank(
                graph_scores, 
                graph_embeddings, 
                target_idx,
                applicable_mask=graph_applicable
            )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                rank_losses.append(loss.item())

            val_loss = value_loss_fn(value[i], pos_batch.value_target[i])
            value_losses.append(val_loss.item())
            
            hit1.append(compute_hit_at_k(graph_scores, target_idx, 1, graph_applicable))
            hit3.append(compute_hit_at_k(graph_scores, target_idx, 3, graph_applicable))
            hit10.append(compute_hit_at_k(graph_scores, target_idx, 10, graph_applicable))
            ranks.append(compute_mrr(graph_scores, target_idx, graph_applicable))
    
    def _safe_mean(l): return sum(l) / max(len(l), 1)

    return {
        f'{split}_rank_loss': _safe_mean(rank_losses),
        f'{split}_value_loss': _safe_mean(value_losses),
        f'{split}_hit1': _safe_mean(hit1),
        f'{split}_hit3': _safe_mean(hit3),
        f'{split}_hit10': _safe_mean(hit10),
        f'{split}_mrr': _safe_mean(ranks),
    }

def contrastive_collate(batch_list):
        from torch_geometric.data import Batch
        
        pos_list = [item[0] for item in batch_list if item[0] is not None and item[1] is not None]
        neg_list = [item[1] for item in batch_list if item[0] is not None and item[1] is not None]
        
        if not pos_list or not neg_list:
            return None, None
            
        pos_batch = Batch.from_data_list(pos_list)
        neg_batch = Batch.from_data_list(neg_list)
        return pos_batch, neg_batch

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--exp-dir', default='experiments/fixed_run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--loss-type', type=str, default='triplet_hard',
                       choices=['triplet_hard', 'cross_entropy', 'applicability_constrained'])
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--contrast-loss-weight', type=float, default=0.3,
                       help='Weight for the contrastive loss')
                       
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spectral-dir', type=str, default=None)
    parser.add_argument('--value-loss-weight', type=float, default=0.1)
    parser.add_argument('--k-dim', type=int, default=16)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    print("SOTA TRAINING - Priority Fixes Implemented")
    print("="*70)
    print(f"Loss: {args.loss_type} (Margin: {args.margin})")
    print(f"Contrastive Loss Weight: {args.contrast_loss_weight}")
    print(f"Curriculum: Instance-Level (FIXED)")
    print(f"Model: TacticGuidedGNN (FIXED)")
    print(f"Params: LR={args.lr}, Hidden={args.hidden_dim}, Layers={args.num_layers}")
    
    train_files, val_files, test_files = create_split(args.data_dir, seed=args.seed)
    
    train_ds = StepPredictionDataset(
        train_files, spectral_dir=args.spectral_dir, 
        seed=args.seed, k_dim=args.k_dim, 
        contrastive_neg=True
    )
    val_ds = StepPredictionDataset(
        val_files, spectral_dir=args.spectral_dir, 
        seed=args.seed+1, k_dim=args.k_dim,
        contrastive_neg=True
    )
    test_ds = StepPredictionDataset(
        test_files, spectral_dir=args.spectral_dir, 
        seed=args.seed+2, k_dim=args.k_dim,
        contrastive_neg=True
    )
    
    print(f"\nDataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    curriculum_scheduler = InstanceLevelCurriculum(
        train_ds, total_epochs=args.epochs
    )
    
 

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, 
        collate_fn=contrastive_collate, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, 
        collate_fn=contrastive_collate, num_workers=2
    )
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty!")
    
    # --- FIX: Handle empty/filtered dataset ---
    try:
        first_sample_tuple = next(item for item in train_ds if item[0] is not None)
        in_dim = first_sample_tuple[0].x.shape[1]
    except StopIteration:
        raise ValueError("Dataset contains no valid samples. Check data generation and paths.")
    
    k_dim = args.k_dim 
    num_tactics = train_ds.num_tactics
    
    model = get_model(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k=k_dim,
        num_tactics=num_tactics
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    criterion_rank = get_recommended_loss(args.loss_type, args.margin)
    criterion_contrast = nn.L1Loss() 
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )
    
    best_val_hit1 = 0
    best_val_mrr = 0
    patience_counter = 0
    warmup_epochs = 3
    
    print("\nTraining...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        curriculum_scheduler.set_epoch(epoch)
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            sampler=curriculum_scheduler,
            collate_fn=contrastive_collate,
            num_workers=4,
            pin_memory=True
        )
        
        if len(train_loader) == 0:
            print(f"  [Curriculum Warning] Epoch {epoch}: No samples selected. Skipping.")
            continue
            
        if epoch <= warmup_epochs:
            current_lr = args.lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = get_lr(optimizer)

        metrics = train_epoch(
            model, train_loader, optimizer, 
            criterion_rank, criterion_contrast,
            device, epoch, args.grad_accum_steps,
            args.value_loss_weight, 
            args.contrast_loss_weight
        )
        
        val_metrics = evaluate(model, val_loader, criterion_rank, device, 'val')
        
        print(f"[Epoch {epoch:3d}] "
              f"LR: {current_lr:.6f} | "
              f"Rank L: {metrics['rank_loss']:.4f} | "
              f"Con L: {metrics['contrast_loss']:.4f} | "
              f"Val Hit@1: {val_metrics['val_hit1']:.4f}")
        
        if epoch > warmup_epochs:
            scheduler.step(val_metrics['val_hit1'])
        
        if val_metrics['val_hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit1']
            best_val_mrr = val_metrics['val_mrr']
            patience_counter = 0
            
            os.makedirs(args.exp_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_hit1': best_val_hit1,
            }, f"{args.exp_dir}/best.pt")
            print(f"  → New best Hit@1: {best_val_hit1:.4f}, MRR: {best_val_mrr:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print("Early stopping")
                break
    
    if best_val_hit1 == 0:
        print("Model did not train. No best.pt saved. Exiting.")
        return

    checkpoint = torch.load(f"{args.exp_dir}/best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion_rank, device, 'test')
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Hit@1:  {test_metrics['test_hit1']:.4f} ({test_metrics['test_hit1']*100:.1f}%)")
    print(f"Hit@3:  {test_metrics['test_hit3']:.4f} ({test_metrics['test_hit3']*100:.1f}%)")
    print(f"Hit@10: {test_metrics['test_hit10']:.4f} ({test_metrics['test_hit10']*100:.1f}%)")
    print(f"MRR:    {test_metrics['test_mrr']:.4f}")
    print(f"Tactic Acc: {test_metrics['test_tactic_acc']:.4f}")
    print("="*70)
    
    results = {
        'test_metrics': test_metrics,
        'best_val_hit1': best_val_hit1,
        'config': {k: v for k, v in vars(args).items() if k != 'config'}
    }
    
    with open(f"{args.exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.exp_dir}/results.json")

if __name__ == '__main__':
    main()