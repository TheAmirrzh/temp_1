#!/usr/bin/env python3
"""
Diagnostic script to identify core training issues
"""

import torch
import numpy as np
from dataset import create_properly_split_dataloaders
from model import CriticallyFixedProofGNN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_data():
    """Diagnose data loading and structure"""
    logger.info("="*80)
    logger.info("DIAGNOSING DATA STRUCTURE")
    logger.info("="*80)
    
    train_loader, val_loader, test_loader = create_properly_split_dataloaders(
        './generated_data',
        spectral_dir='./spectral_cache',
        batch_size=2,
        seed=42
    )
    
    # Check first batch
    batch = next(iter(train_loader))
    
    logger.info(f"\n1. BATCH STRUCTURE:")
    logger.info(f"   Batch type: {type(batch)}")
    logger.info(f"   num_nodes: {batch.num_nodes}")
    logger.info(f"   num_graphs: {batch.num_graphs}")
    
    logger.info(f"\n2. KEY TENSOR SHAPES:")
    logger.info(f"   x shape: {batch.x.shape}")
    logger.info(f"   edge_index shape: {batch.edge_index.shape if batch.edge_index.numel() > 0 else 'empty'}")
    logger.info(f"   y (targets) shape: {batch.y.shape}")
    logger.info(f"   y values: {batch.y.tolist()}")
    logger.info(f"   batch.batch shape: {batch.batch.shape}")
    
    logger.info(f"\n3. TARGET VALIDITY CHECK:")
    all_valid = True
    for i, target_idx in enumerate(batch.y.tolist()):
        is_valid = 0 <= target_idx < batch.num_nodes
        is_applicable = (
            batch.applicable_mask[target_idx].item() 
            if hasattr(batch, 'applicable_mask') and 0 <= target_idx < len(batch.applicable_mask)
            else False
        )
        logger.info(f"   Graph {i}: target={target_idx}, valid={is_valid}, applicable={is_applicable}")
        if not is_valid:
            all_valid = False
    
    if not all_valid:
        logger.error("   ❌ CRITICAL: Invalid target indices!")
    
    logger.info(f"\n4. APPLICABLE MASK:")
    if hasattr(batch, 'applicable_mask'):
        logger.info(f"   Shape: {batch.applicable_mask.shape}")
        logger.info(f"   Num applicable: {batch.applicable_mask.sum().item()} / {len(batch.applicable_mask)}")
        logger.info(f"   Sample values: {batch.applicable_mask[:10].tolist()}")
    else:
        logger.warning("   ❌ MISSING applicable_mask!")
    
    logger.info(f"\n5. DERIVED MASK AND STEP NUMBERS:")
    if hasattr(batch, 'derived_mask'):
        logger.info(f"   derived_mask sum: {batch.derived_mask.sum().item()} / {len(batch.derived_mask)}")
    if hasattr(batch, 'step_numbers'):
        logger.info(f"   step_numbers range: [{batch.step_numbers.min().item()}, {batch.step_numbers.max().item()}]")
    
    logger.info(f"\n6. SPECTRAL FEATURES:")
    if hasattr(batch, 'eigvecs'):
        logger.info(f"   eigvecs shape: {batch.eigvecs.shape}")
        logger.info(f"   eigvals shape: {batch.eigvals.shape}")
        logger.info(f"   eigvals sample: {batch.eigvals[:5].tolist()}")
        logger.info(f"   eig_mask sum: {batch.eig_mask.sum().item()} / {len(batch.eig_mask)}")
    else:
        logger.warning("   ❌ MISSING spectral features!")
    
    return batch


def diagnose_model(batch, device='cpu'):
    """Diagnose model forward pass"""
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSING MODEL FORWARD PASS")
    logger.info("="*80)
    
    model = CriticallyFixedProofGNN(
        in_dim=22,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        k=16
    ).to(device)
    
    batch = batch.to(device)
    
    logger.info(f"\n1. FORWARD PASS:")
    try:
        scores, embeddings, value = model(batch)
        logger.info(f"   ✅ Forward pass succeeded")
    except Exception as e:
        logger.error(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.info(f"\n2. OUTPUT SHAPES AND STATS:")
    logger.info(f"   scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}], mean={scores.mean():.4f}")
    logger.info(f"   embeddings: {embeddings.shape}")
    logger.info(f"   value: {value.shape}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(scores).any() or torch.isnan(embeddings).any()
    has_inf = torch.isinf(scores).any() or torch.isinf(embeddings).any()
    if has_nan or has_inf:
        logger.error(f"   ❌ NaN/Inf detected! NaN={has_nan}, Inf={has_inf}")
    
    logger.info(f"\n3. PER-TARGET ANALYSIS:")
    for i, target_idx in enumerate(batch.y.tolist()):
        if 0 <= target_idx < len(scores):
            target_score = scores[target_idx].item()
            rank = (torch.argsort(scores, descending=True) == target_idx).nonzero(as_tuple=True)[0]
            rank_val = rank[0].item() + 1 if len(rank) > 0 else len(scores)
            top_score = scores.max().item()
            
            is_applicable = (
                batch.applicable_mask[target_idx].item()
                if hasattr(batch, 'applicable_mask')
                else False
            )
            
            logger.info(f"   Graph {i}:")
            logger.info(f"      Target={target_idx}, Score={target_score:.4f}, Rank={rank_val}/{len(scores)}")
            logger.info(f"      Top score={top_score:.4f}, Applicable={is_applicable}")
        else:
            logger.error(f"   Graph {i}: ❌ Target index out of bounds!")
    
    return scores, embeddings, value


if __name__ == "__main__":
    logger.info("STARTING DIAGNOSTIC\n")
    
    batch = diagnose_data()
    diagnose_model(batch, device='cpu')
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)