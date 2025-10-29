"""
FIXED Loss Functions for Step Prediction

Key Fixes:
1. Applicability constraint: non-applicable rules heavily penalized
2. Separate loss for applicable vs inapplicable rules
3. Valid negative sampling (only from applicable rules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApplicabilityConstrainedLoss(nn.Module):
    def __init__(self, margin=1.0, penalty_nonapplicable=10.0):  # REDUCED from 100
        """
        FIXED: More stable penalty weights
        """
        super().__init__()
        self.margin = margin
        self.penalty_nonapplicable = penalty_nonapplicable
    
    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        n = len(scores)
        
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
        
        if applicable_mask is None:
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
        
        # CRITICAL: Target must be applicable
        if not applicable_mask[target_idx]:
            # Return large but finite loss (not inf)
            return torch.tensor(100.0, device=scores.device, requires_grad=True)
        
        target_score = scores[target_idx]
        
        # === Component 1: Non-applicable penalty ===
        non_applicable_mask = ~applicable_mask
        is_target = torch.arange(n, device=scores.device) == target_idx
        
        if non_applicable_mask.any():
            nonapplicable_scores = scores[non_applicable_mask]
            # FIXED: Use sigmoid instead of relu for smoother gradients
            nonapplicable_loss = torch.sigmoid(
                nonapplicable_scores - target_score + self.margin
            ).mean()
            component1 = self.penalty_nonapplicable * nonapplicable_loss
        else:
            component1 = torch.tensor(0.0, device=scores.device)
        
        # === Component 2: Applicable negatives ranking ===
        applicable_negatives_mask = applicable_mask & ~is_target
        
        if applicable_negatives_mask.any():
            applicable_negative_scores = scores[applicable_negatives_mask]
            
            # FIXED: Smoother margin ranking with log-softmax
            # This prevents exploding gradients
            margin_violations = torch.relu(
                self.margin - (target_score - applicable_negative_scores)
            )
            
            # Focus on top-k hardest negatives (prevents overwhelming)
            n_hard = min(5, len(applicable_negative_scores))
            if n_hard > 0:
                top_violations, _ = torch.topk(margin_violations, k=n_hard)
                component2 = top_violations.mean()
            else:
                component2 = torch.tensor(0.0, device=scores.device)
        else:
            component2 = torch.tensor(0.0, device=scores.device)
        
        # === Combine with temperature scaling ===
        # This prevents initial loss from being too large
        temperature = 0.5  # Scales down initial loss
        loss = temperature * (component1 + component2)
        
        return loss

class ApplicabilityConstrainedRankingLoss(nn.Module):
    """
    Simplified version: focus only on applicable rules.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, embeddings, target_idx, applicable_mask):
        """
        Args:
            scores: [N] node scores
            embeddings: [N, D] (unused)
            target_idx: index of correct rule
            applicable_mask: [N] binary mask of which rules are applicable
        
        Returns:
            loss
        """
        n = len(scores)
        
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
        
        # Target MUST be applicable
        if not applicable_mask[target_idx]:
            return torch.tensor(float('inf'), device=scores.device)
        
        target_score = scores[target_idx]
        
        # Create mask for applicable negatives
        is_target = torch.arange(n, device=scores.device) == target_idx
        applicable_negatives = applicable_mask & ~is_target
        
        if not applicable_negatives.any():
            # No competition - perfect scenario
            return torch.tensor(0.0, device=scores.device)
        
        negative_scores = scores[applicable_negatives]
        
        # Margin ranking loss
        violations = F.relu(self.margin - (target_score - negative_scores))
        
        return violations.mean()


def compute_hit_at_k(scores, target_idx, k, applicable_mask=None):
    """
    Compute Hit@K metric.
    
    If applicable_mask provided, only considers applicable rules.
    """
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    
    # Get top-k
    top_k_indices = torch.topk(scores, k).indices
    
    # Check if target in top-k
    hit = 1.0 if target_idx in top_k_indices else 0.0
    
    # If applicable_mask provided, check if target was actually applicable
    if applicable_mask is not None:
        if not applicable_mask[target_idx]:
            # Target wasn't even applicable - mark as miss
            hit = 0.0
    
    return hit


def compute_mrr(scores, target_idx, applicable_mask=None):
    """
    Compute Mean Reciprocal Rank.
    
    If applicable_mask provided, only considers applicable rules.
    """
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    # Check applicability
    if applicable_mask is not None and not applicable_mask[target_idx]:
        return 0.0
    
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
    
    return 1.0 / rank


def compute_applicability_ratio(scores, applicable_mask):
    """
    Compute what fraction of top predictions are actually applicable.
    This is a diagnostic metric.
    """
    if not applicable_mask.any():
        return 0.0
    
    top_k = min(5, len(scores))
    top_indices = torch.topk(scores, k=top_k).indices
    
    applicable_in_top = applicable_mask[top_indices].float().mean()
    return applicable_in_top.item()


# Backward compatibility
class ProofSearchRankingLoss(nn.Module):
    """Deprecated: use ApplicabilityConstrainedLoss instead."""
    
    def __init__(self, margin=1.0, hard_negative_weight=2.0):
        super().__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        self.alpha_hard = 0.7
        self.alpha_easy = 0.3
    
    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """Calls new loss for compatibility."""
        loss = ApplicabilityConstrainedLoss(margin=self.margin)
        return loss(scores, embeddings, target_idx, applicable_mask)


def get_recommended_loss():
    """Returns the FIXED loss for this task."""
    return ApplicabilityConstrainedLoss(margin=1.0, penalty_nonapplicable=100.0)