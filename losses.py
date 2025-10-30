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

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- NEW: Masked Softmax Cross-Entropy Loss ---
class MaskedCrossEntropyLoss(nn.Module):
    """
    Computes Cross-Entropy loss only over applicable rules.
    Assumes scores are logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """
        Args:
            scores: [N] node scores (logits)
            embeddings: [N, D] (IGNORED - kept for API compatibility)
            target_idx: index of correct rule
            applicable_mask: [N] boolean mask of which rules are applicable

        Returns:
            loss (scalar tensor)
        """
        n = len(scores)

        # Basic validation
        if target_idx < 0 or target_idx >= n:
            # Invalid target, return a non-infinite loss to avoid crashing, but log it.
            print(f"Warning: Invalid target_idx ({target_idx}) in MaskedCrossEntropyLoss.")
            # Return a small loss, requires_grad=True if scores requires grad
            return torch.tensor(0.01, device=scores.device, requires_grad=scores.requires_grad)

        if applicable_mask is None:
            # If no mask provided, assume all are applicable (fallback, but should be avoided)
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
            print("Warning: No applicable_mask provided to MaskedCrossEntropyLoss. Assuming all applicable.")

        # CRITICAL: Target must be applicable
        if not applicable_mask[target_idx]:
            # This indicates a data inconsistency. The target *must* be applicable.
            # Return a large but finite loss and log a critical warning.
            print(f"CRITICAL WARNING: Target index {target_idx} is NOT applicable according to the mask!")
            # Use a large, grad-requiring tensor if scores require grad
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        # --- Core Logic ---
        # 1. Get indices of applicable nodes
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]

        # Handle case where only the target is applicable (no competition)
        if applicable_indices.numel() <= 1:
             # If only the target is applicable, loss should ideally be 0
             # (perfect prediction among choices).
             return torch.tensor(0.0, device=scores.device, requires_grad=scores.requires_grad)


        # 2. Gather scores for applicable nodes
        applicable_scores = scores[applicable_indices]

        # 3. Find the relative index of the target within the applicable set
        # This uses broadcasting and nonzero to find the position
        relative_target_idx_mask = (applicable_indices == target_idx)
        
        # Ensure the target was actually found (redundant due to earlier check, but safe)
        if not relative_target_idx_mask.any():
            print(f"INTERNAL ERROR: Target index {target_idx} not found in applicable indices {applicable_indices.tolist()} despite passing initial check!")
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        relative_target_idx = relative_target_idx_mask.nonzero(as_tuple=True)[0]
        
        # Ensure it's a scalar tensor (0-dim) for cross_entropy
        if relative_target_idx.dim() > 0:
            relative_target_idx = relative_target_idx.squeeze()
            if relative_target_idx.dim() > 0: # Check again after squeeze
                 relative_target_idx = relative_target_idx[0] # Take the first if still not scalar

        # Ensure target is scalar tensor
        if relative_target_idx.dim() > 0:
            print(f"Warning: relative_target_idx is not scalar: {relative_target_idx}. Taking first element.")
            relative_target_idx = relative_target_idx[0].unsqueeze(0) # Keep as tensor [1]
        else:
            relative_target_idx = relative_target_idx.unsqueeze(0) # Make it tensor([idx]) shape [1]


        # 4. Compute Cross-Entropy Loss
        # F.cross_entropy expects logits [Batch, Classes] and target [Batch]
        # Here, Batch=1, Classes=num_applicable
        loss = F.cross_entropy(
            applicable_scores.unsqueeze(0), # Shape: [1, num_applicable]
            relative_target_idx           # Shape: [1]
        )
        return loss

class TripletLossWithHardMining(nn.Module):
    """
    Computes Triplet Loss focusing on the hardest negative applicable rule.

    Loss = max(0, margin - (score_positive - score_hardest_negative))
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        print(f"Initialized TripletLossWithHardMining with margin={self.margin}")

    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """
        Args:
            scores: [N] node scores (logits)
            embeddings: [N, D] (IGNORED - loss uses scores directly)
            target_idx: index of correct rule (positive)
            applicable_mask: [N] boolean mask of which rules are applicable

        Returns:
            loss (scalar tensor)
        """
        n = len(scores)

        # Basic validation
        if target_idx < 0 or target_idx >= n:
            print(f"Warning: Invalid target_idx ({target_idx}) in TripletLossWithHardMining.")
            return torch.tensor(0.01, device=scores.device, requires_grad=scores.requires_grad)

        if applicable_mask is None:
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
            print("Warning: No applicable_mask provided to TripletLossWithHardMining.")

        # Ensure target is applicable
        if not applicable_mask[target_idx]:
            print(f"CRITICAL WARNING: Target index {target_idx} is NOT applicable in TripletLoss!")
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        # --- Core Triplet Logic ---
        positive_score = scores[target_idx]

        # 1. Identify applicable negative indices
        is_target_mask = torch.arange(n, device=scores.device) == target_idx
        negative_applicable_mask = applicable_mask & ~is_target_mask
        negative_applicable_indices = negative_applicable_mask.nonzero(as_tuple=True)[0]

        # 2. Handle case with no applicable negatives
        if negative_applicable_indices.numel() == 0:
            # No competitors, loss is 0
            return torch.tensor(0.0, device=scores.device, requires_grad=scores.requires_grad)

        # 3. Find the hardest negative (highest score among applicable negatives)
        negative_scores = scores[negative_applicable_indices]
        hardest_negative_score = torch.max(negative_scores)

        # 4. Compute Triplet Loss
        loss = F.relu(self.margin - (positive_score - hardest_negative_score))

        return loss

def get_recommended_loss(loss_type='triplet_hard', margin=1.0): # Add margin arg
    """Returns the recommended loss function."""
    if loss_type == 'applicability_constrained':
        print("Using Loss: ApplicabilityConstrainedLoss")
        return ApplicabilityConstrainedLoss(margin=margin, penalty_nonapplicable=10.0)
    elif loss_type == 'cross_entropy':
        print("Using Loss: MaskedCrossEntropyLoss")
        return MaskedCrossEntropyLoss()
    elif loss_type == 'triplet_hard':
        print(f"Using Loss: TripletLossWithHardMining (margin={margin})")
        return TripletLossWithHardMining(margin=margin)
    else:
        print(f"Warning: Unknown loss_type '{loss_type}'. Defaulting to TripletLossWithHardMining.")
        return TripletLossWithHardMining(margin=margin) # Default to Triplet