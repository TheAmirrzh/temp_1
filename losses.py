"""
FIXED Loss Functions for Step Prediction

Key Fix: ProofSearchRankingLoss now properly weights hard AND easy negatives
Previous bug: Easy negatives had weight 0.0, so model ignored 90% of negative examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProofSearchRankingLoss(nn.Module):
    """
    FIXED: Margin ranking loss with proper hard/easy negative weighting.
    
    Previous bug: With hard_negative_weight=2.0, easy negatives got weight 0.0
    Fix: Use focal loss style with 70% hard, 30% easy
    """
    def __init__(self, margin=1.0, hard_negative_weight=2.0):
        super().__init__()
        self.margin = margin
        # Parameter kept for backward compatibility, but not used in fixed version
        self.hard_negative_weight = hard_negative_weight 
        
        # FIXED: Use focal loss style weighting
        self.alpha_hard = 0.7
        self.alpha_easy = 0.3
        
    def forward(self, scores, embeddings, target_idx):
        """
        Args:
            scores (torch.Tensor): [N] scores for each node
            embeddings (torch.Tensor): [N, D] embeddings (for future use)
            target_idx (int): Index of the correct node
        """
        n = len(scores)
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
            
        target_score = scores[target_idx]
        
        # Create mask for negatives
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[target_idx] = False
        negative_scores = scores[mask]
        
        if len(negative_scores) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Margin ranking loss: want target_score > negative_scores + margin
        violations = F.relu(self.margin - (target_score - negative_scores))
        
        # Find hard negatives (top-5 highest-scoring negatives)
        sorted_negs, indices = torch.sort(negative_scores, descending=True)
        hard_indices = indices[:min(5, len(indices))]
        
        easy_mask = torch.ones(len(violations), dtype=torch.bool, device=scores.device)
        easy_mask[hard_indices] = False
        
        # FIXED: Focal loss style - 70% hard, 30% easy
        hard_loss = torch.tensor(0.0, device=scores.device)
        easy_loss = torch.tensor(0.0, device=scores.device)
        
        n_hard = len(hard_indices)
        n_easy = easy_mask.sum().item()

        if n_hard > 0:
            hard_loss = violations[hard_indices].mean()

        if n_easy > 0:
            easy_loss = violations[easy_mask].mean()

        # FIXED: Both hard and easy negatives contribute
        loss = self.alpha_hard * hard_loss + self.alpha_easy * easy_loss
        
        return loss


class SimpleMultiMarginLoss(nn.Module):
    """
    Multi-class margin loss for ranking.
    Alternative to ProofSearchRankingLoss if you want simpler approach.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores: torch.Tensor, target_idx: int) -> torch.Tensor:
        n_nodes = scores.shape[0]
        device = scores.device
        
        if target_idx >= n_nodes or target_idx < 0:
            return torch.tensor(0.01, device=device, requires_grad=True)
        
        target_score = scores[target_idx]
        
        # Compute violations for ALL other nodes
        losses = []
        for i in range(n_nodes):
            if i != target_idx:
                loss_i = F.relu(self.margin - (target_score - scores[i]))
                losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.01, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class FocalLossForRanking(nn.Module):
    """
    Focal Loss adapted for ranking.
    Down-weights easy examples, focuses on hard ones.
    """
    
    def __init__(self, gamma: float = 2.0, margin: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
    
    def forward(self, scores: torch.Tensor, target_idx: int) -> torch.Tensor:
        n_nodes = scores.shape[0]
        device = scores.device
        
        if target_idx >= n_nodes or target_idx < 0:
            return torch.tensor(0.01, device=device, requires_grad=True)
        
        target_score = scores[target_idx]
        
        losses = []
        for i in range(n_nodes):
            if i != target_idx:
                violation = F.relu(self.margin - (target_score - scores[i]))
                
                # Focal weighting
                p = torch.exp(-violation)
                focal_weight = (1 - p) ** self.gamma
                
                weighted_loss = focal_weight * violation
                losses.append(weighted_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.01, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


def compute_hit_at_k(scores: torch.Tensor, target_idx: int, k: int) -> float:
    """Compute Hit@K metric."""
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    
    top_k = torch.topk(scores, k).indices
    return 1.0 if target_idx in top_k else 0.0


def compute_mrr(scores: torch.Tensor, target_idx: int) -> float:
    """Compute Mean Reciprocal Rank."""
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
    
    return 1.0 / rank


def get_recommended_loss():
    """Returns the FIXED loss for this task."""
    return ProofSearchRankingLoss(margin=1.0, hard_negative_weight=2.0)


def get_multimargin_loss():
    """Alternative: PyTorch's built-in MultiMarginLoss."""
    return nn.MultiMarginLoss(
        p=1,
        margin=1.0,
        reduction='mean'
    )