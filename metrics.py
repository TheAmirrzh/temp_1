
import numpy as np
import torch
from typing import Dict, List


class ProofMetricsCompute:
    """Compute comprehensive metrics for proof search"""
    
    @staticmethod
    def compute_metrics_dict(scores: torch.Tensor, target_idx: int,
                            applicable_mask: torch.Tensor, 
                            step_idx: int, proof_length: int) -> Dict:
        """
        Compute full metric suite for a single step.
        """
        metrics = {}
        
        if not applicable_mask[target_idx]:
            return {m: 0.0 for m in [
                'hit@1', 'hit@3', 'hit@5', 'mrr', 'ndcg', 'app_accuracy'
            ]}
        
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
        applicable_scores = scores[applicable_indices]
        
        # Rank of target among applicable
        sorted_idx = torch.argsort(applicable_scores, descending=True)
        rank = (sorted_idx == (applicable_indices == target_idx).nonzero(as_tuple=True)[0][0]).nonzero(as_tuple=True)[0]
        
        if len(rank) == 0:
            rank_val = len(applicable_indices) + 1
        else:
            rank_val = rank[0].item() + 1
        
        # Hit@K
        for k in [1, 3, 5, 10]:
            metrics[f'hit@{k}'] = 1.0 if rank_val <= k else 0.0
        
        # MRR
        metrics['mrr'] = 1.0 / rank_val
        
        # NDCG
        dcg = 1.0 / np.log2(rank_val + 1)
        idcg = 1.0
        metrics['ndcg'] = dcg / idcg
        
        # Applicability aware
        metrics['app_accuracy'] = 1.0 if rank_val == 1 else 0.0
        
        # Difficulty estimate
        metrics['num_applicable'] = len(applicable_indices)
        metrics['difficulty'] = np.log(max(len(applicable_indices), 1))
        
        # Depth-weighted (later steps easier)
        depth_weight = 1.0 / (1.0 + step_idx / max(proof_length, 1) / 10.0)
        metrics['mrr_weighted'] = metrics['mrr'] * depth_weight
        
        return metrics
    
    @staticmethod
    def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
        """Aggregate per-step metrics into epoch metrics"""
        if not all_metrics:
            return {}
        
        agg = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                agg[f'{key}_mean'] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
        
        return agg