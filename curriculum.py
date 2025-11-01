
import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SetToSetCurriculumScheduler:
    """
    Progressive curriculum learning with set-to-set strategy.
    Groups proofs by difficulty and progressively increases challenge.
    """
    
    def __init__(self, total_epochs: int, dataset_metadata: Dict):
        self.total_epochs = total_epochs
        self.metadata = dataset_metadata
        
        # Group by difficulty
        self.groups = self._group_by_difficulty()
        self.current_epoch = 0
    
    def _group_by_difficulty(self) -> Dict:
        """Group instances by difficulty level"""
        groups = {
            'easy': [],
            'medium': [],
            'hard': [],
            'very_hard': []
        }
        
        for inst_id, meta in self.metadata.items():
            diff = meta.get('difficulty', 'medium')
            groups[diff].append({
                'id': inst_id,
                'proof_length': meta.get('proof_length', 0),
                'num_nodes': meta.get('num_nodes', 0),
                'num_rules': meta.get('num_rules', 0)
            })
        
        return groups
    
    def get_phase_config(self, epoch: int) -> Dict:
        """
        Get curriculum phase configuration for this epoch.
        Returns difficulty sets, depth limits, and sampling weights.
        """
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        
        if progress < 0.25:
            return {
                'phase': 'Foundation',
                'description': 'Learn on easy proofs (depth ≤ 3)',
                'include_difficulties': ['easy'],
                'sampling_weights': {'easy': 1.0},
                'max_proof_length': 3,
                'max_step_depth': 3,
                'confidence_threshold': 0.5,
            }
        
        elif progress < 0.50:
            return {
                'phase': 'Consolidation',
                'description': 'Mix easy (40%) + medium (60%)',
                'include_difficulties': ['easy', 'medium'],
                'sampling_weights': {'easy': 0.4, 'medium': 0.6},
                'max_proof_length': 5,
                'max_step_depth': 5,
                'confidence_threshold': 0.6,
            }
        
        elif progress < 0.75:
            return {
                'phase': 'Challenge',
                'description': 'Medium (50%) + hard (50%)',
                'include_difficulties': ['medium', 'hard'],
                'sampling_weights': {'medium': 0.5, 'hard': 0.5},
                'max_proof_length': 8,
                'max_step_depth': 8,
                'confidence_threshold': 0.7,
            }
        
        else:
            return {
                'phase': 'Mastery',
                'description': 'All difficulties (balanced)',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.1, 
                    'medium': 0.35, 
                    'hard': 0.4, 
                    'very_hard': 0.15
                },
                'max_proof_length': 15,
                'max_step_depth': 15,
                'confidence_threshold': 0.8,
            }
    
    def filter_batch(self, batch, epoch: int):
        """
        Filter batch to respect curriculum constraints.
        Remove samples that exceed difficulty limits for current epoch.
        """
        config = self.get_phase_config(epoch)
        max_length = config['max_proof_length']
        max_depth = config['max_step_depth']
        
        if hasattr(batch, 'proof_lengths'):
            length_mask = torch.tensor(
                [l <= max_length for l in batch.proof_lengths],
                dtype=torch.bool
            )
        else:
            length_mask = torch.ones(batch.num_graphs, dtype=torch.bool)
        
        if hasattr(batch, 'step_numbers'):
            depth_mask = batch.step_numbers <= max_depth
        else:
            depth_mask = torch.ones(batch.num_nodes, dtype=torch.bool)
        
        return length_mask.all() and depth_mask.any()
    
    def get_loss_weight(self, epoch: int, sample_difficulty: str, 
                       step_idx: int, proof_length: int) -> float:
        """
        Get loss weight for this sample based on curriculum.
        Easier samples weighted higher early on.
        """
        config = self.get_phase_config(epoch)
        
        # Weight based on difficulty sampling
        if sample_difficulty in config['sampling_weights']:
            w = config['sampling_weights'][sample_difficulty]
        else:
            w = 0.0  # Exclude this difficulty
        
        # Boost weight for samples matching curriculum
        if w > 0:
            # Progress through proof: early steps easier
            step_progress = step_idx / max(proof_length, 1)
            if step_progress > 0.7:  # Last 30% of proof
                w *= 0.7  # Down-weight harder steps early in training
            else:
                w *= 1.0
        
        return w
    
    def get_epoch_stats(self, epoch: int) -> str:
        """Log curriculum progress"""
        config = self.get_phase_config(epoch)
        return (
            f"Epoch {epoch}: {config['phase']} - "
            f"{config['description']} "
            f"(max_length={config['max_proof_length']}, "
            f"max_depth={config['max_step_depth']})"
        )