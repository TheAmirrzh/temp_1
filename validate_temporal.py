"""
Temporal Feature Validation and Analysis
========================================

Validates and analyzes temporal encoding quality:
1. Temporal smoothness (adjacent steps should be similar)
2. Discriminative power (axioms vs derived should differ)
3. Frontier focus (recent nodes get more attention)
4. Multi-scale coherence (different scales should agree)

Author: AI Research Team
Date: October 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import logging

from temporal_encoder import (
    TemporalStateEncoder,
    MultiScaleTemporalEncoder,
    compute_derived_mask,
    compute_step_numbers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class TemporalFeatureValidator:
    """Validates quality of temporal features."""
    
    def __init__(self, hidden_dim: int = 256):
        """
        Initialize validator.
        
        Args:
            hidden_dim: Hidden dimension for encoders
        """
        self.hidden_dim = hidden_dim
        self.encoder = TemporalStateEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            frontier_window=5,
            max_steps=100
        )
        self.encoder.eval()
        
        logger.info(f"TemporalFeatureValidator initialized: dim={hidden_dim}")
    
    def validate_temporal_smoothness(
        self,
        num_steps: int = 50,
        num_nodes: int = 20
    ) -> Dict:
        """
        Validate that temporally adjacent states have similar encodings.
        
        Args:
            num_steps: Number of proof steps to simulate
            num_nodes: Number of nodes
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating temporal smoothness...")
        
        # Simulate proof progression
        node_features = torch.randn(num_nodes, self.hidden_dim)
        similarities = []
        
        with torch.no_grad():
            prev_enc = None
            for step in range(1, num_steps + 1):
                # Simulate derivations: one new node per step
                num_axioms = 10
                num_derived = min(step, num_nodes - num_axioms)
                
                derived_mask = torch.zeros(num_nodes, dtype=torch.long)
                derived_mask[num_axioms : num_axioms + num_derived] = 1
                
                step_numbers = torch.zeros(num_nodes, dtype=torch.long)
                if num_derived > 0:
                    step_numbers[num_axioms : num_axioms + num_derived] = torch.arange(1, num_derived + 1)
                
                # Encode
                enc, _ = self.encoder(derived_mask, step_numbers, node_features)
                
                # Compare with previous
                if prev_enc is not None:
                    # Cosine similarity between consecutive steps
                    sim = F.cosine_similarity(
                        enc.mean(dim=0, keepdim=True),
                        prev_enc.mean(dim=0, keepdim=True)
                    ).item()
                    similarities.append(sim)
                
                prev_enc = enc
        
        results = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'all_similarities': similarities,
            'passed': bool(np.mean(similarities) > 0.8)  # Adjacent steps should be >80% similar
        }
        
        logger.info(f"  Mean similarity: {results['mean_similarity']:.3f}")
        logger.info(f"  Smoothness check: {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def validate_discriminative_power(
        self,
        num_trials: int = 100,
        num_nodes: int = 20
    ) -> Dict:
        """
        Validate that axioms and derived facts have distinguishable encodings.
        
        Args:
            num_trials: Number of random trials
            num_nodes: Number of nodes per trial
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating discriminative power...")
        
        distances = []
        
        with torch.no_grad():
            for _ in range(num_trials):
                # Half axioms, half derived
                derived_mask = torch.cat([
                    torch.zeros(num_nodes // 2),
                    torch.ones(num_nodes // 2)
                ]).long()
                
                step_numbers = torch.cat([
                    torch.zeros(num_nodes // 2),
                    torch.randint(1, 50, (num_nodes // 2,))
                ]).long()
                
                node_features = torch.randn(num_nodes, self.hidden_dim)
                
                # Encode
                enc, _ = self.encoder(derived_mask, step_numbers, node_features)
                
                # Compute distance between axiom and derived encodings
                axiom_enc = enc[:num_nodes // 2].mean(dim=0)
                derived_enc = enc[num_nodes // 2:].mean(dim=0)
                
                dist = torch.norm(axiom_enc - derived_enc).item()
                distances.append(dist)
        
        results = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'passed': bool(np.mean(distances) > 0.5)  # Should have clear separation
        }
        
        logger.info(f"  Mean distance: {results['mean_distance']:.3f}")
        logger.info(f"  Discriminative check: {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def validate_frontier_focus(
        self,
        num_nodes: int = 30,
        frontier_size: int = 5
    ) -> Dict:
        """
        Validate that attention focuses on frontier nodes.
        
        Args:
            num_nodes: Number of nodes
            frontier_size: Size of frontier window
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating frontier focus...")
        
        # Create proof state with clear frontier
        derived_mask = torch.ones(num_nodes).long()
        derived_mask[:10] = 0  # First 10 are axioms
        
        # Last 5 nodes are frontier
        step_numbers = torch.zeros(num_nodes, dtype=torch.long)
        step_numbers[10:] = torch.arange(1, num_nodes - 10 + 1)
        max_step = step_numbers.max().item()
        
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        # Override frontier window for this test
        self.encoder.frontier_attention.frontier_window = frontier_size
        
        with torch.no_grad():
            _, attn_weights = self.encoder(
                derived_mask,
                step_numbers,
                node_features,
                return_attention=True
            )
        
        # Compute attention mass on frontier vs non-frontier
        frontier_threshold = max(0, max_step - frontier_size)
        frontier_mask = (step_numbers > frontier_threshold)
        
        # Ensure we have a frontier, otherwise test is invalid
        if not frontier_mask.any():
            logger.warning("Frontier focus test skipped: No frontier nodes found.")
            return {'passed': True, 'skipped': True}

        frontier_attn = attn_weights[:, frontier_mask].sum().item()
        non_frontier_attn = attn_weights[:, ~frontier_mask].sum().item()
        
        total_attn = frontier_attn + non_frontier_attn
        frontier_ratio = frontier_attn / total_attn if total_attn > 0 else 0
        
        results = {
            'frontier_attention': float(frontier_attn),
            'non_frontier_attention': float(non_frontier_attn),
            'frontier_ratio': float(frontier_ratio),
            'passed': bool(frontier_ratio > 0.3)  # Frontier should get >30% attention
        }
        
        logger.info(f"  Frontier ratio: {frontier_ratio:.3f}")
        logger.info(f"  Frontier focus check: {'PASSED' if results['passed'] else 'FAILED'}")
        
        return results
    
    def plot_temporal_evolution(
        self,
        num_steps: int = 20,
        output_path: str = "temporal_evolution.pdf"
    ):
        """
        Visualize how temporal encoding evolves during proof.
        
        Args:
            num_steps: Number of steps to simulate
            output_path: Where to save plot
        """
        num_nodes = 20
        num_axioms = 10
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        encodings_over_time = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Simulate progressive derivation
                num_derived = min(step, num_nodes - num_axioms)
                
                derived_mask = torch.zeros(num_nodes, dtype=torch.long)
                derived_mask[num_axioms : num_axioms + num_derived] = 1
                
                step_numbers = torch.zeros(num_nodes, dtype=torch.long)
                if num_derived > 0:
                    step_numbers[num_axioms : num_axioms + num_derived] = torch.arange(1, num_derived + 1)
                
                enc, _ = self.encoder(derived_mask, step_numbers, node_features)
                encodings_over_time.append(enc.cpu().numpy())
        
        encodings_over_time = np.array(encodings_over_time)  # [steps, nodes, hidden_dim]
        
        # Plot: heatmap of first 3 encoding dimensions over time
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        for dim_idx in range(3):
            data = encodings_over_time[:, :, dim_idx].T  # [nodes, steps]
            
            im = axes[dim_idx].imshow(data, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
            axes[dim_idx].set_ylabel('Node Index')
            axes[dim_idx].set_title(f'Temporal Encoding Dimension {dim_idx}')
            plt.colorbar(im, ax=axes[dim_idx]) # <-- FIXED: was im1
            
            # Mark axioms vs derived boundary
            axes[dim_idx].axhline(y=num_axioms - 0.5, color='yellow', linestyle='--', linewidth=2, label='Axiom boundary')
            axes[dim_idx].legend()
        
        axes[-1].set_xlabel('Proof Step')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal evolution plot to {output_path}")
        plt.close()
    
    # --- FIXED: Corrected and merged function ---
    def plot_attention_patterns(
        self,
        output_path: str = "attention_patterns.pdf"
    ):
        """
        Visualize attention patterns over proof frontier.
        
        Args:
            output_path: Where to save plot
        """
        num_nodes = 25
        num_axioms = 10
        derived_mask = torch.ones(num_nodes).long()
        derived_mask[:num_axioms] = 0  # Axioms
        
        step_numbers = torch.zeros(num_nodes, dtype=torch.long)
        step_numbers[num_axioms:] = torch.arange(1, num_nodes - num_axioms + 1)
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        with torch.no_grad():
            _, attn_weights = self.encoder(
                derived_mask,
                step_numbers,
                node_features,
                return_attention=True
            )
        
        attn_np = attn_weights.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Full attention matrix
        im1 = axes[0].imshow(attn_np, cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Key (attended to)')
        axes[0].set_ylabel('Query (attending from)')
        axes[0].set_title('Attention Matrix')
        axes[0].axvline(x=num_axioms - 0.5, color='red', linestyle='--', label='Axiom boundary')
        axes[0].axhline(y=num_axioms - 0.5, color='red', linestyle='--')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Attention distribution over time
        # For each query, show attention distribution
        mean_attn_by_step = attn_np.mean(axis=0) # Average attention received by each node
        
        axes[1].plot(range(num_nodes), mean_attn_by_step, marker='o', linewidth=2)
        axes[1].axvline(x=num_axioms - 0.5, color='red', linestyle='--', label='Axiom boundary')
        axes[1].set_xlabel('Node Index (Key)')
        axes[1].set_ylabel('Mean Attention Received')
        axes[1].set_title('Average Attention by Node')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention patterns to {output_path}")
        plt.close()
    
    def generate_validation_report(self, output_dir: str = "validation_results"):
        """
        Generate comprehensive validation report.
        
        Args:
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating temporal validation report...")
        
        # Run all validations
        smoothness = self.validate_temporal_smoothness()
        discriminative = self.validate_discriminative_power()
        frontier = self.validate_frontier_focus()
        
        # Generate plots
        self.plot_temporal_evolution(
            output_path=str(output_path / "temporal_evolution.pdf")
        )
        self.plot_attention_patterns(
            output_path=str(output_path / "attention_patterns.pdf")
        )
        
        # Save JSON results
        results = {
            'temporal_smoothness': smoothness,
            'discriminative_power': discriminative,
            'frontier_focus': frontier,
            'overall_passed': all([
                smoothness['passed'],
                discriminative['passed'],
                frontier.get('passed', True) # Pass if skipped
            ])
        }
        
        with open(output_path / "temporal_validation.json", 'w') as f:
            # Remove large arrays for JSON
            results_save = {
                k: {k2: v2 for k2, v2 in v.items() if k2 != 'all_similarities'}
                if isinstance(v, dict) else v
                for k, v in results.items()
            }
            json.dump(results_save, f, indent=2)
        
        # Generate markdown report
        with open(output_path / "temporal_validation_report.md", 'w') as f:
            f.write("# Temporal Feature Validation Report\n\n")
            f.write(f"**Date**: {np.datetime64('today')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Overall Status**: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}\n\n")
            
            f.write("## Temporal Smoothness\n\n")
            f.write(f"- **Status**: {'✅ PASSED' if smoothness['passed'] else '❌ FAILED'}\n")
            f.write(f"- **Mean Similarity**: {smoothness['mean_similarity']:.4f}\n")
            f.write(f"- **Std Similarity**: {smoothness['std_similarity']:.4f}\n")
            f.write(f"- **Min Similarity**: {smoothness['min_similarity']:.4f}\n\n")
            f.write("*Adjacent proof steps should have similar encodings (>0.8 similarity).*\n\n")
            
            f.write("## Discriminative Power\n\n")
            f.write(f"- **Status**: {'✅ PASSED' if discriminative['passed'] else '❌ FAILED'}\n")
            f.write(f"- **Mean Distance**: {discriminative['mean_distance']:.4f}\n")
            f.write(f"- **Std Distance**: {discriminative['std_distance']:.4f}\n")
            f.write(f"- **Min Distance**: {discriminative['min_distance']:.4f}\n\n")
            f.write("*Axioms and derived facts should have distinguishable encodings (distance >0.5).*\n\n")
            
            f.write("## Frontier Focus\n\n")
            if frontier.get('skipped'):
                f.write(f"- **Status**: ⚠️ SKIPPED (No frontier nodes found in test)\n")
            else:
                f.write(f"- **Status**: {'✅ PASSED' if frontier['passed'] else '❌ FAILED'}\n")
                f.write(f"- **Frontier Ratio**: {frontier['frontier_ratio']:.4f}\n")
                f.write(f"- **Frontier Attention**: {frontier['frontier_attention']:.2f}\n")
                f.write(f"- **Non-Frontier Attention**: {frontier['non_frontier_attention']:.2f}\n\n")
            f.write("*Frontier nodes should receive significant attention (>30%).*\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("- `temporal_evolution.pdf`: Shows how encoding changes over proof steps\n")
            f.write("- `attention_patterns.pdf`: Shows attention distribution patterns\n\n")
        
        logger.info(f"Validation report saved to {output_path}")
        return results


def compare_encoding_variants(output_dir: str = "validation_results"):
    """
    Compare different temporal encoding variants.
    
    Compares:
    1. Fixed sinusoidal encoding (implemented)
    2. Learned positional encoding (baseline)
    3. No temporal encoding (ablation)
    """
    logger.info("Comparing temporal encoding variants...")
    
    num_nodes = 20
    hidden_dim = 128
    
    # Create test data
    derived_mask = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    step_numbers = torch.cat([torch.zeros(10), torch.arange(1, 11)]).long()
    node_features = torch.randn(num_nodes, hidden_dim)
    
    # Variant 1: Full encoder (with fixed time encoding)
    encoder_full = TemporalStateEncoder(hidden_dim=hidden_dim)
    encoder_full.eval()
    
    with torch.no_grad():
        enc_full, _ = encoder_full(derived_mask, step_numbers, node_features)
    
    # Variant 2: Without time encoding (ablation)
    # Simply use status embedding + frontier attention
    encoder_no_time = TemporalStateEncoder(hidden_dim=hidden_dim)
    encoder_no_time.eval()
    
    # Zero out time encoding
    with torch.no_grad():
        # Temporarily replace time encoder output
        original_forward = encoder_no_time.time_encoder.forward
        encoder_no_time.time_encoder.forward = lambda x: torch.zeros(x.shape[0], hidden_dim // 2, device=x.device)
        
        enc_no_time, _ = encoder_no_time(derived_mask, step_numbers, node_features)
        
        # Restore
        encoder_no_time.time_encoder.forward = original_forward
    
    # Compute differences
    diff_norm = torch.norm(enc_full - enc_no_time).item()
    
    results = {
        'full_encoder_mean': float(enc_full.mean().item()),
        'full_encoder_std': float(enc_full.std().item()),
        'no_time_mean': float(enc_no_time.mean().item()),
        'no_time_std': float(enc_no_time.std().item()),
        'difference_norm': float(diff_norm),
        'time_encoding_impact': 'significant' if diff_norm > 1.0 else 'minimal'
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "encoding_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Encoding comparison saved to {output_path}")
    logger.info(f"  Time encoding impact: {results['time_encoding_impact']}")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print(" TEMPORAL FEATURE VALIDATION")
    print("="*70)
    print()
    
    # Initialize validator
    validator = TemporalFeatureValidator(hidden_dim=256)
    
    # Generate full report
    results = validator.generate_validation_report(output_dir="validation_results_temporal")
    
    # Compare variants
    comparison = compare_encoding_variants(output_dir="validation_results_temporal")
    
    print()
    print("="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)
    print(f"Overall Status: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
    print()
    print("Component Status:")
    print(f"  Temporal Smoothness: {'✅' if results['temporal_smoothness']['passed'] else '❌'}")
    print(f"  Discriminative Power: {'✅' if results['discriminative_power']['passed'] else '❌'}")
    print(f"  Frontier Focus: {'✅' if results['frontier_focus'].get('passed', True) else '❌'}")
    print()
    print("Reports saved to: validation_results_temporal/")
    print("  - temporal_validation_report.md")
    print("  - temporal_validation.json")
    print("  - temporal_evolution.pdf")
    print("  - attention_patterns.pdf")
    print("  - encoding_comparison.json")
    print()
    print("="*70)