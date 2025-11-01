from losses import FocalApplicabilityLoss
from model import GatedPathwayFusion
from temporal_encoder import CausalProofTemporalEncoder

import torch

def test_gated_fusion_final():
    """Verify the final corrected version"""
    
    print("\n" + "="*80)
    print("TESTING FINAL CORRECTED GATED PATHWAY FUSION")
    print("="*80 + "\n")
    
    hidden_dim = 64
    batch_size = 10
    num_pathways = 3
    
    # Initialize
    fusion = GatedPathwayFusion(hidden_dim, num_pathways)
    
    # Create test inputs
    h1 = torch.randn(batch_size, hidden_dim)
    h2 = torch.randn(batch_size, hidden_dim)
    h3 = torch.randn(batch_size, hidden_dim)
    
    print("Input shapes:")
    print(f"  h1: {h1.shape}")
    print(f"  h2: {h2.shape}")
    print(f"  h3: {h3.shape}\n")
    
    # Forward pass
    try:
        output, gate_stats = fusion([h1, h2, h3])
        print("✅ Forward pass successful\n")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify output
    print("Output verification:")
    print(f"  Output shape: {output.shape}")
    
    if output.shape != (batch_size, hidden_dim):
        print(f"  ❌ Shape mismatch! Expected ({batch_size}, {hidden_dim})")
        return False
    print(f"  ✅ Output shape correct\n")
    
    # Check for NaNs
    if torch.isnan(output).any():
        print(f"  ❌ NaN in output!")
        return False
    print(f"  ✅ No NaN values\n")
    
    # Gate statistics
    print("Gate statistics:")
    
    importance_weights = gate_stats['importance_weights']
    print(f"  Importance weights: {importance_weights.tolist()}")
    print(f"  Sum: {importance_weights.sum().item():.4f}")
    
    if not torch.allclose(importance_weights.sum(), torch.tensor(1.0), atol=1e-5):
        print(f"  ❌ Weights don't sum to 1!")
        return False
    print(f"  ✅ Weights sum to 1.0\n")
    
    # Pathway contributions
    print("Pathway contributions:")
    contributions = gate_stats['pathway_contribution']
    for i, contrib in enumerate(contributions):
        print(f"  Pathway {i}: {contrib:.6f}")
    
    # Check no NaNs in stats
    for i, contrib in enumerate(contributions):
        if isinstance(contrib, float) and (contrib != contrib or contrib == float('inf')):
            print(f"  ❌ Invalid contribution for pathway {i}: {contrib}")
            return False
    print(f"  ✅ All contributions valid\n")
    
    # Gate variance
    print("Gate variance per pathway:")
    for i, var in enumerate(gate_stats['gate_variance']):
        print(f"  Pathway {i}: {var:.6f}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - GATED FUSION FINAL CORRECTED")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_gated_fusion_final()
    if not success:
        exit(1)