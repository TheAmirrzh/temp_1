from model import GatedPathwayFusion
import torch


def test_gated_fusion_fixed():
    """Verify the fixed gated fusion works correctly"""
    
    print("\n" + "="*80)
    print("TESTING FIXED GATED PATHWAY FUSION")
    print("="*80 + "\n")
    
    hidden_dim = 64
    batch_size = 10
    num_pathways = 3
    
    # Initialize module
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
        return False
    
    # Verify output shape
    print("Output verification:")
    print(f"  Output shape: {output.shape}")
    
    if output.shape != (batch_size, hidden_dim):
        print(f"  ❌ Output shape mismatch! Expected ({batch_size}, {hidden_dim})")
        return False
    print(f"  ✅ Output shape correct\n")
    
    # Verify no NaNs
    if torch.isnan(output).any():
        print(f"  ❌ NaN values in output!")
        return False
    print(f"  ✅ No NaN values\n")
    
    # Analyze gate statistics
    print("Gate statistics:")
    
    importance_weights = gate_stats['importance_weights']
    print(f"  Importance weights shape: {importance_weights.shape}")
    print(f"  Importance weights: {importance_weights.tolist()}")
    
    # Check weights sum to 1
    if not torch.allclose(importance_weights.sum(), torch.tensor(1.0), atol=1e-5):
        print(f"  ❌ Weights don't sum to 1: {importance_weights.sum().item()}")
        return False
    print(f"  ✅ Weights sum to 1.0\n")
    
    # Check for pathway collapse (one pathway dominates)
    max_weight = importance_weights.max().item()
    min_weight = importance_weights.min().item()
    
    print(f"  Max importance: {max_weight:.4f}")
    print(f"  Min importance: {min_weight:.4f}")
    print(f"  Ratio: {max_weight / (min_weight + 1e-8):.2f}x\n")
    
    if max_weight > 0.8:
        print(f"  ⚠️  WARNING: One pathway dominates ({max_weight:.2%})")
        print(f"     This may indicate the model prefers one information source")
    else:
        print(f"  ✅ Balanced pathway usage")
    
    # Gate variance
    print("\nGate variance per pathway:")
    for i, var in enumerate(gate_stats['gate_variance']):
        print(f"  Pathway {i}: {var:.6f}")
    
    # Pathway contribution
    print("\nPathway contribution:")
    for i, contrib in enumerate(gate_stats['pathway_contribution']):
        print(f"  Pathway {i}: {contrib:.6f}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - GATED FUSION FIXED")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    test_gated_fusion_fixed()