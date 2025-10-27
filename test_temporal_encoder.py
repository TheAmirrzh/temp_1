"""
Comprehensive Test Suite for Temporal State Encoder
===================================================

Tests all components of temporal encoding system:
1. Fixed time encoding
2. Proof frontier attention  
3. Complete temporal encoder
4. Multi-scale encoder
5. Utility functions
6. Integration with structural features

Author: AI Research Team
Date: October 2025
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from temporal_encoder import (
    FixedTimeEncoding,
    ProofFrontierAttention,
    TemporalStateEncoder,
    MultiScaleTemporalEncoder,
    compute_derived_mask,
    compute_step_numbers,
    compute_derivation_dependencies
)


class TestFixedTimeEncoding(unittest.TestCase):
    """Test fixed sinusoidal time encoding."""
    
    def setUp(self):
        self.d_model = 128
        self.max_steps = 100
        self.encoder = FixedTimeEncoding(self.d_model, self.max_steps)
    
    def test_output_shape(self):
        """Test output shape is correct."""
        step_numbers = torch.arange(10)
        encoding = self.encoder(step_numbers)
        self.assertEqual(encoding.shape, (10, self.d_model))
    
    def test_deterministic(self):
        """Test encoding is deterministic (not random)."""
        step_numbers = torch.tensor([5, 10, 15])
        enc1 = self.encoder(step_numbers)
        enc2 = self.encoder(step_numbers)
        torch.testing.assert_close(enc1, enc2)
    
    def test_different_steps_different_encodings(self):
        """Test different steps have different encodings."""
        step1 = torch.tensor([5])
        step2 = torch.tensor([10])
        enc1 = self.encoder(step1)
        enc2 = self.encoder(step2)
        self.assertFalse(torch.allclose(enc1, enc2))
    
    def test_adjacent_steps_similar(self):
        """Test adjacent steps have similar encodings."""
        step1 = torch.tensor([10])
        step2 = torch.tensor([11])
        enc1 = self.encoder(step1)
        enc2 = self.encoder(step2)
        
        # Cosine similarity should be high
        similarity = F.cosine_similarity(enc1, enc2, dim=-1)
        self.assertGreater(similarity.item(), 0.9)
    
    def test_range_bounded(self):
        """Test encoding values are in reasonable range."""
        step_numbers = torch.arange(50)
        encoding = self.encoder(step_numbers)
        
        # Sinusoidal functions bounded in [-1, 1]
        self.assertTrue(torch.all(encoding >= -1.1))
        self.assertTrue(torch.all(encoding <= 1.1))
    
    def test_batch_processing(self):
        """Test encoding works for batches."""
        batch_steps = torch.randint(0, self.max_steps, (100,))
        encoding = self.encoder(batch_steps)
        self.assertEqual(encoding.shape, (100, self.d_model))


class TestProofFrontierAttention(unittest.TestCase):
    """Test proof frontier attention mechanism."""
    
    def setUp(self):
        self.hidden_dim = 128
        self.frontier_window = 5
        self.attention = ProofFrontierAttention(
            self.hidden_dim,
            num_heads=4,
            frontier_window=self.frontier_window
        )
    
    def test_output_shape(self):
        """Test output shape matches input."""
        num_nodes = 20
        x = torch.randn(num_nodes, self.hidden_dim)
        step_numbers = torch.arange(num_nodes)
        
        output, attn_weights = self.attention(x, step_numbers, max_step=19)
        
        self.assertEqual(output.shape, (num_nodes, self.hidden_dim))
        self.assertEqual(attn_weights.shape, (num_nodes, num_nodes))
    
    def test_frontier_focus(self):
        """Test attention focuses on frontier nodes."""
        num_nodes = 20
        x = torch.randn(num_nodes, self.hidden_dim)
        
        # Nodes 15-19 are frontier (recent)
        step_numbers = torch.arange(num_nodes)
        max_step = 19
        
        output, attn_weights = self.attention(x, step_numbers, max_step)
        
        # Attention weights should be higher for frontier nodes (15-19)
        frontier_attn = attn_weights[:, 15:].sum()
        non_frontier_attn = attn_weights[:, :15].sum()
        
        # Frontier should get more attention
        self.assertGreater(frontier_attn, non_frontier_attn)
    
    def test_no_frontier_fallback(self):
        """Test fallback when no frontier nodes exist."""
        num_nodes = 10
        x = torch.randn(num_nodes, self.hidden_dim)
        
        # All nodes are old (step 0)
        step_numbers = torch.zeros(num_nodes, dtype=torch.long)
        
        output, attn_weights = self.attention(x, step_numbers, max_step=50)
        
        # Should still produce valid output
        self.assertEqual(output.shape, (num_nodes, self.hidden_dim))
    
    def test_gradient_flow(self):
        """Test gradients flow through attention."""
        num_nodes = 10
        x = torch.randn(num_nodes, self.hidden_dim, requires_grad=True)
        step_numbers = torch.arange(num_nodes)
        
        output, _ = self.attention(x, step_numbers, max_step=9)
        loss = output.mean()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestTemporalStateEncoder(unittest.TestCase):
    """Test complete temporal state encoder."""
    
    def setUp(self):
        self.hidden_dim = 128
        self.encoder = TemporalStateEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=4,
            frontier_window=5,
            max_steps=100
        )
    
    def test_output_shape(self):
        """Test output shape is correct."""
        num_nodes = 20
        derived_mask = torch.randint(0, 2, (num_nodes,))
        step_numbers = torch.randint(0, 50, (num_nodes,))
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        output, _ = self.encoder(derived_mask, step_numbers, node_features)
        
        self.assertEqual(output.shape, (num_nodes, self.hidden_dim))
    
    def test_axioms_vs_derived(self):
        """Test encoder distinguishes axioms from derived facts."""
        num_nodes = 20
        
        # First 10 are axioms (step 0)
        derived_mask = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        step_numbers = torch.cat([torch.zeros(10), torch.arange(1, 11)]).long()
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        output, _ = self.encoder(derived_mask, step_numbers, node_features)
        
        # Axioms vs derived should have different encodings
        axiom_enc = output[:10]
        derived_enc = output[10:]
        
        # Mean encoding should differ
        axiom_mean = axiom_enc.mean(dim=0)
        derived_mean = derived_enc.mean(dim=0)
        
        self.assertFalse(torch.allclose(axiom_mean, derived_mean, atol=0.1))
    
    def test_temporal_progression(self):
        """Test encoding changes with step progression."""
        num_nodes = 10
        derived_mask = torch.ones(num_nodes).long()
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        # Early steps
        step_numbers_early = torch.arange(1, 11)
        enc_early, _ = self.encoder(derived_mask, step_numbers_early, node_features)
        
        # Later steps
        step_numbers_late = torch.arange(41, 51)
        enc_late, _ = self.encoder(derived_mask, step_numbers_late, node_features)
        
        # Encodings should differ due to temporal progression
        self.assertFalse(torch.allclose(enc_early, enc_late, atol=0.1))
    
    def test_attention_output(self):
        """Test attention weights are returned when requested."""
        num_nodes = 20
        derived_mask = torch.randint(0, 2, (num_nodes,))
        step_numbers = torch.randint(0, 50, (num_nodes,))
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        output, attn = self.encoder(
            derived_mask, step_numbers, node_features,
            return_attention=True
        )
        
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (num_nodes, num_nodes))
    
    def test_gradient_flow(self):
        """Test gradients flow through entire encoder."""
        num_nodes = 10
        derived_mask = torch.randint(0, 2, (num_nodes,))
        step_numbers = torch.randint(0, 50, (num_nodes,))
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        self.encoder.train()
        output, _ = self.encoder(derived_mask, step_numbers, node_features)
        
        loss = output.mean()
        loss.backward()
        
        # Check all components have gradients
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
    
    def test_no_derived_nodes(self):
        """Test encoder handles case with no derived nodes."""
        num_nodes = 10
        derived_mask = torch.zeros(num_nodes).long()  # All axioms
        step_numbers = torch.zeros(num_nodes).long()
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        output, _ = self.encoder(derived_mask, step_numbers, node_features)
        
        self.assertEqual(output.shape, (num_nodes, self.hidden_dim))


class TestMultiScaleTemporalEncoder(unittest.TestCase):
    """Test multi-scale temporal encoder."""
    
    def setUp(self):
        self.hidden_dim = 128
        self.encoder = MultiScaleTemporalEncoder(
            hidden_dim=self.hidden_dim,
            num_scales=3,
            max_steps=100
        )
    
    def test_output_shape(self):
        """Test output shape is correct."""
        num_nodes = 20
        derived_mask = torch.randint(0, 2, (num_nodes,))
        step_numbers = torch.randint(0, 50, (num_nodes,))
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        output = self.encoder(derived_mask, step_numbers, node_features)
        
        self.assertEqual(output.shape, (num_nodes, self.hidden_dim))
    
    def test_multiple_scales(self):
        """Test encoder uses multiple temporal scales."""
        # Check that multiple scale encoders exist
        self.assertEqual(len(self.encoder.scale_encoders), 3)
        
        # Check different windows
        windows = self.encoder.temporal_windows
        self.assertEqual(len(windows), 3)
        self.assertTrue(windows[0] < windows[1] < windows[2])
    
    def test_gradient_flow(self):
        """Test gradients flow through all scales."""
        num_nodes = 10
        derived_mask = torch.randint(0, 2, (num_nodes,))
        step_numbers = torch.randint(0, 50, (num_nodes,))
        node_features = torch.randn(num_nodes, self.hidden_dim)
        
        self.encoder.train()
        output = self.encoder(derived_mask, step_numbers, node_features)
        
        loss = output.mean()
        loss.backward()
        
        # Check all scale encoders have gradients
        for i, scale_encoder in enumerate(self.encoder.scale_encoders):
            for name, param in scale_encoder.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, 
                                       f"No gradient for scale {i}, param {name}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for computing temporal features."""
    
    def test_compute_derived_mask(self):
        """Test derived mask computation."""
        proof_state = {
            'num_nodes': 10,
            'derivations': [(5, 1), (6, 2), (7, 3)]
        }
        
        mask = compute_derived_mask(proof_state, current_step=2)
        
        self.assertEqual(mask.shape, (10,))
        self.assertEqual(mask[5].item(), 1)  # Derived at step 1
        self.assertEqual(mask[6].item(), 1)  # Derived at step 2
        self.assertEqual(mask[7].item(), 0)  # Derived at step 3 (future)
        self.assertEqual(mask[0].item(), 0)  # Axiom
    
    def test_compute_step_numbers(self):
        """Test step number computation."""
        proof_state = {
            'num_nodes': 10,
            'derivations': [(5, 1), (6, 2), (7, 3)]
        }
        
        steps = compute_step_numbers(proof_state, current_step=2)
        
        self.assertEqual(steps.shape, (10,))
        self.assertEqual(steps[5].item(), 1)
        self.assertEqual(steps[6].item(), 2)
        self.assertEqual(steps[7].item(), 0)  # Future derivation
        self.assertEqual(steps[0].item(), 0)  # Axiom
    
    def test_compute_derivation_dependencies(self):
        """Test dependency matrix computation."""
        proof_state = {
            'num_nodes': 10,
            'dependencies': [
                (5, [0, 1], 1),
                (6, [1, 2], 2),
                (7, [5, 6], 3)
            ]
        }
        
        deps = compute_derivation_dependencies(proof_state, current_step=2)
        
        self.assertEqual(deps.shape, (10, 10))
        self.assertEqual(deps[5, 0].item(), 1.0)  # Node 5 uses node 0
        self.assertEqual(deps[5, 1].item(), 1.0)  # Node 5 uses node 1
        self.assertEqual(deps[6, 1].item(), 1.0)  # Node 6 uses node 1
        self.assertEqual(deps[7, 5].item(), 0.0)  # Future dependency
    
    def test_empty_proof_state(self):
        """Test utility functions handle empty proof states."""
        proof_state = {'num_nodes': 10}
        
        mask = compute_derived_mask(proof_state, current_step=5)
        steps = compute_step_numbers(proof_state, current_step=5)
        deps = compute_derivation_dependencies(proof_state, current_step=5)
        
        self.assertEqual(mask.sum().item(), 0)  # No derivations
        self.assertEqual(steps.sum().item(), 0)  # All zeros
        self.assertEqual(deps.sum().item(), 0)  # No dependencies


class TestIntegration(unittest.TestCase):
    """Integration tests combining all components."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from proof state to encoding."""
        # Create realistic proof state
        num_nodes = 20
        proof_state = {
            'num_nodes': num_nodes,
            'derivations': [(i, i-9) for i in range(10, 20)],
            'dependencies': [
                (10, [0, 1], 1),
                (11, [1, 2], 2),
                (12, [2, 3], 3)
            ]
        }
        current_step = 10
        
        # Compute temporal features
        derived_mask = compute_derived_mask(proof_state, current_step)
        step_numbers = compute_step_numbers(proof_state, current_step)
        
        # Simulate structural features
        node_features = torch.randn(num_nodes, 128)
        
        # Encode temporal state
        encoder = TemporalStateEncoder(hidden_dim=128)
        temporal_enc, attn = encoder(
            derived_mask, step_numbers, node_features,
            return_attention=True
        )
        
        # Verify outputs
        self.assertEqual(temporal_enc.shape, (num_nodes, 128))
        self.assertEqual(attn.shape, (num_nodes, num_nodes))
        
        # Verify axioms have different encoding than derived
        axiom_enc = temporal_enc[:10].mean(dim=0)
        derived_enc = temporal_enc[10:].mean(dim=0)
        self.assertFalse(torch.allclose(axiom_enc, derived_enc, atol=0.1))
    
    def test_temporal_dynamics(self):
        """Test temporal encoding captures proof dynamics."""
        num_nodes = 15
        hidden_dim = 128
        
        proof_state = {
            'num_nodes': num_nodes,
            'derivations': [(i, i-9) for i in range(10, 15)]
        }
        
        encoder = TemporalStateEncoder(hidden_dim=hidden_dim)
        node_features = torch.randn(num_nodes, hidden_dim)
        
        # Encode at step 3
        mask_3 = compute_derived_mask(proof_state, current_step=3)
        steps_3 = compute_step_numbers(proof_state, current_step=3)
        enc_3, _ = encoder(mask_3, steps_3, node_features)
        
        # Encode at step 5
        mask_5 = compute_derived_mask(proof_state, current_step=5)
        steps_5 = compute_step_numbers(proof_state, current_step=5)
        enc_5, _ = encoder(mask_5, steps_5, node_features)
        
        # Encodings should differ as proof progresses
        self.assertFalse(torch.allclose(enc_3, enc_5, atol=0.1))
        
        # Newly derived nodes should have different encoding
        newly_derived = (mask_5 == 1) & (mask_3 == 0)
        if newly_derived.any():
            new_enc = enc_5[newly_derived].mean(dim=0)
            old_enc = enc_3[mask_3 == 1].mean(dim=0) if (mask_3 == 1).any() else enc_3.mean(dim=0)
            self.assertFalse(torch.allclose(new_enc, old_enc, atol=0.1))


def run_all_tests():
    """Run all test suites."""
    print("="*70)
    print(" TEMPORAL ENCODER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFixedTimeEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestProofFrontierAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalStateEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiScaleTemporalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("="*70)
    print(" TEST SUITE SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✓ All tests passed! Temporal encoder ready for integration.")
    else:
        print("✗ Some tests failed. Please review errors above.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)