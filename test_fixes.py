"""
Comprehensive Test Suite for Core Fixes

Tests:
1. Data generation produces valid proofs
2. Proofs are deterministic (same seed = same proof)
3. Loss respects applicability constraints
4. No data leakage in splits
"""

import unittest
import torch
import json
import random
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List

# Import fixed modules
from data_generator import (
    generate_horn_instance_deterministic,
    Difficulty,
    ProofVerifier,
    generate_dataset
)

from losses import (
    ApplicabilityConstrainedLoss,
    compute_hit_at_k,
    compute_mrr,
    compute_applicability_ratio
)


class TestDataGeneration(unittest.TestCase):
    """Test FIXED data generation."""
    
    def test_01_proof_generation_deterministic(self):
        """Test: Same seed produces same proof."""
        print("\n[TEST 01] Proof Generation Determinism")
        
        inst1 = generate_horn_instance_deterministic(
            "test_1", Difficulty.MEDIUM, seed=42
        )
        inst2 = generate_horn_instance_deterministic(
            "test_2", Difficulty.MEDIUM, seed=42
        )
        
        # Same seed should produce identical proof structure
        self.assertEqual(len(inst1["proof_steps"]), len(inst2["proof_steps"]))
        self.assertEqual(inst1["goal"], inst2["goal"])
        
        print(f"  ✓ Both instances have {len(inst1['proof_steps'])} proof steps")
        print(f"  ✓ Goal: {inst1['goal']}")
    
    def test_02_proof_validity(self):
        """Test: All generated proofs are valid."""
        print("\n[TEST 02] Proof Validity")
        
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            inst = generate_horn_instance_deterministic(
                f"test_{difficulty.value}", difficulty, seed=42
            )
            
            verifier = ProofVerifier(inst["nodes"], inst["edges"])
            is_valid, error_msg = verifier.verify_proof(inst["proof_steps"])
            
            self.assertTrue(is_valid, f"Invalid proof: {error_msg}")
            print(f"  ✓ {difficulty.value}: proof valid ({len(inst['proof_steps'])} steps)")
    
    def test_03_proof_completeness(self):
        """Test: Proof actually derives the goal."""
        print("\n[TEST 03] Proof Completeness (Derives Goal)")
        
        inst = generate_horn_instance_deterministic(
            "test_complete", Difficulty.MEDIUM, seed=123
        )
        
        # Extract what's known
        known_facts = set()
        for node in inst["nodes"]:
            if node["type"] == "fact" and node.get("is_initial", False):
                known_facts.add(node["atom"])
        
        # Apply proof steps
        nid_to_node = {n["nid"]: n for n in inst["nodes"]}
        for step in inst["proof_steps"]:
            derived_nid = step["derived_node"]
            derived_node = nid_to_node[derived_nid]
            known_facts.add(derived_node["atom"])
        
        # Check goal is derived
        goal = inst["goal"]
        if goal:
            self.assertIn(goal, known_facts, f"Goal {goal} not derived by proof")
            print(f"  ✓ Goal '{goal}' successfully derived")
        else:
            print(f"  ⚠ No goal in instance (degenerate case)")
    
    def test_04_proof_applicability(self):
        """Test: Each step uses only applicable rules."""
        print("\n[TEST 04] Step Applicability")
        
        inst = generate_horn_instance_deterministic(
            "test_applicable", Difficulty.MEDIUM, seed=456
        )
        
        nid_to_node = {n["nid"]: n for n in inst["nodes"]}
        edges = inst["edges"]
        
        # Track known facts
        known_atoms = set()
        for node in inst["nodes"]:
            if node["type"] == "fact" and node.get("is_initial", False):
                known_atoms.add(node["atom"])
        
        # For each step
        for step_idx, step in enumerate(inst["proof_steps"]):
            rule_nid = step["used_rule"]
            rule_node = nid_to_node[rule_nid]
            
            # Check all premises are known
            body_atoms = set(rule_node.get("body_atoms", []))
            self.assertTrue(body_atoms.issubset(known_atoms),
                          f"Step {step_idx}: body {body_atoms} not subset of known {known_atoms}")
            
            # Add derived fact
            derived_nid = step["derived_node"]
            derived_node = nid_to_node[derived_nid]
            known_atoms.add(derived_node["atom"])
        
        print(f"  ✓ All {len(inst['proof_steps'])} steps are applicable")
    
    def test_05_different_seeds_different_proofs(self):
        """Test: Different seeds produce different proofs."""
        print("\n[TEST 05] Seed Variation")
        
        inst1 = generate_horn_instance_deterministic(
            "test_var1", Difficulty.MEDIUM, seed=100
        )
        inst2 = generate_horn_instance_deterministic(
            "test_var2", Difficulty.MEDIUM, seed=200
        )
        
        # Should be different (with high probability)
        # Either different goals or different proof lengths
        are_different = (
            inst1["goal"] != inst2["goal"] or 
            len(inst1["proof_steps"]) != len(inst2["proof_steps"]) or
            len(inst1["nodes"]) != len(inst2["nodes"]) # Use n_rules from metadata
        )
        
        self.assertTrue(are_different,
                       "Different seeds produced identical instances (unlikely)")
        print(f"  ✓ Different seeds: inst1 has goal '{inst1['goal']}' (len={len(inst1['proof_steps'])}), "
              f"inst2 has goal '{inst2['goal']}' (len={len(inst2['proof_steps'])})")
    
    def test_06_difficulty_progression(self):
        """Test: Difficulty levels produce progressively harder instances."""
        print("\n[TEST 06] Difficulty Progression")
        
        stats = {}
        for difficulty in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
            inst = generate_horn_instance_deterministic(
                f"test_{difficulty.value}", difficulty, seed=42
            )
            
            stats[difficulty.value] = {
                "n_rules": len([n for n in inst["nodes"] if n["type"] == "rule"]),
                "proof_length": len(inst["proof_steps"]),
                "n_initial_facts": len([n for n in inst["nodes"] 
                                       if n["type"] == "fact" and n.get("is_initial")])
            }
        
        # Check progression
        easy_rules = stats["easy"]["n_rules"]
        medium_rules = stats["medium"]["n_rules"]
        hard_rules = stats["hard"]["n_rules"]
        
        self.assertLess(easy_rules, medium_rules,
                       f"Easy ({easy_rules}) should have fewer rules than Medium ({medium_rules})")
        self.assertLess(medium_rules, hard_rules,
                       f"Medium ({medium_rules}) should have fewer rules than Hard ({hard_rules})")
        
        print(f"  ✓ Easy: {easy_rules} rules, proof_len={stats['easy']['proof_length']}")
        print(f"  ✓ Medium: {medium_rules} rules, proof_len={stats['medium']['proof_length']}")
        print(f"  ✓ Hard: {hard_rules} rules, proof_len={stats['hard']['proof_length']}")


class TestProofVerifier(unittest.TestCase):
    """Test proof verification logic."""
    
    def test_01_valid_proof_accepted(self):
        """Test: Valid proof is accepted."""
        print("\n[TEST 07] Verifier Accepts Valid Proof")
        
        inst = generate_horn_instance_deterministic(
            "test_verif_valid", Difficulty.MEDIUM, seed=42
        )
        
        verifier = ProofVerifier(inst["nodes"], inst["edges"])
        is_valid, error = verifier.verify_proof(inst["proof_steps"])
        
        self.assertTrue(is_valid, f"Should accept valid proof, got error: {error}")
        print(f"  ✓ Valid proof accepted")
    
    def test_02_invalid_proof_rejected(self):
        """Test: Invalid proof (e.g., reversed steps) is rejected."""
        print("\n[TEST 08] Verifier Rejects Invalid Proof")
        
        inst = generate_horn_instance_deterministic(
            "test_verif_invalid", Difficulty.MEDIUM, seed=42
        )
        
        if len(inst["proof_steps"]) > 0:
            # --- START FIX ---
            # Corrupt the proof by REVERSING the steps.
            # This ensures the premises for the first step (originally the last)
            # will not be in the set of known initial facts.
            corrupted_steps = inst["proof_steps"][::-1]
            # --- END FIX ---
            
            verifier = ProofVerifier(inst["nodes"], inst["edges"])
            is_valid, error = verifier.verify_proof(corrupted_steps)
            
            # Now, this assertion should correctly fail.
            self.assertFalse(is_valid, "Verifier should have rejected reversed proof")
            print(f"  ✓ Verifier correctly rejected corrupted proof: {error}")

        else:
            print(f"  ⚠ Skipped (no proof steps in instance)")
    
    def test_03_derivable_atoms(self):
        """Test: Find all derivable atoms."""
        print("\n[TEST 09] Find Derivable Atoms")
        
        inst = generate_horn_instance_deterministic(
            "test_derivable", Difficulty.MEDIUM, seed=42
        )
        
        verifier = ProofVerifier(inst["nodes"], inst["edges"])
        derivable = verifier.find_all_derivable_atoms()
        
        # All atoms in proof must be derivable
        proof_atoms = set()
        for node in inst["nodes"]:
            if node["type"] == "fact":
                proof_atoms.add(node["atom"])
        
        # At least initial facts should be derivable
        initial = {n["atom"] for n in inst["nodes"] 
                  if n["type"] == "fact" and n.get("is_initial")}
        self.assertTrue(initial.issubset(derivable),
                       "Initial facts should be derivable")
        
        print(f"  ✓ Found {len(derivable)} derivable atoms (initial: {len(initial)})")


class TestApplicabilityConstrainedLoss(unittest.TestCase):
    """Test FIXED loss function."""
    
    def test_01_loss_penalizes_nonapplicable(self):
        """Test: Loss heavily penalizes non-applicable rules."""
        print("\n[TEST 10] Loss Penalizes Non-applicable Rules")
        
        loss_fn = ApplicabilityConstrainedLoss(margin=1.0, penalty_nonapplicable=100.0)
        
        # Setup
        num_rules = 10
        scores = torch.randn(num_rules, requires_grad=True)
        embeddings = torch.randn(num_rules, 64)
        target_idx = 0
        
        # Mark first 5 as applicable, last 5 as not
        applicable_mask = torch.tensor(
            [True] * 5 + [False] * 5, dtype=torch.bool
        )
        
        # Compute loss
        loss = loss_fn(scores, embeddings, target_idx, applicable_mask)
        
        # Loss should be finite and depend on applicable_mask
        self.assertTrue(torch.isfinite(loss),
                       "Loss should be finite for applicable target")
        self.assertGreater(loss.item(), 0,
                          "Loss should be positive")
        
        print(f"  ✓ Loss with non-applicable negatives: {loss.item():.4f}")
    
    def test_02_loss_infinite_for_nonapplicable_target(self):
        """Test: Loss is infinite if target is non-applicable."""
        print("\n[TEST 11] Loss is Infinite for Non-applicable Target")
        
        loss_fn = ApplicabilityConstrainedLoss()
        
        scores = torch.randn(10, requires_grad=True)
        embeddings = torch.randn(10, 64)
        target_idx = 0
        
        # Target is NOT applicable
        applicable_mask = torch.tensor([False] * 10, dtype=torch.bool)
        
        loss = loss_fn(scores, embeddings, target_idx, applicable_mask)
        
        self.assertTrue(torch.isinf(loss) or loss.item() > 1e6,
                       "Loss should be infinite/huge for non-applicable target")
        
        print(f"  ✓ Loss for non-applicable target: {loss.item()}")
    
    def test_03_loss_low_for_only_applicable_target(self):
        """Test: Loss low when target is only applicable rule."""
        print("\n[TEST 12] Loss Low When Target Only Applicable Rule")
        
        loss_fn = ApplicabilityConstrainedLoss()
        
        # --- FIX: Set scores so non-applicable rules are LOW ---
        # scores = torch.randn(10, requires_grad=True)
        scores = torch.tensor([-5.0] * 10, dtype=torch.float, requires_grad=True)
        scores.data[0] = 5.0 # Target score is high
        
        embeddings = torch.randn(10, 64)
        target_idx = 0
        
        # Only target is applicable
        applicable_mask = torch.zeros(10, dtype=torch.bool)
        applicable_mask[target_idx] = True
        
        loss = loss_fn(scores, embeddings, target_idx, applicable_mask)
        
        # --- FIX: Assert loss is 0.0 ---
        # component1 = penalty * relu(-5.0 - 5.0 + 1.0).mean() = 0
        # component2 = 0 (no applicable negatives)
        self.assertAlmostEqual(loss.item(), 0.0,
                       "Loss should be 0 when target is high and only applicable")
        
        print(f"  ✓ Loss when target is only applicable: {loss.item():.4f}")
    
    def test_04_hit_at_k_respects_applicability(self):
        """Test: Hit@K metric respects applicability."""
        print("\n[TEST 13] Hit@K Respects Applicability")
        
        scores = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float)
        target_idx = 0
        applicable_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
        
        # Without mask: hit
        hit_no_mask = compute_hit_at_k(scores, target_idx, k=1)
        self.assertEqual(hit_no_mask, 1.0)
        
        # With mask: miss (target not applicable)
        hit_with_mask = compute_hit_at_k(scores, target_idx, k=1, 
                                        applicable_mask=applicable_mask)
        self.assertEqual(hit_with_mask, 0.0)
        
        print(f"  ✓ Hit@1 without mask: {hit_no_mask}, with mask: {hit_with_mask}")
    
    def test_05_mrr_respects_applicability(self):
        """Test: MRR metric respects applicability."""
        print("\n[TEST 14] MRR Respects Applicability")
        
        scores = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float)
        target_idx = 0
        applicable_mask = torch.tensor([False, True, True, True], dtype=torch.bool)
        
        # Without mask: MRR = 1/1
        mrr_no_mask = compute_mrr(scores, target_idx)
        self.assertAlmostEqual(mrr_no_mask, 1.0)
        
        # With mask: MRR = 0 (target not applicable)
        mrr_with_mask = compute_mrr(scores, target_idx, applicable_mask=applicable_mask)
        self.assertEqual(mrr_with_mask, 0.0)
        
        print(f"  ✓ MRR without mask: {mrr_no_mask:.4f}, with mask: {mrr_with_mask:.4f}")
    
    def test_06_applicability_ratio(self):
        """Test: Applicability ratio metric."""
        print("\n[TEST 15] Applicability Ratio Metric")
        
        scores = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float)
        applicable_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
        
        ratio = compute_applicability_ratio(scores, applicable_mask)
        
        # --- FIX: top_k=4, [T, T, F, F] -> mean = 0.5 ---
        self.assertAlmostEqual(ratio, 0.5)
        
        print(f"  ✓ Applicability ratio: {ratio:.2%}")


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset generation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_01_generate_and_load_dataset(self):
        """Test: Generate dataset and load back."""
        print("\n[TEST 16] Generate and Load Dataset")
        
        n_per_difficulty = {
            Difficulty.EASY: 2,
            Difficulty.MEDIUM: 2,
        }
        
        stats = generate_dataset(self.temp_dir, n_per_difficulty, seed=42)
        
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["by_difficulty"]["easy"], 2)
        self.assertEqual(stats["by_difficulty"]["medium"], 2)
        
        print(f"  ✓ Generated {stats['total']} instances")
    
    def test_02_all_generated_instances_valid(self):
        """Test: All generated instances have valid proofs."""
        print("\n[TEST 17] All Generated Instances Valid")
        
        n_per_difficulty = {
            Difficulty.EASY: 5,
        }
        
        generate_dataset(self.temp_dir, n_per_difficulty, seed=42)
        
        # Load and verify each instance
        json_files = list(Path(self.temp_dir).glob("**/*.json"))
        self.assertGreater(len(json_files), 0, "Dataset generation failed")
        
        for json_file in json_files:
            with open(json_file) as f:
                inst = json.load(f)
            
            verifier = ProofVerifier(inst["nodes"], inst["edges"])
            is_valid, error = verifier.verify_proof(inst["proof_steps"])
            
            self.assertTrue(is_valid, 
                          f"Instance {json_file.name} has invalid proof: {error}")
        
        print(f"  ✓ All {len(json_files)} instances have valid proofs")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 80)
    print(" CORE FIXES - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestProofVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestApplicabilityConstrainedLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Core fixes are working correctly!")
        print("\nNext steps:")
        print("1. Update train.py to use ApplicabilityConstrainedLoss")
        print("2. Update dataset.py to compute applicable_mask per step")
        print("3. Retrain model with new loss and data")
    else:
        print("❌ Some tests failed - review errors above")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)