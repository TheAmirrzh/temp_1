#!/usr/bin/env python3
"""
Test Suite for Critical Fixes (Issues 4 & 5)

This module specifically validates the corrected logic for:
1.  **Applicability Mask:** Ensures that rules whose conclusions are
    already known are correctly marked as NOT applicable.
2.  **Proof Simulator:** Ensures that node features are correctly
    recomputed after a simulation step, allowing the *next*
    rule to become applicable.

Usage:
  python test_critical_fixes.py
"""

import unittest
import torch
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# --- Import corrected project modules ---
# (Assumes fixes have been applied to these files)
try:
    from dataset import StepPredictionDataset, compute_applicable_rules_for_step
    from search_agent import ProofSimulator
    from model import get_model, TacticGuidedGNN
    from torch_geometric.data import Data
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure all project .py files (dataset.py, search_agent.py, model.py) are in the Python path.")
    exit(1)


class TestApplicabilityMaskFix(unittest.TestCase):
    """
    Validates Issue 4: Corrupt Applicability Mask in dataset.py
    
    This test confirms that the corrected `compute_applicable_rules_for_step`
    function correctly identifies a rule as *NOT applicable* if its
    head atom (conclusion) is already in the set of known facts.
    """
    
    def setUp(self):
        print("\n" + "="*70)
        print(" [TEST] Corrected Applicability Mask (Issue 4)")
        print("="*70)
        
        # --- Mock Graph Structure ---
        # Facts: A (initial), B (initial)
        # Rules: r1: A -> C
        #        r2: B -> D
        #        r3: C -> E
        # Proof: [step 0] r1 derives C
        #        [step 1] r2 derives D
        #        [step 2] r3 derives E
        #
        # We will test applicability at step_idx = 1
        # At this point, known_facts = {A, B, C}
        
        self.nodes = [
            {"nid": "fA", "type": "fact", "atom": "A", "is_initial": True},   # 0
            {"nid": "fB", "type": "fact", "atom": "B", "is_initial": True},   # 1
            {"nid": "r1", "type": "rule", "body_atoms": ["A"], "head_atom": "C"}, # 2
            {"nid": "r2", "type": "rule", "body_atoms": ["B"], "head_atom": "D"}, # 3
            {"nid": "r3", "type": "rule", "body_atoms": ["C"], "head_atom": "E"}, # 4
            {"nid": "fC", "type": "fact", "atom": "C", "is_initial": False},  # 5
            {"nid": "fD", "type": "fact", "atom": "D", "is_initial": False},  # 6
            {"nid": "fE", "type": "fact", "atom": "E", "is_initial": False}   # 7
        ]
        
        self.edges = [
            {"src": "fA", "dst": "r1", "etype": "body"},
            {"src": "fB", "dst": "r2", "etype": "body"},
            {"src": "fC", "dst": "r3", "etype": "body"},
            {"src": "r1", "dst": "fC", "etype": "head"},
            {"src": "r2", "dst": "fD", "etype": "head"},
            {"src": "r3", "dst": "fE", "etype": "head"},
        ]
        
        self.proof_steps = [
            {"derived_node": "fC", "used_rule": "r1"}, # step 0
            {"derived_node": "fD", "used_rule": "r2"}, # step 1
            {"derived_node": "fE", "used_rule": "r3"}  # step 2
        ]
    
    def test_applicability_logic(self):
        print("Testing applicability logic at step_idx = 1")
        
        # At step_idx = 1, the proof has only executed step 0.
        # Known facts = {A, B} (initial) + {C} (from step 0)
        step_idx = 1
        
        print("  Known facts should be: {A, B, C}")
        
        # Run the corrected function
        applicable_mask, known_facts = compute_applicable_rules_for_step(
            self.nodes,
            self.edges,
            step_idx,
            self.proof_steps
        )
        
        print(f"  Function reported known facts: {known_facts}")
        self.assertEqual(known_facts, {"A", "B", "C"})
        
        print(f"  Applicable mask result: {applicable_mask.tolist()}")
        
        # --- Assertions ---
        
        # Rule r1 (idx 2): A -> C
        # Premises {A} are met. Head {C} IS known.
        # This rule should be FALSE (not applicable).
        print("  Checking Rule r1 (A -> C)...")
        self.assertFalse(
            applicable_mask[2], 
            "FIX FAILED: Rule r1 (A -> C) should be FALSE (head 'C' is already known)"
        )
        print("    ✓ PASSED: Rule r1 correctly marked as NOT applicable.")
        
        # Rule r2 (idx 3): B -> D
        # Premises {B} are met. Head {D} is NOT known.
        # This rule should be TRUE (applicable).
        print("  Checking Rule r2 (B -> D)...")
        self.assertTrue(
            applicable_mask[3],
            "Rule r2 (B -> D) should be TRUE (premises met, head unknown)"
        )
        print("    ✓ PASSED: Rule r2 correctly marked as applicable.")

        # Rule r3 (idx 4): C -> E
        # Premises {C} are met. Head {E} is NOT known.
        # This rule should be TRUE (applicable).
        print("  Checking Rule r3 (C -> E)...")
        self.assertTrue(
            applicable_mask[4],
            "Rule r3 (C -> E) should be TRUE (premises met, head unknown)"
        )
        print("    ✓ PASSED: Rule r3 correctly marked as applicable.")


class TestProofSimulatorFix(unittest.TestCase):
    """
    Validates Issue 5: Broken Proof Simulator in search_agent.py
    
    This test confirms that the corrected `ProofSimulator.apply_rule`
    function correctly recomputes node features (`next_data.x`).
    
    We test this by:
    1. Applying rule 1.
    2. Asserting that the features for rule 2 are updated.
    3. Asserting that `get_applicable_rules_indices` *now* sees rule 2
       as applicable, which would fail without the feature update.
    """
    
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*70)
        print(" [TEST] Corrected Proof Simulator (Issue 5)")
        print("="*70)
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.instance_file = Path(cls.temp_dir) / "test_sim_instance.json"
        
        # --- Mock 2-Step Instance ---
        # Facts: A (init), B (init)
        # Rules: r1: A -> C        (idx=2)
        #        r2: (B, C) -> D  (idx=3)
        # Goal:  D
        
        cls.DUMMY_INSTANCE = {
            "id": "test_sim_instance",
            "nodes": [
                {"nid": "fA", "type": "fact", "atom": "A", "is_initial": True},    # 0
                {"nid": "fB", "type": "fact", "atom": "B", "is_initial": True},    # 1
                {"nid": "r1", "type": "rule", "body_atoms": ["A"], "head_atom": "C"},  # 2
                {"nid": "r2", "type": "rule", "body_atoms": ["B", "C"], "head_atom": "D"}, # 3
                {"nid": "fC", "type": "fact", "atom": "C", "is_initial": False},   # 4
                {"nid": "fD", "type": "fact", "atom": "D", "is_initial": False}    # 5 (Goal)
            ],
            "edges": [
                {"src": "fA", "dst": "r1", "etype": "body"},
                {"src": "fB", "dst": "r2", "etype": "body"},
                {"src": "fC", "dst": "r2", "etype": "body"},
                {"src": "r1", "dst": "fC", "etype": "head"},
                {"src": "r2", "dst": "fD", "etype": "head"}
            ],
            "proof_steps": [
                {"step_id": 0, "derived_node": "fC", "used_rule": "r1", "premises": ["fA"]},
                {"step_id": 1, "derived_node": "fD", "used_rule": "r2", "premises": ["fB", "fC"]}
            ],
            "goal": "D",
            "metadata": {"proof_length": 2, "n_nodes": 6, "n_rules": 2}
        }
        
        # Write the dummy instance file
        with open(cls.instance_file, 'w') as f:
            json.dump(cls.DUMMY_INSTANCE, f)
            
        print(f"  Created dummy instance: {cls.instance_file}")
        
        # Set up a minimal dataset and model for the simulator
        cls.dataset = StepPredictionDataset([str(cls.instance_file)], k_dim=16)
        cls.model = get_model(in_dim=22, hidden_dim=32, k=16, num_tactics=6)
        cls.simulator = ProofSimulator(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        if Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
        print(f"\n  Cleaned up temp dir: {cls.temp_dir}")
        
    def test_01_full_proof_simulation(self):
        """
        Tests the full simulation loop from step 0 to the goal,
        verifying feature recomputation at each step.
        """
        
        # --- Define node indices for clarity ---
        rule1_idx = 2  # r1: A -> C
        rule2_idx = 3  # r2: (B, C) -> D
        fact_C_idx = 4 # fC
        goal_fact_D_idx = 5 # fD

        # ==========================================
        #  PHASE 1: Test State 0 -> State 1
        # ==========================================
        print("\n--- Testing State 0 -> State 1 ---")
        print("  Getting initial state (step 0)...")
        initial_state = self.simulator.get_initial_state(self.DUMMY_INSTANCE)
        self.assertIsNotNone(initial_state)
        
        x_before = initial_state.x.clone()
        
        # Check initial applicable rules
        initial_applicable = self.simulator.get_applicable_rules_indices(initial_state)
        print(f"  Applicable rules at step 0: {initial_applicable}")
        self.assertIn(rule1_idx, initial_applicable)
        self.assertNotIn(rule2_idx, initial_applicable, "Rule 2 should NOT be applicable yet")

        print(f"  Applying rule {rule1_idx} (A -> C)...")
        next_state = self.simulator.apply_rule(initial_state, rule1_idx)
        self.assertIsNotNone(next_state)
        
        x_after = next_state.x
        
        # --- Assert Feature Recomputation ---
        print("  Verifying feature recomputation...")

        # 1. Check [3] (is_derived) for the new fact 'fC'
        print(f"    Checking feature [3] (is_derived) for fact 'C' (idx {fact_C_idx})...")
        self.assertEqual(x_before[fact_C_idx, 3].item(), 0.0, "Fact C should not be derived at start")
        self.assertEqual(x_after[fact_C_idx, 3].item(), 1.0, "FIX FAILED: Fact C 'is_derived' feature was not updated")
        print("      ✓ PASSED: 'is_derived' feature updated.")

        # 2. Check [21] (body_satisfied) for 'r2'
        #    Body of r2 is {B, C}. At step 0, only {B} was known (1/2 = 0.5)
        #    At step 1, {B, C} are known (2/2 = 1.0)
        print(f"    Checking feature [21] (body_satisfied) for rule 'r2' (idx {rule2_idx})...")
        self.assertAlmostEqual(
            x_before[rule2_idx, 21].item(), 
            0.5, 
            msg="r2 body_satisfied should be 0.5 at start"
        )
        self.assertAlmostEqual(
            x_after[rule2_idx, 21].item(), 
            1.0, 
            msg="FIX FAILED: r2 'body_satisfied' feature was not updated to 1.0"
        )
        print("      ✓ PASSED: 'body_satisfied' feature updated.")

        # 3. Check [19] (head_is_derived) for 'r1'
        #    Head of r1 is {C}. At step 0, {C} was not known.
        #    At step 1, {C} is now known.
        print(f"    Checking feature [19] (head_is_derived) for rule 'r1' (idx {rule1_idx})...")
        self.assertEqual(x_before[rule1_idx, 19].item(), 0.0, "r1 head_is_derived should be 0.0 at start")
        self.assertEqual(x_after[rule1_idx, 19].item(), 1.0, "FIX FAILED: r1 'head_is_derived' feature was not updated")
        print("      ✓ PASSED: 'head_is_derived' feature updated.")

        # ==========================================
        #  PHASE 2: Test State 1 -> State 2
        # ==========================================
        print("\n--- Testing State 1 -> State 2 ---")
        
        # This is the *most critical* test.
        # If features weren't updated, this check will fail.
        print("  Checking applicable rules in *new* state (State 1)...")
        applicable_rules_state_1 = self.simulator.get_applicable_rules_indices(next_state)
        print(f"  Applicable rules at step 1: {applicable_rules_state_1}")

        # r1 (A -> C) should now be INAPPLICABLE because its head {C} is known
        self.assertNotIn(
            rule1_idx, 
            applicable_rules_state_1,
            "FIX FAILED: Rule r1 should be INAPPLICABLE (head is known)"
        )
        
        # r2 (B, C -> D) should now be APPLICABLE
        self.assertIn(
            rule2_idx, 
            applicable_rules_state_1,
            "FIX FAILED: Rule r2 is NOT in applicable list. Feature recomputation failed."
        )
        print(f"    ✓ PASSED: Rule {rule2_idx} is now applicable, and r1 is not.")
        
        # --- Final Step: Apply rule 2 and check goal ---
        print(f"  Applying rule {rule2_idx} (B, C -> D)...")
        final_state = self.simulator.apply_rule(next_state, rule2_idx)
        self.assertIsNotNone(final_state)
        
        print("  Verifying goal state...")
        self.assertEqual(
            final_state.derived_mask[goal_fact_D_idx].item(), 
            1, 
            "Goal fact 'D' was not marked as derived"
        )
        self.assertTrue(
            self.simulator.is_goal_satisfied(final_state),
            "FIX FAILED: Simulator 'is_goal_satisfied' returned False for final state"
        )
        print("    ✓ PASSED: Goal state reached and verified.")
        print("\n" + "="*70)
        print(" [SUCCESS] Proof Simulator fix is working correctly.")
        print("="*70)

if __name__ == '__main__':
    unittest.main()