# IN: search_agent.py (NEW/MODIFIED)
"""
Inference-Time Proof Search Agent
Implements batched Beam Search (Phase 1 Goal) using the
trained Phase 2 GNN as a scoring function.
"""
import torch
from torch_geometric.data import Data, Batch
from model import TacticGuidedGNN # Assuming this is the final model
from dataset import StepPredictionDataset # Needed for simulation logic
from typing import List, Dict, Tuple, Optional
import copy

class ProofSimulator:
    """
    Simulates the application of a rule to update a proof state (Data object).
    Placeholder - Requires detailed implementation based on dataset features.
    """
    def __init__(self, dataset_instance: StepPredictionDataset):
        # Store necessary info or methods from the dataset
        self.dataset = dataset_instance
        print("⚠️ ProofSimulator initialized - Needs feature update logic implementation!")

    def get_initial_state(self, instance: Dict) -> Optional[Data]:
        """Creates the initial graph state (step 0)."""
        # Find the sample corresponding to step 0 for this instance
        for sample_tuple in self.dataset.samples:
            inst, step_idx, _, _ = sample_tuple
            if inst.get("id") == instance.get("id") and step_idx == 0:
                # Use dataset's __getitem__ to generate the initial Data object
                try:
                    sample_index = self.dataset.samples.index(sample_tuple)
                    initial_data = self.dataset[sample_index]
                    # Store necessary metadata for future steps if needed
                    initial_data.instance_data = instance # Store raw instance data
                    initial_data.current_step_simulated = 0
                    return initial_data
                except Exception as e:
                    print(f"Error creating initial state for {instance.get('id')}: {e}")
                    return None
        print(f"Warning: Could not find step 0 for instance {instance.get('id')}")
        return None


    def apply_rule(self, current_data: Data, rule_node_idx: int) -> Optional[Data]:
        """
        Applies a rule and returns the *next* state Data object.
        Placeholder - This is the core logic to implement.
        """
        print(f"Simulating application of rule node index: {rule_node_idx} at step {current_data.current_step_simulated + 1}")

        # --- CRITICAL IMPLEMENTATION NEEDED ---
        # 1. Deep copy the current_data object.
        next_data = current_data.clone() # Use .clone() for PyG Data objects

        # 2. Increment simulated step counter.
        next_data.current_step_simulated = current_data.current_step_simulated + 1
        current_step_sim = next_data.current_step_simulated

        # 3. Identify the derived fact (head node) from the rule_node_idx.
        #    Requires looking up the rule in the original instance data.
        instance = current_data.instance_data
        nodes = instance.get("nodes", [])
        edges = instance.get("edges", [])
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}

        if rule_node_idx >= len(nodes) or nodes[rule_node_idx]['type'] != 'rule':
             print(f"Error: Invalid rule node index {rule_node_idx}")
             return None

        rule_node = nodes[rule_node_idx]
        derived_fact_nid = None
        for edge in edges:
             if edge['src'] == rule_node['nid'] and edge['etype'] == 'head':
                 derived_fact_nid = edge['dst']
                 break

        if derived_fact_nid is None or derived_fact_nid not in id2idx:
             print(f"Error: Cannot find derived fact for rule {rule_node_idx}")
             return None
        derived_fact_idx = id2idx[derived_fact_nid]

        # 4. Update features in next_data:
        #    - next_data.derived_mask: Set derived_fact_idx to 1 (True).
        if hasattr(next_data, 'derived_mask') and derived_fact_idx < len(next_data.derived_mask):
            # Ensure derived_mask is writable
            if not next_data.derived_mask.is_contiguous():
                next_data.derived_mask = next_data.derived_mask.contiguous()
            next_data.derived_mask[derived_fact_idx] = 1

        #    - next_data.step_numbers: Set derived_fact_idx to current_step_simulated.
        if hasattr(next_data, 'step_numbers') and derived_fact_idx < len(next_data.step_numbers):
             if not next_data.step_numbers.is_contiguous():
                  next_data.step_numbers = next_data.step_numbers.contiguous()
             next_data.step_numbers[derived_fact_idx] = current_step_sim

        #    - next_data.x (Base Features): Recompute features affected by the derivation.
        #      This is the most complex part. You might need to call parts of
        #      _compute_node_features or refactor it for reusability.
        #      Features to update:
        #          - [3]: Is derived (for derived_fact_idx)
        #          - [19]: Is head of rule derived (for rules using the derived atom)
        #          - [21]: Rule applicability (for rules whose body now might be satisfied)
        #      Recomputing all features might be easier initially but less efficient.
        print(f"      Derived fact node index: {derived_fact_idx}")
        print("      TODO: Implement recomputation of affected base features in next_data.x")
        # Ensure features are writable
        if not next_data.x.is_contiguous():
             next_data.x = next_data.x.contiguous()

        # Build a set of ALL known facts *in the new state*
        known_atoms_in_new_state = set()
        for i, node in enumerate(nodes):
            # Check initial facts from instance
            if node.get('is_initial', False) and node["type"] == "fact":
                known_atoms_in_new_state.add(node.get("atom"))
            # Check previously derived facts from the *new* mask
            elif next_data.derived_mask[i].item() == 1 and node["type"] == "fact":
                 known_atoms_in_new_state.add(node.get("atom"))

        # 1. Update [3]: Is derived (for derived_fact_idx)
        next_data.x[derived_fact_idx, 3] = 1.0

        # 2. Iterate all nodes to update rule-related features
        for i, node in enumerate(nodes):
            if node['type'] == 'rule':
                # Update [19]: Is head of rule derived
                head_atom = node.get("head_atom", "")
                if head_atom in known_atoms_in_new_state:
                    next_data.x[i, 19] = 1.0
                else:
                    next_data.x[i, 19] = 0.0 # Ensure it's 0 if not
                
                # Update [21]: Rule applicability (fraction of body satisfied)
                body_atoms = node.get("body_atoms", [])
                if body_atoms:
                    body_satisfied = sum(1 for atom in body_atoms if atom in known_atoms_in_new_state)
                    next_data.x[i, 21] = body_satisfied / len(body_atoms)
                else:
                    next_data.x[i, 21] = 0.0 # Rule with no body
        
        print("      ✓ Minimal feature recomputation complete.")
        # --- END FIX ---

        #    - next_data.value_target: Recalculate based on new step.
        proof_length = instance.get("metadata", {}).get('proof_length', current_step_sim + 1)
        if proof_length == 0: proof_length = 1
        next_data.value_target = torch.tensor([1.0 - (current_step_sim / float(proof_length))], dtype=torch.float)

        #    - next_data.tactic_target: This is tricky in simulation. Maybe set to 'unknown'?
        #      Or, try to infer from the applied rule? For search, it might not be needed.
        if hasattr(next_data, 'tactic_target'):
             next_data.tactic_target = torch.tensor([0], dtype=torch.long) # Set to 'unknown'


        # --- END CRITICAL IMPLEMENTATION ---

        return next_data

    def is_goal_satisfied(self, data: Data) -> bool:
        """Checks if the goal state is reached."""
        # This depends on how the goal is represented in your instance/data.
        # Example: Check if a specific 'goal' node has been derived.
        instance = data.instance_data
        goal_formula = instance.get('goal', '') # Assuming goal is a formula string

        if not goal_formula: return False

        nodes = instance.get("nodes", [])
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}

        goal_node_idx = -1
        for i, node in enumerate(nodes):
            if node.get('type') == 'fact' and node.get('atom') == goal_formula:
                 goal_node_idx = i
                 break

        if goal_node_idx != -1 and hasattr(data, 'derived_mask') and goal_node_idx < len(data.derived_mask):
             return data.derived_mask[goal_node_idx].item() == 1

        return False

    def get_applicable_rules_indices(self, data: Data) -> List[int]:
         """ Identifies indices of rule nodes whose premises are met. """
         instance = data.instance_data
         nodes = instance.get("nodes", [])
         edges = instance.get("edges", [])
         id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
         idx2nid = {i: nid for nid, i in id2idx.items()}

         known_node_indices = set(data.derived_mask.nonzero().squeeze(-1).tolist())
         # Add indices of initial axioms
         for i, node in enumerate(nodes):
              # Check is_initial field from the *original* instance data
              if node.get('type') == 'fact' and node.get('is_initial', False):
                  known_node_indices.add(i)

         applicable_rule_indices = []
         for i, node in enumerate(nodes):
             if node.get('type') == 'rule':
                 rule_nid = node['nid']
                 premises_met = True
                 has_premises = False

                 # Find premise fact nodes connected to this rule node via 'body' edges
                 premise_fact_nids = set()
                 for edge in edges:
                      if edge['dst'] == rule_nid and edge['etype'] == 'body':
                          premise_fact_nids.add(edge['src'])
                          has_premises = True

                 # If no premises required, it's potentially applicable if head isn't known
                 if not has_premises:
                     # Check if head is already known
                     head_already_known = False
                     for edge in edges:
                          if edge['src'] == rule_nid and edge['etype'] == 'head':
                              head_nid = edge['dst']
                              if head_nid in id2idx and id2idx[head_nid] in known_node_indices:
                                  head_already_known = True
                                  break
                     if not head_already_known:
                          applicable_rule_indices.append(i)
                     continue # Skip premise check


                 # Check if all required premises are in the known set
                 for premise_nid in premise_fact_nids:
                     if premise_nid not in id2idx or id2idx[premise_nid] not in known_node_indices:
                         premises_met = False
                         break

                 if premises_met:
                     # Check if the rule's conclusion (head) is already known
                     head_already_known = False
                     for edge in edges:
                          if edge['src'] == rule_nid and edge['etype'] == 'head':
                              head_nid = edge['dst']
                              if head_nid in id2idx and id2idx[head_nid] in known_node_indices:
                                  head_already_known = True
                                  break
                     # Rule is applicable only if its head is not already derived
                     if not head_already_known:
                          applicable_rule_indices.append(i)

         return applicable_rule_indices

class BeamSearchRanker:
    """
    Beam search over proof steps using the Phase 2 GNN.
    Faster than MCTS, explores multiple paths, and is batchable.
    """
    def __init__(self, model: TacticGuidedGNN, dataset_instance: StepPredictionDataset, beam_width: int = 5, device: str = 'cpu'):
        self.model = model
        self.beam_width = beam_width
        self.device = device
        self.model.eval()
        # --- NEW: Simulator needed ---
        self.simulator = ProofSimulator(dataset_instance)

    @torch.no_grad()
    def search(self, initial_instance: dict, max_depth: int = 10):
        """
        Performs beam search to find a proof.
        Requires ProofSimulator to be implemented.
        """
        initial_state_data = self.simulator.get_initial_state(initial_instance)
        if initial_state_data is None:
            print("Error: Could not create initial state for beam search.")
            return None, 0.0

        beams: List[Tuple[Data, float, List[int]]] = [(initial_state_data, 0.0, [])]  # (state_data, score, path_indices)
        completed_proofs: List[Tuple[List[int], float]] = []

        for depth in range(max_depth):
            candidates: List[Tuple[Data, float, List[int]]] = []
            
            # --- Potential Batching Point ---
            # Collect all states from current beams for batch prediction
            batch_list = [state_data for state_data, _, _ in beams]
            if not batch_list: break # No active beams

            # Create a PyG Batch object
            current_batch = Batch.from_data_list(batch_list).to(self.device)

            # Get scores for all nodes in the batch
            scores_batch, _, _, _ = self.model(
                current_batch.x, current_batch.edge_index, current_batch.derived_mask,
                current_batch.step_numbers, current_batch.eigvecs, current_batch.eigvals,
                current_batch.eig_mask, current_batch.edge_attr, current_batch.batch
            )
            # --- End Batching Point ---

            beam_idx = 0
            for state_data, score, path in beams:
                # Find applicable rules for *this specific state* within the batch
                applicable_rules = self.simulator.get_applicable_rules_indices(state_data)

                if not applicable_rules:
                    continue

                # Get scores corresponding to this state from the batch results
                is_current_graph = (current_batch.batch == beam_idx)
                graph_scores = scores_batch[is_current_graph] # Scores for nodes in this graph

                # Filter scores to only applicable rule *nodes*
                # Ensure applicable_rules contains valid indices within graph_scores
                valid_applicable_indices = [r for r in applicable_rules if r < len(graph_scores)]
                if not valid_applicable_indices:
                     beam_idx += 1
                     continue

                applicable_scores = graph_scores[valid_applicable_indices]
                
                # We need the original indices, not indices into applicable_scores
                applicable_rule_node_indices = torch.tensor(valid_applicable_indices, device=self.device)


                # Get top-k *applicable* rules based on scores
                num_expand = min(self.beam_width, len(applicable_scores))
                top_k_scores, top_k_relative_indices = torch.topk(applicable_scores, k=num_expand)
                
                # Map relative indices back to original node indices
                top_k_absolute_indices = applicable_rule_node_indices[top_k_relative_indices]


                for rule_node_idx_tensor, rule_score_tensor in zip(top_k_absolute_indices, top_k_scores):
                    rule_node_idx = rule_node_idx_tensor.item()
                    rule_score = rule_score_tensor.item()
                    
                    next_state_data = self.simulator.apply_rule(state_data, rule_node_idx)

                    if next_state_data:
                        new_score = score + rule_score # Add log-probability or score
                        new_path = path + [rule_node_idx]

                        if self.simulator.is_goal_satisfied(next_state_data):
                            completed_proofs.append((new_path, new_score))
                        else:
                            candidates.append((next_state_data, new_score, new_path))
                
                beam_idx += 1 # Move to the next graph in the batch

            # Prune to top beam_width candidates (excluding completed proofs)
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]

            if not beams and completed_proofs: # Stop if all beams finished
                 break
            if not beams: # Stop if search space exhausted
                 break


        # Return best completed proof or best partial proof
        if completed_proofs:
            completed_proofs.sort(key=lambda x: x[1], reverse=True)
            return completed_proofs[0] # (path, score)
        elif beams: # Return best partial beam if no proof found
            return beams[0][2], beams[0][1] # (path, score)
        else:
            return None, 0.0 # Failed to find any path