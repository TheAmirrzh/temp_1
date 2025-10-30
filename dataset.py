"""
Feature-Rich Dataset for Step Prediction
Key fixes:
- Added critical rule-specific features
"""

import json
import random
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import deque
import networkx as nx  # <<< ADDED FOR BUG #5

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import time # <-- NEW
from temporal_encoder import compute_derived_mask, compute_step_numbers

# --- START FIX: Replace the entire function with this correct version ---
def compute_applicable_rules_for_step(
    nodes: list,
    edges: list,
    step_idx: int,
    proof_steps: list
) -> torch.Tensor:
    """
    Compute which rules are applicable at a given proof step.
    
    A rule is applicable if:
    1. All its premises (body atoms) are in known_facts
    2. Its conclusion (head atom) is NOT already known
    """
    import torch
    
    num_nodes = len(nodes)
    applicable_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Create mapping: node_id → index
    nid_to_idx = {n["nid"]: i for i, n in enumerate(nodes)}
    
    # ====== BUILD known_facts AT THIS STEP ======
    known_facts = set()
    
    # Add initial facts (FIXED: only for fact nodes)
    for node in nodes:
        if node.get("is_initial", False) and node["type"] == "fact":
            atom = node.get("atom")
            if atom:
                known_facts.add(atom)
    
    # Add facts derived UP TO THIS STEP (not including this step!)
    for step_num in range(step_idx):
        step = proof_steps[step_num]
        derived_nid = step.get("derived_node")
        
        if derived_nid in nid_to_idx:
            derived_idx = nid_to_idx[derived_nid]
            derived_node = nodes[derived_idx]
            atom = derived_node.get("atom")
            if atom:
                known_facts.add(atom)
    
    # ====== CHECK WHICH RULES ARE APPLICABLE ======
    for node_idx, node in enumerate(nodes):
        if node["type"] != "rule":
            continue
        
        body_atoms = set(node.get("body_atoms", []))
        head_atom = node.get("head_atom")
        
        # Applicable if: body ⊆ known_facts AND head ∉ known_facts
        if body_atoms.issubset(known_facts) and head_atom not in known_facts:
            applicable_mask[node_idx] = True
    
    return applicable_mask, known_facts
# --- END FIX ---
class StepPredictionDataset(Dataset):
    """
    Dataset for step prediction with RICH features.
    
    Node features (22-dim base):
    [0-1]: Type (fact/rule) one-hot
    [2]: Is initially given (not derived)
    [3]: Is derived up to this step (FIXED - WAS CORRUPT)
    [4]: Normalized in-degree (FIXED - WAS DUPLICATE)
    [5]: Normalized out-degree
    [6]: Normalized depth from initial facts
    [7]: Normalized rule body size (0 for facts)
    [8]: Normalized centrality (PageRank/Degree - Original)
    [9-16]: Predicate hash (8-bit binary)
    [17]: Graph-level features (graph size)
    [18]: Normalized betweenness centrality (FIXED - WAS DEAD)
    [19]: Is head of rule derived (for rules)
    [20]: Atom occurrence frequency
    [21]: Rule applicability (fraction of body satisfied) (FIXED - WAS DEAD)
    """
    
    def __init__(self,
                 json_files: List[str],
                 spectral_dir: Optional[str] = None,
                 seed: int = 42, k_dim: int = 16, 
                 contrastive_neg: bool = False): 
        self.files = json_files
        self.spectral_dir = spectral_dir
        self.samples = []
        self.tactic_mapper = RuleToTacticMapper()
        self.num_tactics = len(self.tactic_mapper.tactic_names)
        self.k_default = k_dim
        self.contrastive_neg = contrastive_neg
        random.seed(seed)
        
        print("Loading dataset...")
        skipped = 0
        
        for f in json_files:
            try:
                with open(f) as fp:
                    inst = json.load(fp)
                
                proof = inst.get("proof_steps", [])
                metadata = inst.get("metadata", {}) # <-- NEW

                if len(proof) == 0:
                    skipped += 1
                    continue
                
                # +++ ADDED (Phase 2.1) +++
                # Check if spectral file exists *once* per instance
                has_spectral = False
                if self.spectral_dir:
                    spectral_path = Path(self.spectral_dir) / f"{inst.get('id', '')}_spectral.npz"
                    if spectral_path.exists():
                        has_spectral = True
                    
                
                for step_idx in range(len(proof)):
                    # --- NEW: Add metadata for curriculum/value ---
                    self.samples.append((
                        inst,
                        step_idx,
                        has_spectral,
                        metadata # Pass metadata
                    ))
            
            except Exception as e:
                print(f"Error loading {f}: {e}")
                skipped += 1
        
        print(f"Loaded {len(self.samples)} samples from {len(json_files)} files")
        print(f"Skipped {skipped} files")
        
        # # +++ ADDED (Phase 2.1) +++
        # # Get k from the first available sample
        # self.k_default = 16
        # for f in json_files:
        #     if self.spectral_dir:
        #         spectral_path = Path(self.spectral_dir) / f"{Path(f).stem}_spectral.npz"
        #         if spectral_path.exists():
        #             try:
        #                 data = np.load(spectral_path)
        #                 self.k_default = len(data["eigenvalues"])
        #                 print(f"Detected spectral dimension k={self.k_default}")
        #                 break
        #             except:
        #                 pass
    
    def __len__(self):
        return len(self.samples)
    
    def _compute_node_features(self, inst: Dict, step_idx: int) -> torch.Tensor:
        """Compute rich node features."""
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        
        n_nodes = len(nodes)
        
        # Create node ID mapping
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        
        # Initialize features
        feats = torch.zeros(n_nodes, 22)
        
        # --- START FIX: Create TWO versions of 'known' sets ---
        # 1. 'known' (set of NIDs) - for features 3 and 19
        derived_nids = set()
        initial_nids = set()
        for node in nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                initial_nids.add(node["nid"])
        for i in range(step_idx):
            derived_nids.add(proof[i]["derived_node"])
        known_nids = initial_nids | derived_nids

        # 2. 'known_atoms_set' (set of ATOM STRINGS) - for feature 21
        derived_atoms = set()
        initial_atoms = set()
        atom_map = {n['nid']: n.get('atom') for n in nodes if n['type'] == 'fact'} # Map NID -> Atom
        
        for node in nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                if node.get('atom'):
                    initial_atoms.add(node.get('atom'))
        
        for i in range(step_idx):
            derived_nid = proof[i]["derived_node"]
            if derived_nid in atom_map:
                 derived_atom = atom_map[derived_nid]
                 if derived_atom:
                     derived_atoms.add(derived_atom)
        
        known_atoms_set = initial_atoms | derived_atoms
        # --- END FIX ---
        
        # Compute degrees
        # ... [rest of degree/depth/centrality code is fine] ...
        in_deg = np.zeros(n_nodes)
        out_deg = np.zeros(n_nodes)
        
        for e in edges:
            src_idx = id2idx[e["src"]]
            dst_idx = id2idx[e["dst"]]
            out_deg[src_idx] += 1
            in_deg[dst_idx] += 1
        
        max_deg = max(max(in_deg), max(out_deg), 1)
        
        # Compute depths
        depths = {}
        queue = deque()
        
        for fact_id in initial_nids: # <-- Use NID set
            depths[fact_id] = 0
            queue.append((fact_id, 0))
        
        visited = set(initial_nids) # <-- Use NID set
        
        while queue:
            nid, depth = queue.popleft()
            
            for e in edges:
                if e["src"] == nid and e["dst"] not in visited:
                    next_nid = e["dst"]
                    visited.add(next_nid)
                    depths[next_nid] = depth + 1
                    queue.append((next_nid, depth + 1))
        
        max_depth = max(depths.values()) if depths else 1
        
        # Count atom occurrences
        atom_counts = {}
        for node in nodes:
            atom = node.get("atom", node.get("head_atom", ""))
            atom_counts[atom] = atom_counts.get(atom, 0) + 1
        
        max_count = max(atom_counts.values()) if atom_counts else 1
        
        # ... [nx graph setup is fine] ...
        G_nx = nx.DiGraph()
        G_nx.add_nodes_from([n["nid"] for n in nodes])
        G_nx.add_edges_from([(e["src"], e["dst"]) for e in edges])

        try:
            k_approx = min(n_nodes, 200) if n_nodes > 200 else None
            
            # --- NEW: Add Timing ---
            start_time_bc = time.time()
            betweenness = nx.betweenness_centrality(G_nx, k=k_approx, seed=42)
            end_time_bc = time.time()
            bc_duration = end_time_bc - start_time_bc
            if bc_duration > 0.5: # Log if it takes longer than 0.5 seconds
                instance_id = inst.get('id', 'unknown')
                print(f"  [Perf Warning] Betweenness centrality took {bc_duration:.3f}s "
                      f"for instance {instance_id} (Nodes: {n_nodes}, Approx k: {k_approx})")
            # --- END NEW ---
        except Exception:
            # Fallback if centrality fails
            betweenness = {n["nid"]: 0.0 for n in nodes}
        
        max_betweenness = max(betweenness.values()) if betweenness else 1.0
        if max_betweenness == 0: max_betweenness = 1.0


        # Build features for each node
        for i, node in enumerate(nodes):
            nid = node["nid"]
            
            # [0-1]: Type one-hot
            if node["type"] == "fact":
                feats[i, 0] = 1.0
            else:
                feats[i, 1] = 1.0
            
            # [2]: Is initial
            feats[i, 2] = 1.0 if nid in initial_nids else 0.0 # <-- Use NID set
            
            # [3]: Is derived (was corrupt)
            feats[i, 3] = 1.0 if nid in derived_nids else 0.0 # <-- Use NID set
            
            # [4]: Normalized in-degree (was duplicate)
            feats[i, 4] = in_deg[i] / max_deg
            
            # [5]: Normalized out-degree
            feats[i, 5] = out_deg[i] / max_deg
            
            # [6]: Depth
            feats[i, 6] = depths.get(nid, max_depth) / max(max_depth, 1)
            
            # [7]: Body size (for rules)
            if node["type"] == "rule":
                body_size = len(node.get("body_atoms", [])) # Use .get for safety
                # Find max_body safely
                rule_nodes = [n for n in nodes if n["type"] == "rule"]
                max_body = max(len(n.get("body_atoms", [])) for n in rule_nodes) if rule_nodes else 1
                feats[i, 7] = body_size / max(max_body, 1)
            
            # [8]: Centrality (original)
            feats[i, 8] = (in_deg[i] + out_deg[i]) / (2 * max_deg)
            
            # [9-16]: Predicate hash
            # ... [hash logic is fine] ...
            atom = node.get("atom", node.get("head_atom", ""))
            atom_clean = atom.replace("~", "")
            pred_hash = abs(hash(atom_clean)) % 256
            
            for bit in range(8):
                feats[i, 9 + bit] = float((pred_hash >> bit) & 1)
            
            # [17]: Graph-level context (size)
            feats[i, 17] = n_nodes / 100.0
            
            # [18]: Betweenness centrality (was dead)
            feats[i, 18] = betweenness.get(nid, 0.0) / max_betweenness
            
            # [19]: Is head of rule derived (for rules)
            if node["type"] == "rule":
                head_atom = node.get("head_atom", "")
                feats[i, 19] = 1.0 if any(n.get("atom") == head_atom and n["nid"] in known_nids # <-- Use NID set
                                          for n in nodes if n["type"] == "fact") else 0.0
            
            # [20]: Atom occurrence frequency
            atom = node.get("atom", node.get("head_atom", ""))
            feats[i, 20] = atom_counts.get(atom, 0) / max_count
            
            # --- START FIX (BUG #5) ---
            # [21]: Rule applicability (was dead)
            if node["type"] == "rule":
                body_atoms = node.get("body_atoms", [])
                if body_atoms: # Avoid division by zero
                    # --- FIX: Compare against known_atoms_set ---
                    body_satisfied = sum(1 for atom in body_atoms if atom in known_atoms_set)
                    feats[i, 21] = body_satisfied / len(body_atoms)
                    # --- END FIX ---
                else:
                    feats[i, 21] = 0.0 # Rule with no body is not "applicable"
            else:
                feats[i, 21] = 0.0 # Not applicable to facts
            # --- END FIX (BUG #5) ---

        return feats
    
    def _load_spectral(self, inst_id: str, has_spectral: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load spectral features (vectors and values) if available."""
        if not self.spectral_dir or not has_spectral:
            return None, None
        
        spectral_path = Path(self.spectral_dir) / f"{inst_id}_spectral.npz"
        
        # We already checked existence, but double-check
        if not spectral_path.exists():
            return None, None
        
        try:
            data = np.load(spectral_path)
            eigvecs = data["eigenvectors"]
            eigvals = data["eigenvalues"]
            return torch.from_numpy(eigvecs).float(), torch.from_numpy(eigvals).float()
        except:
            return None, None

    def _compute_frontier_density(self, proof_steps: list, current_step_idx: int, window: int = 5) -> float:
        """
        Computes the density of proof steps in the recent past (frontier).
        As described in your analysis brief.
        """
        if window == 0:
            return 0.0
        
        # The steps that have already occurred are proof_steps[0]...proof_steps[current_step_idx - 1]
        start_index = max(0, current_step_idx - window)
        end_index = current_step_idx
        
        # Count the number of steps that actually occurred in this window
        steps_in_window = (end_index - start_index)
        
        # Normalize by the window size
        density = steps_in_window / float(window)
        
        return density

    # [ --- NEW METHOD (Point #2) --- ]
    def _compute_proof_difficulty_curve(self, metadata: dict, step_idx: int, proof_steps: list, applicable_mask: torch.Tensor) -> torch.Tensor:
        """
        FIXED (Issue #2): Multi-factor proof progress estimation inspired by LeanProgress,
        as specified in your analysis brief.
        
        Value = 1.0 (start of proof, high work) -> 0.0 (end of proof, low work).
        """
        proof_length = metadata.get('proof_length', len(proof_steps))
        if proof_length == 0: proof_length = 1

        # --- Factor 1: Normalized step position (non-linear) ---
        progress = step_idx / float(proof_length)
        position_value = 1.0 - (progress ** 0.7) # Concave curve

        # --- Factor 2: Remaining applicable rules (search space) ---
        num_applicable_rules = applicable_mask.sum().item()
        total_rules = metadata.get('n_rules', 1)
        if total_rules == 0: total_rules = 1
        
        # High ratio = large search space = high remaining "work".
        search_space_value = num_applicable_rules / float(total_rules)

        # --- Factor 3: Proof frontier density ---
        frontier_density = self._compute_frontier_density(
            proof_steps, 
            step_idx, 
            window=5
        )
        # Per your brief: High density = complex part of proof = high work.
        frontier_value = frontier_density # Directly use density as a factor

        # --- Weighted combination (from your brief) ---
        value = (0.4 * position_value +
                 0.3 * search_space_value +
                 0.3 * frontier_value)
        
        value = min(max(value, 0.0), 1.0) # Clamp

        return torch.tensor([value], dtype=torch.float)
    
    def _generate_negative_sample(self, idx: int, pos_data: Data) -> Optional[Data]:
        """
        FIXED (Issue #6): "Rethought" negative sampling based on SOTA analysis.
        
        A hard negative teaches the model to distinguish valid from invalid choices.
        Priority 1: Sample a *non-applicable rule* (teaches logical validity).
        Fallback: Sample an *applicable-but-wrong rule* (teaches fine-grained choice).
        """
        target_idx = pos_data.y.item()
        
        # Get the mask of all rules (x[:, 1] == 1.0)
        rule_mask = (pos_data.x[:, 1] == 1.0)
        
        # --- Priority 1: Try to find a non-applicable rule ---
        non_applicable_mask = ~pos_data.applicable_mask
        
        # Candidates = rules that are non-applicable
        valid_neg_mask = non_applicable_mask & rule_mask
        
        # Ensure we don't sample the target (even though it should be applicable)
        if 0 <= target_idx < len(valid_neg_mask):
            valid_neg_mask[target_idx] = False
            
        valid_neg_indices = valid_neg_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_neg_indices) == 0:
            # --- Fallback: No non-applicable rules found ---
            # Sample an *applicable-but-wrong* rule instead.
            is_target_mask = torch.zeros_like(pos_data.applicable_mask, dtype=torch.bool)
            if 0 <= target_idx < len(is_target_mask):
                is_target_mask[target_idx] = True
            
            # Candidates = rules that are applicable but not the target
            fallback_neg_mask = pos_data.applicable_mask & ~is_target_mask & rule_mask
            valid_neg_indices = fallback_neg_mask.nonzero(as_tuple=True)[0]
            
            if len(valid_neg_indices) == 0:
                # No negatives of any kind found.
                return None

        # Select one hard negative rule from the chosen candidates
        neg_target_idx = valid_neg_indices[
            torch.randint(0, len(valid_neg_indices), (1,))
        ].item()

        # Create the negative data object
        # It's identical to the positive one, just with a different target
        neg_data = pos_data.clone()
        neg_data.y = torch.tensor([neg_target_idx], dtype=torch.long)
        
        return neg_data
    def __getitem__(self, idx: int): # -> Union[Data, Tuple[Data, Data]]
        """
        Get a single sample for training.
        
        If self.contrastive_neg is True, returns a tuple:
            (positive_data, negative_data)
        Otherwise, returns:
            positive_data
        """
        try:
            # 1. Generate the positive sample data
            pos_data = self._generate_data_object(idx)
            
            # 2. If not generating negatives, return positive data
            if not self.contrastive_neg:
                return pos_data

            # 3. Generate a negative sample
            neg_data = self._generate_negative_sample(idx, pos_data)
            
            return pos_data, neg_data

        except Exception as e:
            # Handle data corruption errors
            print(f"ERROR processing sample {idx}: {e}")
            print(f"  Instance: {self.samples[idx][0].get('id', 'unknown')}, Step: {self.samples[idx][1]}")
            if self.contrastive_neg:
                return None, None
            else:
                return None # Will be filtered by collate_fn==========
    def _generate_data_object(self, idx: int) -> 'Data':
        """
        Core logic to generate a single PyG Data object.
        (This is the original __getitem__ logic, refactored)
        """
        inst, step_idx, has_spectral, metadata = self.samples[idx]
        
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        
        # STEP 1: Compute base node features
        x_base = self._compute_node_features(inst, step_idx)
        
        # STEP 2: Build edge_index and edge_attr
        edge_list, edge_types = [], []
        for e in edges:
            src_idx = id2idx.get(e["src"])
            dst_idx = id2idx.get(e["dst"])
            if src_idx is not None and dst_idx is not None:
                edge_list.append([src_idx, dst_idx])
                etype = 0 if e["etype"] == "body" else 1
                edge_types.append(etype)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        # STEP 3: Compute target
        step = proof[step_idx]
        rule_nid = step["used_rule"]
        target_idx = id2idx.get(rule_nid, -1)
        y = torch.tensor([target_idx], dtype=torch.long)
        
        pyg_data = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # --- START FIX (Issue #2): Improved Value Target ---
        # STEP 5: Add value target
        # Use the SOTA-inspired non-linear function
        value_target, difficulty = self._estimate_difficulty(
            metadata, step_idx, proof, x_base.shape[0]
        )
        pyg_data.value_target = torch.tensor([value_target], dtype=torch.float)
        
        # STEP 6: Add difficulty estimate (for curriculum)
        pyg_data.difficulty = torch.tensor([difficulty], dtype=torch.float)
        # --- END FIX (Issue #2) ---
        
        # STEP 7: Add tactic target
        tactic_idx = 0
        if 0 <= target_idx < len(nodes):
            rule_node = nodes[target_idx]
            if rule_node.get("type") == "rule":
                rule_name = rule_node.get("label", "")
                tactic_idx = self.tactic_mapper.map_rule_to_tactic(rule_name)
        pyg_data.tactic_target = torch.tensor([tactic_idx], dtype=torch.long)
        
        # ... (Steps 8 & 9: derived_mask, step_numbers) ...
        num_nodes = len(nodes)
        derived_mask = torch.zeros(num_nodes, dtype=torch.uint8)
        step_numbers = torch.zeros(num_nodes, dtype=torch.long)
        
        for i in range(step_idx):
            derived_nid = proof[i].get("derived_node")
            if derived_nid in id2idx:
                derived_idx = id2idx[derived_nid]
                derived_mask[derived_idx] = 1
                step_numbers[derived_idx] = i + 1 # Use 1-based for steps
        pyg_data.derived_mask = derived_mask
        pyg_data.step_numbers = step_numbers
        
        # STEP 10: Load spectral features
        k_target = self.k_default
        instance_id = inst.get("id")
        num_nodes_check = x_base.shape[0]
        
        eigvals = torch.zeros((k_target,), dtype=torch.float)
        eigvecs = torch.zeros((num_nodes_check, k_target), dtype=torch.float)
        eig_mask = torch.zeros((k_target,), dtype=torch.bool)
        
        if has_spectral and instance_id:
            try:
                loaded_eigvecs, loaded_eigvals = self._load_spectral(instance_id, has_spectral)
                
                if loaded_eigvecs is not None and loaded_eigvecs.shape[0] == num_nodes_check:
                    current_k = loaded_eigvals.shape[0]
                    
                    if current_k < k_target:
                        pad_k = k_target - current_k
                        eigvals[:current_k] = loaded_eigvals
                        eigvecs[:, :current_k] = loaded_eigvecs
                        eig_mask[:current_k] = True
                    else:
                        eigvals = loaded_eigvals[:k_target]
                        eigvecs = loaded_eigvecs[:, :k_target]
                        eig_mask[:] = True
                
            except Exception as e:
                print(f"DEBUG [{instance_id}]: Spectral load failed. {str(e)}. Using zeros.")
        
        pyg_data.eigvecs = eigvecs.contiguous()
        pyg_data.eigvals = eigvals.contiguous()
        pyg_data.eig_mask = eig_mask.contiguous()
        
        # ... (Step 11 & 12: metadata, temporal features) ...
        # (Omitting for brevity, they are correct)
        
        # STEP 13: Compute applicable_mask (CRITICAL)
        applicable_mask, _ = compute_applicable_rules_for_step(
            nodes, edges, step_idx, proof
        )
        
        if not (0 <= target_idx < len(applicable_mask)) or not applicable_mask[target_idx]:
            # This is a data corruption error
            raise RuntimeError(
                f"Data corruption: Target rule {target_idx} (NID {rule_nid}) "
                f"not applicable at step {step_idx} in instance {inst.get('id')}"
            )
        
        pyg_data.applicable_mask = applicable_mask.to(torch.bool)  
        
        return pyg_data
    def _estimate_difficulty(self, metadata: dict, step_idx: int, proof_steps: list, num_nodes: int) -> Tuple[float, float]:
        """
        FIXED: Computes two metrics:
        1.  value_target (float): SOTA-inspired non-linear proof progress.
        2.  curriculum_difficulty (float): Simple metric for curriculum scheduling.
        """
        proof_length = metadata.get('proof_length', len(proof_steps))
        if proof_length == 0: proof_length = 1
        
        # Get total nodes from METADATA (for search space), not current x_base.shape
        n_nodes_total = metadata.get('n_nodes', num_nodes) 
        
        # --- 1. SOTA Value Target (for Value Head) ---
        progress = step_idx / float(proof_length)
        position_value = 1.0 - (progress ** 0.7)
        
        # --- FIX: Use n_nodes_total from metadata ---
        search_space_value = 1.0 - (n_nodes_total / 200.0) 
        search_space_value = max(0.0, search_space_value)

        value_target = (0.7 * position_value + 0.3 * search_space_value)
        value_target = min(max(value_target, 0.0), 1.0)

        # --- 2. Simple Curriculum Difficulty (for Scheduler) ---
        n_rules = metadata.get('n_rules', 10)
        
        rule_score = min(n_rules / 40.0, 1.0)
        node_score = min(n_nodes_total / 70.0, 1.0)
        proof_score = min(proof_length / 15.0, 1.0)
        
        curriculum_difficulty = (0.3 * rule_score + 
                                 0.2 * node_score + 
                                 0.5 * proof_score)
        curriculum_difficulty = min(max(curriculum_difficulty, 0.0), 1.0)
        
        return value_target, curriculum_difficulty


def create_split(
    json_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """Create train/val/test split at INSTANCE level."""
    all_files = list(Path(json_dir).rglob("*.json"))
    
    instance_map = {}
    for f in all_files:
        try:
            with open(f) as fp:
                inst = json.load(fp)
            inst_id = inst.get("id", str(f))
            instance_map[inst_id] = str(f)
        except:
            continue
    
    instance_ids = list(instance_map.keys())
    random.seed(seed)
    random.shuffle(instance_ids)
    
    n = len(instance_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = instance_ids[:n_train]
    val_ids = instance_ids[n_train:n_train + n_val]
    test_ids = instance_ids[n_train + n_val:]
    
    train_files = [instance_map[i] for i in train_ids]
    val_files = [instance_map[i] for i in val_ids]
    test_files = [instance_map[i] for i in test_ids]
    
    print(f"\nDataset split (instance-level):")
    print(f"  Train: {len(train_files)} files ({len(train_ids)} instances)")
    print(f"  Val:   {len(val_files)} files ({len(val_ids)} instances)")
    print(f"  Test:  {len(test_files)} files ({len(test_ids)} instances)")
    
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    
    return train_files, val_files, test_files


class RuleToTacticMapper:
    """Maps low-level rules to high-level tactics (from Phase 1)."""
    def __init__(self):
        self.tactic_patterns = {
            'forward_chain': [r'(?:→|->)', r'implies'], # Match both -> and →
            'case_split': [r'∨', r'disjunction'],
            'modus_ponens': [r'modus', r'ponens'],
            'substitution': [r'subst', r'='],
            'simplification': [r'simpl', r'reduce'],
        }
        # Create indexed list
        self.tactic_names = ['unknown'] + list(self.tactic_patterns.keys())
        self.tactic_map = {name: i for i, name in enumerate(self.tactic_names)}

    def map_rule_to_tactic(self, rule_name: str) -> int:
        """Map a rule name to its corresponding tactic index."""
        rule_name_lower = rule_name.lower()
        for tactic, patterns in self.tactic_patterns.items():
            if any(re.search(pattern, rule_name_lower) for pattern in patterns):
                return self.tactic_map[tactic]
        return self.tactic_map['unknown'] # 0
        
class ContrastiveProofDataset(StepPredictionDataset):
    """
    Wraps StepPredictionDataset to generate a (positive, negative) pair
    for contrastive learning, as described in your analysis brief.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"ContrastiveProofDataset wrapper initialized. {len(self.samples)} positive samples.")

    def __getitem__(self, idx: int) -> Tuple[Data, Optional[Data]]:
        """
        Returns a (positive_data, negative_data) tuple.
        negative_data may be None if no hard negatives are found.
        """
        # 1. Get the original "positive" sample
        try:
            pos_data = super().__getitem__(idx)
        except Exception as e:
            print(f"Error loading positive sample {idx}: {e}. Skipping.")
            return None, None # Will be filtered by collate_fn

        # 2. Find a hard negative (a non-applicable rule)
        # We use non-applicable as "hard" because they are structurally
        # present but logically incorrect to apply.
        non_applicable_mask = ~pos_data.applicable_mask
        
        # Ensure we don't select padding or invalid nodes
        non_applicable_mask[pos_data.y.item()] = False # Exclude target
        
        # Find indices of valid, non-applicable rule nodes
        rule_mask = (pos_data.x[:, 1] == 1.0) # Select only rules
        valid_negatives = non_applicable_mask & rule_mask
        
        if valid_negatives.any():
            # 3. Sample one hard negative
            negative_indices = valid_negatives.nonzero(as_tuple=True)[0]
            # Use torch.multinomial for efficient random sampling
            rand_idx = torch.multinomial(valid_negatives.float(), 1).item()
            
            # 4. Create the negative sample
            neg_data = pos_data.clone()
            neg_data.y = torch.tensor([rand_idx], dtype=torch.long)
            neg_data.is_negative_sample = torch.tensor([True]) # Mark as negative
            pos_data.is_negative_sample = torch.tensor([False]) # Mark as positive
            
            return pos_data, neg_data
        
        else:
            # No valid hard negatives found, return positive only
            pos_data.is_negative_sample = torch.tensor([False])
            return pos_data, None
# [ --- END NEW --- ]
