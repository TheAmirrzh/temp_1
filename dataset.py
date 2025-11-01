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
                 seed: int = 42, k_dim: int = 16):
        self.files = json_files
        self.spectral_dir = spectral_dir
        self.samples = []
        self.tactic_mapper = RuleToTacticMapper()
        self.num_tactics = len(self.tactic_mapper.tactic_names)
        self.k_default = k_dim
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
    
# IN: dataset.py
# Inside StepPredictionDataset.__getitem__

    def __getitem__(self, idx: int) -> 'Data':
        """
        Get a single sample for training.
        
        Fixed to:
        1. Properly compute all variables in order
        2. Compute derived_mask from proof_steps
        3. Compute applicable_mask for training
        """
        inst, step_idx, has_spectral, metadata = self.samples[idx]
        
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        
        # ========================================================================
        # STEP 1: Compute base node features (22-dim)
        # ========================================================================
        x_base = self._compute_node_features(inst, step_idx)
        
        # ========================================================================
        # STEP 2: Build edge_index and edge_attr
        # ========================================================================
        edge_list = []
        edge_types = []
        for e in edges:
            src = id2idx.get(e["src"], -1)
            dst = id2idx.get(e["dst"], -1)
            if src != -1 and dst != -1:
                edge_list.append([src, dst])
                etype = 0 if e["etype"] == "body" else (1 if e["etype"] == "head" else 2)
                edge_types.append(etype)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        
        # ========================================================================
        # STEP 3: Compute target (which rule was used at this step)
        # ========================================================================
        step = proof[step_idx]
        rule_nid = step["used_rule"]
        target_idx = id2idx.get(rule_nid, -1)
        if target_idx == -1:
            target_idx = -1
        y = torch.tensor([target_idx], dtype=torch.long)
        
        # ========================================================================
        # STEP 4: Create PyG Data object
        # ========================================================================
        pyg_data = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # ========================================================================
        # STEP 5: Add value target (for value prediction head)
        # ========================================================================
        proof_length = metadata.get('proof_length', len(proof))
        if proof_length == 0:
            proof_length = 1
        value_target = 1.0 - (step_idx / float(proof_length))
        pyg_data.value_target = torch.tensor([value_target], dtype=torch.float)
        
        # ========================================================================
        # STEP 6: Add difficulty estimate
        # ========================================================================
        difficulty = self._estimate_difficulty(metadata)
        pyg_data.difficulty = torch.tensor([difficulty], dtype=torch.float)
        
        # ========================================================================
        # STEP 7: Add tactic target
        # ========================================================================
        tactic_idx = 0
        if target_idx != -1 and target_idx < len(nodes):
            rule_node = nodes[target_idx]
            if rule_node.get("type") == "rule":
                rule_name = rule_node.get("label", "")
                tactic_idx = self.tactic_mapper.map_rule_to_tactic(rule_name)
        pyg_data.tactic_target = torch.tensor([tactic_idx], dtype=torch.long)
        
        # ========================================================================
        # STEP 8: Compute derived_mask (which facts have been derived so far)
        # ========================================================================
        # THIS IS THE KEY STEP THAT WAS MISSING!
        num_nodes = len(nodes)
        derived_mask = torch.zeros(num_nodes, dtype=torch.uint8)
        
        # Mark which facts are derived UP TO THIS STEP
        for i in range(step_idx):  # Only steps BEFORE this one
            derived_step = proof[i]
            derived_nid = derived_step.get("derived_node")
            if derived_nid in id2idx:
                derived_idx = id2idx[derived_nid]
                derived_mask[derived_idx] = 1
        
        pyg_data.derived_mask = derived_mask
        
        # ========================================================================
        # STEP 9: Compute step_numbers (when each fact was derived)
        # ========================================================================
        step_numbers = torch.zeros(num_nodes, dtype=torch.long)
        
        # Set step number for each derived fact
        for i in range(step_idx):
            derived_step = proof[i]
            derived_nid = derived_step.get("derived_node")
            if derived_nid in id2idx:
                derived_idx = id2idx[derived_nid]
                step_numbers[derived_idx] = i
        
        pyg_data.step_numbers = step_numbers
        
        # ========================================================================
        # STEP 10: Load spectral features (if available)
        # ========================================================================
        k_target = self.k_default
        instance_id = inst.get("id")
        
        # Check if we have spectral cache
        has_spectral_file = False
        if self.spectral_dir and instance_id:
            spectral_path = Path(self.spectral_dir) / f"{instance_id}_spectral.npz"
            has_spectral_file = spectral_path.exists()
        
        num_nodes_check = x_base.shape[0]
        
        if has_spectral_file and instance_id:
            spectral_path = Path(self.spectral_dir) / f"{instance_id}_spectral.npz"
            try:
                data = np.load(spectral_path)
                eigvals_np = data["eigenvalues"].astype(np.float32)
                eigvecs_np = data["eigenvectors"].astype(np.float32)
                eig_mask = torch.ones((k_target,), dtype=torch.bool)

                
                # Validate dimensions
                if eigvecs_np.shape[0] != num_nodes_check:
                    raise ValueError(f"Node count mismatch: spectral={eigvecs_np.shape[0]}, graph={num_nodes_check}")
                
                # Pad or truncate to k_target
                current_k = len(eigvals_np)
                eig_mask_np = np.ones(k_target, dtype=bool) # Assume all are real
                
                if current_k < k_target:
                    pad_k = k_target - current_k
                    eigvals_np = np.pad(eigvals_np, (0, pad_k), mode='constant')
                    eigvecs_np = np.pad(eigvecs_np, ((0, 0), (0, pad_k)), mode='constant')
                    
                    # Mark the padded values as False (invalid)
                    eig_mask_np[current_k:] = False 
                    
                elif current_k > k_target:
                    eigvals_np = eigvals_np[:k_target]
                    eigvecs_np = eigvecs_np[:, :k_target]
                    # Mask is already all True, which is correct
                
                eigvals = torch.from_numpy(eigvals_np)
                eigvecs = torch.from_numpy(eigvecs_np)
                eig_mask = torch.from_numpy(eig_mask_np)

                
                
            except Exception as e:
                print(f"DEBUG [{instance_id}]: {str(e)}. Using zeros.")
                eigvals = torch.zeros((k_target,), dtype=torch.float)
                eigvecs = torch.zeros((num_nodes_check, k_target), dtype=torch.float)
                eig_mask = torch.ones((k_target,), dtype=torch.bool)
        else:
            # No spectral data available
            eigvals = torch.zeros((k_target,), dtype=torch.float)
            eigvecs = torch.zeros((num_nodes_check, k_target), dtype=torch.float)
            eig_mask = torch.ones((k_target,), dtype=torch.bool)  # All valid

        
        pyg_data.eigvecs = eigvecs.contiguous()
        pyg_data.eigvals = eigvals.contiguous()
        pyg_data.eig_mask = eig_mask.contiguous()
        
        # ========================================================================
        # STEP 11: Add metadata
        # ========================================================================
        pyg_data.meta = {
            "instance_id": inst.get("id", ""),
            "step_idx": step_idx,
            "num_nodes": num_nodes_check,
            "num_tactics": self.num_tactics
        }
        
        # ========================================================================
        # STEP 12: Compute temporal features
        # ========================================================================
        proof_state_sim = {
            'num_nodes': num_nodes_check,
            'derivations': [
                (id2idx.get(step_data['derived_node']), i)
                for i, step_data in enumerate(proof[:step_idx])
                if id2idx.get(step_data.get('derived_node')) is not None
            ]
        }
        
        from temporal_encoder import compute_derived_mask, compute_step_numbers
        
        derived_mask_temporal = compute_derived_mask(proof_state_sim, step_idx)
        step_numbers_temporal = compute_step_numbers(proof_state_sim, step_idx)
        
        # Ensure sizes match
        if len(derived_mask_temporal) == num_nodes_check:
            pyg_data.derived_mask = derived_mask_temporal
        else:
            print(f"Warning: derived_mask length mismatch ({len(derived_mask_temporal)} vs {num_nodes_check})")
            pyg_data.derived_mask = torch.zeros(num_nodes_check, dtype=torch.uint8)
        
        if len(step_numbers_temporal) == num_nodes_check:
            pyg_data.step_numbers = step_numbers_temporal
        else:
            print(f"Warning: step_numbers length mismatch ({len(step_numbers_temporal)} vs {num_nodes_check})")
            pyg_data.step_numbers = torch.zeros(num_nodes_check, dtype=torch.long)
        
        # ========================================================================
        # STEP 13: Compute applicable_mask (NEW FIX - THIS IS THE KEY ADDITION)
        # ========================================================================
        applicable_mask, known_facts = compute_applicable_rules_for_step(
            nodes,
            edges,
            step_idx,
            proof
        )
        if not applicable_mask[target_idx]:
            raise RuntimeError(
                f"Data corruption: Target rule {target_idx} not applicable at "
                f"step {step_idx} in instance {inst.get('id')}"
            )
        # Attach to data object
        pyg_data.applicable_mask = applicable_mask.to(torch.bool)  
        
        # Verify target is applicable (sanity check)
        target_idx_val = pyg_data.y.item()
        # if 0 <= target_idx_val < len(applicable_mask):
        #     if not applicable_mask[target_idx_val]:
        #         print(f"WARNING: Target rule {target_idx_val} not applicable at step {step_idx}")
        #         print(f"Instance ID: {inst.get('id', 'unknown')}")  # For traceability
        #         target_node = nodes[target_idx_val]
        #         print(f"Target Rule Details: {target_node}")  # Full node dict
        #         body_atoms = set(target_node.get("body_atoms", []))
        #         head_atom = target_node.get("head_atom")
        #         print(f"Body Atoms: {body_atoms}")
        #         print(f"Head Atom: {head_atom}")
        #         missing_body = body_atoms - known_facts  # Assuming known_facts is accessible (move from function)
        #         print(f"Missing Body Atoms: {missing_body}")
        #         if head_atom in known_facts:
        #             print(f"Head Already Known: Yes")
        #         # Optionally: print full known_facts if small, else len(known_facts)
        #         print(f"Known Facts Count: {len(known_facts)}")
        
        # ========================================================================
        # RETURN THE COMPLETE DATA OBJECT
        # ========================================================================
        return pyg_data

    
    def _estimate_difficulty(self, metadata: dict) -> float:
        """
        FIXED: Properly scaled difficulty estimation
        Maps instances to [0, 1] range with better discrimination
        """
        # Get raw metrics
        n_rules = metadata.get('n_rules', 10)
        n_nodes = metadata.get('n_nodes', 20)
        proof_length = metadata.get('proof_length', 5)
        
        # Define expected ranges per difficulty (from your generation params)
        # Easy: 4 rules, 12 nodes, 3 steps
        # Medium: 12 rules, 20 nodes, 5 steps  
        # Hard: 20 rules, 35 nodes, 8 steps
        # Very Hard: 35 rules, 60 nodes, 12 steps
        
        # Normalize using non-linear scaling (captures spread better)
        rule_score = min(n_rules / 40.0, 1.0)  # Cap at 40 rules
        node_score = min(n_nodes / 70.0, 1.0)  # Cap at 70 nodes
        proof_score = min(proof_length / 15.0, 1.0)  # Cap at 15 steps
        
        # Weighted combination (heavier on proof length - most indicative)
        difficulty = (0.3 * rule_score + 
                    0.2 * node_score + 
                    0.5 * proof_score)
        
        return min(max(difficulty, 0.0), 1.0)


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
        
