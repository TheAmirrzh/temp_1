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
                 seed: int = 42):
        self.files = json_files
        self.spectral_dir = spectral_dir
        self.samples = []
        self.tactic_mapper = RuleToTacticMapper()
        self.num_tactics = len(self.tactic_mapper.tactic_names)
        
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
        
        # +++ ADDED (Phase 2.1) +++
        # Get k from the first available sample
        self.k_default = 16
        for f in json_files:
            if self.spectral_dir:
                spectral_path = Path(self.spectral_dir) / f"{Path(f).stem}_spectral.npz"
                if spectral_path.exists():
                    try:
                        data = np.load(spectral_path)
                        self.k_default = len(data["eigenvalues"])
                        print(f"Detected spectral dimension k={self.k_default}")
                        break
                    except:
                        pass
    
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
        
        # Track what's derived up to this step
        derived = set()
        initial_facts = set()
        
        for node in nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                initial_facts.add(node["nid"])
        
        for i in range(step_idx):
            derived.add(proof[i]["derived_node"])
        
        known = initial_facts | derived
        
        # Compute degrees
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
        
        for fact_id in initial_facts:
            depths[fact_id] = 0
            queue.append((fact_id, 0))
        
        visited = set(initial_facts)
        
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
        
        # --- START FIX (BUG #5) ---
        # Create nx graph for betweenness centrality
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
        # --- END FIX (BUG #5) ---

        # Build features for each node
        for i, node in enumerate(nodes):
            nid = node["nid"]
            
            # [0-1]: Type one-hot
            if node["type"] == "fact":
                feats[i, 0] = 1.0
            else:
                feats[i, 1] = 1.0
            
            # [2]: Is initial
            feats[i, 2] = 1.0 if nid in initial_facts else 0.0
            
            # --- START FIX (BUG #1) ---
            # [3]: Is derived (was corrupt)
            feats[i, 3] = 1.0 if nid in derived else 0.0
            
            # [4]: Normalized in-degree (was duplicate)
            feats[i, 4] = in_deg[i] / max_deg
            # --- END FIX (BUG #1) ---
            
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
            atom = node.get("atom", node.get("head_atom", ""))
            atom_clean = atom.replace("~", "")
            pred_hash = abs(hash(atom_clean)) % 256
            
            for bit in range(8):
                feats[i, 9 + bit] = float((pred_hash >> bit) & 1)
            
            # [17]: Graph-level context (size)
            feats[i, 17] = n_nodes / 100.0
            
            # --- START FIX (BUG #5) ---
            # [18]: Betweenness centrality (was dead)
            feats[i, 18] = betweenness.get(nid, 0.0) / max_betweenness
            # --- END FIX (BUG #5) ---
            
            # [19]: Is head of rule derived (for rules)
            if node["type"] == "rule":
                head_atom = node.get("head_atom", "")
                feats[i, 19] = 1.0 if any(n.get("atom") == head_atom and n["nid"] in known 
                                          for n in nodes if n["type"] == "fact") else 0.0
            
            # [20]: Atom occurrence frequency
            atom = node.get("atom", node.get("head_atom", ""))
            feats[i, 20] = atom_counts.get(atom, 0) / max_count
            
            # --- START FIX (BUG #5) ---
            # [21]: Rule applicability (was dead)
            if node["type"] == "rule":
                body_atoms = node.get("body_atoms", [])
                if body_atoms: # Avoid division by zero
                    body_satisfied = sum(1 for atom in body_atoms if atom in known)
                    feats[i, 21] = body_satisfied / len(body_atoms)
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

    def __getitem__(self, idx: int) -> Data:
        inst, step_idx, has_spectral, metadata = self.samples[idx]
        
        # FIX: Define instance_id_for_spectral FIRST
        instance_id_for_spectral = inst.get("id")
        if not instance_id_for_spectral:
            instance_id_for_spectral = metadata.get("id", f"unknown_instance_{idx}") # Unpack correctly
        nodes = inst["nodes"]
        edges = inst["edges"]
        proof = inst["proof_steps"]
        id2idx = {n["nid"]: i for i, n in enumerate(nodes)}
        has_spectral = False
        instance_id_for_spectral = inst.get("id")
        if not instance_id_for_spectral:
            instance_id_for_spectral = metadata.get("id")
            if not instance_id_for_spectral:
                instance_id_for_spectral = f"unknown_instance_{idx}"

        if self.spectral_dir:
            spectral_path = Path(self.spectral_dir) / f"{inst.get('id', '')}_spectral.npz"
            has_spectral = spectral_path.exists()
        
        # if not instance_id_for_spectral:
        #      # Fallback: Try deriving from metadata if 'id' wasn't top-level
        #      instance_id_for_spectral = metadata.get("id")
        #      if not instance_id_for_spectral:
        #           # Last resort: Log a warning, spectral loading will likely fail
        #           print(f"Warning: Cannot determine reliable instance ID for spectral loading in __getitem__ for sample index {idx}.")
        #           instance_id_for_spectral = f"unknown_instance_{idx}" # Placeholder
        # --- End get instance ID ---

        # --- Ensure BASE features are computed FIRST ---
        x_base = self._compute_node_features(inst, step_idx)

        # --- Compute edge_index, edge_attr ---
        edge_list = []
        edge_types = []
        for e in edges:
            src = id2idx.get(e["src"], -1) # Use .get for safety
            dst = id2idx.get(e["dst"], -1) # Use .get for safety
            if src != -1 and dst != -1: # Add edge only if both nodes exist
                edge_list.append([src, dst])
                etype = 0 if e["etype"] == "body" else (1 if e["etype"] == "head" else 2)
                edge_types.append(etype)
        # Handle case with no valid edges
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # --- Compute target y ---
        step = proof[step_idx]
        rule_nid = step["used_rule"]
        target_idx = id2idx.get(rule_nid, -1)
        if target_idx == -1: target_idx = -1 # Keep dummy target if needed
        y = torch.tensor([target_idx], dtype=torch.long)

        # --- CREATE the Data object HERE ---
        pyg_data = Data(x=x_base, edge_index=edge_index, edge_attr=edge_attr, y=y)
        # --- Use a distinct name like pyg_data ---

        # --- Add other attributes ---
        proof_length = metadata.get('proof_length', step_idx + 1)
        if proof_length == 0: proof_length = 1
        value_target = 1.0 - (step_idx / float(proof_length))
        pyg_data.value_target = torch.tensor([value_target], dtype=torch.float)

        difficulty = self._estimate_difficulty(metadata)
        pyg_data.difficulty = torch.tensor([difficulty], dtype=torch.float)

        tactic_idx = 0 # Default to unknown
        if target_idx != -1 and target_idx < len(nodes):
            rule_node = nodes[target_idx] # Get the correct node
            # Ensure the target is actually a rule before getting label
            if rule_node.get("type") == "rule":
                 rule_name = rule_node.get("label", "")
                 tactic_idx = self.tactic_mapper.map_rule_to_tactic(rule_name)
        pyg_data.tactic_target = torch.tensor([tactic_idx], dtype=torch.long)

        # --- Add spectral features ---
        # eigvecs, eigvals = self._load_spectral(instance_id_for_spectral, has_spectral)


         # --- MODIFIED: Spectral Loading ---
        k_target = self.k_default

        # Define num_nodes_check early based on base features (reliable graph size)
        num_nodes_check = x_base.shape[0]

        # Check if spectral data available
        if has_spectral:
            spectral_path = Path(self.spectral_dir) / f"{inst.get('id', '')}_spectral.npz"
            try:
                data = np.load(spectral_path)
                eigvals_np = data["eigenvalues"].astype(np.float32)
                eigvecs_np = data["eigenvectors"].astype(np.float32)

                # Validate against eigenvectors' first dimension (num_nodes)
                if eigvecs_np.shape[0] != num_nodes_check:
                    raise ValueError(f"Node count mismatch: spectral={eigvecs_np.shape[0]}, graph={num_nodes_check}")

                # Pad or truncate to k_target
                current_k = len(eigvals_np)
                if current_k < k_target:
                    pad_k = k_target - current_k
                    eigvals_np = np.pad(eigvals_np, (0, pad_k), mode='constant')
                    eigvecs_np = np.pad(eigvecs_np, ((0, 0), (0, pad_k)), mode='constant')
                elif current_k > k_target:
                    eigvals_np = eigvals_np[:k_target]
                    eigvecs_np = eigvecs_np[:, :k_target]

                eigvals = torch.from_numpy(eigvals_np)
                eigvecs = torch.from_numpy(eigvecs_np)

            except Exception as e:
                print(f"DEBUG [{inst.get('id')}]: {str(e)}. Using zeros.")
                eigvals = torch.zeros((k_target,), dtype=torch.float)
                eigvecs = torch.zeros((num_nodes_check, k_target), dtype=torch.float)  # Use num_nodes_check here

            pyg_data.eigvecs = eigvecs.contiguous()
            pyg_data.eigvals = eigvals.contiguous()
            # --- END ADDED ---

        else:
            # Pad with zeros if spectral data is missing or mismatched
            pyg_data.eigvecs = torch.zeros((num_nodes_check, k_target), dtype=torch.float)  # Use num_nodes_check here
            pyg_data.eigvals = torch.zeros((k_target,), dtype=torch.float)

        # --- Add meta and temporal features ---
        pyg_data.meta = {
            "instance_id": inst.get("id", ""),
            "step_idx": step_idx,
            "num_nodes": num_nodes_check, # Use checked num_nodes
            "num_tactics": self.num_tactics
        }

        # Use num_nodes_check for consistency
        proof_state_sim = {
            'num_nodes': num_nodes_check,
            'derivations': [
                (id2idx.get(step_data['derived_node']), i) # Use .get
                for i, step_data in enumerate(proof[:step_idx])
                if id2idx.get(step_data.get('derived_node')) is not None # Check existence
            ]
        }
        derived_mask = compute_derived_mask(proof_state_sim, step_idx)
        step_numbers = compute_step_numbers(proof_state_sim, step_idx)

        # Ensure mask/steps match node count before assigning
        if len(derived_mask) == num_nodes_check:
             pyg_data.derived_mask = derived_mask
        else:
             # Handle potential mismatch, e.g., create default tensors
             print(f"Warning: derived_mask length mismatch ({len(derived_mask)} vs {num_nodes_check})")
             pyg_data.derived_mask = torch.zeros(num_nodes_check, dtype=torch.uint8)

        if len(step_numbers) == num_nodes_check:
             pyg_data.step_numbers = step_numbers
        else:
             print(f"Warning: step_numbers length mismatch ({len(step_numbers)} vs {num_nodes_check})")
             pyg_data.step_numbers = torch.zeros(num_nodes_check, dtype=torch.long)


        return pyg_data

    def _estimate_difficulty(self, metadata: dict) -> float:
        """Estimate proof difficulty (reuse from Phase 1)."""
        num_rules = metadata.get('n_rules', 10) / 100.0
        num_facts = metadata.get('n_nodes', 20) / 200.0
        proof_length = metadata.get('proof_length', 5) / 50.0
        
        # Combine normalized features
        difficulty = (0.4 * num_rules + 
                      0.3 * num_facts + 
                      0.3 * proof_length)
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
        