
import json
from pathlib import Path
import torch
from data_generator import ProofVerifier
def check_and_fix_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if "proof_steps" not in data or not isinstance(data["proof_steps"], list):
            print(f"Fixing {file_path}: Missing or invalid 'proof_steps'.")
            data["proof_steps"] = []
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return

        # Check for invalid 'derived_node' entries
        for step in data["proof_steps"]:
            if "derived_node" not in step or not isinstance(step["derived_node"], str):
                print(f"Fixing {file_path}: Invalid 'derived_node' in a step.")
                step["derived_node"] = ""  # Or some other default
        
        # Add more checks as needed...

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {e}")

def scan_and_fix_json_files(directory):
    path = Path(directory)
    for file_path in path.rglob("*.json"):
        check_and_fix_json_file(file_path)
def validate_and_fix_dataset(directory):
    path = Path(directory)
    invalid_dir = path / "invalid"
    invalid_dir.mkdir(exist_ok=True)
    invalid_count = 0
    
    for file_path in path.rglob("*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        proof_steps = data.get("proof_steps", [])
        
        verifier = ProofVerifier(nodes, edges)
        is_valid, error = verifier.verify_proof(proof_steps)
        
        if not is_valid:
            print(f"Invalid proof in {file_path}: {error}")
            # Move to invalid
            file_path.rename(invalid_dir / file_path.name)
            invalid_count += 1
        
    print(f"Validated dataset. Moved {invalid_count} invalid files.")
def compute_applicable_rules_for_step_fixed(nodes: list, edges: list, step_idx: int, proof_steps: list) -> torch.Tensor:
    num_nodes = len(nodes)
    applicable_mask = torch.zeros(num_nodes, dtype=torch.bool)
    nid_to_idx = {n["nid"]: i for i, n in enumerate(nodes)}
    
    known_facts = set()
    
    # Correctly identify all initial facts/axioms
    for node in nodes:
        if node.get("is_initial", False) and node["type"] == "fact":
            atom = node.get("atom")
            if atom:
                known_facts.add(atom)

    # Add facts derived in previous steps
    for i in range(step_idx):
        step = proof_steps[i]
        derived_nid = step.get("derived_node")
        if derived_nid in nid_to_idx:
            derived_node = nodes[nid_to_idx[derived_nid]]
            atom = derived_node.get("atom")
            if atom:
                known_facts.add(atom)

    # Check applicability
    for i, node in enumerate(nodes):
        if node["type"] == "rule":
            body_atoms = set(node.get("body_atoms", []))
            head_atom = node.get("head_atom")
            
            if body_atoms.issubset(known_facts) and head_atom not in known_facts:
                applicable_mask[i] = True
                
    return applicable_mask

if __name__ == '__main__':
    validate_and_fix_dataset('./generated_data')
    
