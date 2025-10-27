"""
Horn Clause Generator with Guaranteed Valid Proofs
"""

import json
import random
from typing import Dict, List, Set
from enum import Enum
from pathlib import Path


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    EXTREME_HARD = "extreme_hard"


def generate_horn_instance(
    instance_id: str,
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: int = None
) -> Dict:
    """
    Generate Horn clause problem with GUARANTEED valid proof chain.
    """
    if seed:
        random.seed(seed)
    
    # Difficulty parameters
    params = {
        Difficulty.EASY: {
            "n_initial_facts": 6,
            "n_rules": 8,
            "max_depth": 4,
            "body_size": (1, 2),
            "atoms_pool": 15
        },
        Difficulty.MEDIUM: {
            "n_initial_facts": 10,
            "n_rules": 15,
            "max_depth": 6,
            "body_size": (2, 3),
            "atoms_pool": 25
        },
        Difficulty.HARD: {
            "n_initial_facts": 15,
            "n_rules": 25,
            "max_depth": 10,
            "body_size": (2, 4),
            "atoms_pool": 40
        },
        Difficulty.VERY_HARD: {
            "n_initial_facts": 30,
            "n_rules": 50,
            "max_depth": 20,
            "body_size": (3, 6),
            "atoms_pool": 80
        },
        Difficulty.EXTREME_HARD: {
            "n_initial_facts": 50,
            "n_rules": 100,
            "max_depth": 30,
            "body_size": (4, 8),
            "atoms_pool": 150
        }
    }
    
    p = params[difficulty]
    atoms = [f"P{i}" for i in range(p["atoms_pool"])]
    
    nodes = []
    edges = []
    proof_steps = []
    nid_counter = 0
    
    # STEP 1: Create initial facts
    initial_atoms = random.sample(atoms, p["n_initial_facts"])
    fact_map = {}
    
    for atom in initial_atoms:
        node = {
            "nid": nid_counter,
            "type": "fact",
            "label": atom,
            "atom": atom,
            "is_initial": True
        }
        nodes.append(node)
        fact_map[atom] = nid_counter
        nid_counter += 1
    
    # STEP 2: Forward-chaining rule generation
    derived_facts = set(initial_atoms)
    
    for _ in range(p["n_rules"]):
        body_size = random.randint(*p["body_size"])
        available = list(derived_facts)
        
        if len(available) < body_size:
            body_size = len(available)
        
        if body_size == 0:
            continue
            
        body_atoms = random.sample(available, body_size)
        
        # Choose NEW atom as head
        if random.random() < 0.3 and len(derived_facts) < p["atoms_pool"] * 0.6:
            unused = [a for a in atoms if a not in derived_facts]
            if unused:
                head_atom = random.choice(unused)
            else:
                head_atom = random.choice(atoms)
        else:
            head_atom = random.choice(atoms)
        
        # Create rule node
        rule_node = {
            "nid": nid_counter,
            "type": "rule",
            "label": f"({' ∧ '.join(body_atoms)}) → {head_atom}",
            "body_atoms": body_atoms,
            "head_atom": head_atom
        }
        nodes.append(rule_node)
        rule_nid = nid_counter
        nid_counter += 1
        
        # Connect body facts to rule
        for atom in body_atoms:
            if atom not in fact_map:
                fact_node = {
                    "nid": nid_counter,
                    "type": "fact",
                    "label": atom,
                    "atom": atom,
                    "is_initial": False
                }
                nodes.append(fact_node)
                fact_map[atom] = nid_counter
                nid_counter += 1
            
            edges.append({
                "src": fact_map[atom],
                "dst": rule_nid,
                "etype": "body"
            })
        
        # Connect rule to head fact
        if head_atom not in fact_map:
            fact_node = {
                "nid": nid_counter,
                "type": "fact",
                "label": head_atom,
                "atom": head_atom, 
                "is_initial": False
            }
            nodes.append(fact_node)
            fact_map[head_atom] = nid_counter
            nid_counter += 1
        
        edges.append({
            "src": rule_nid,
            "dst": fact_map[head_atom],
            "etype": "head"
        })
        
        derived_facts.add(head_atom)
    
    # STEP 3: Generate proof by forward chaining
    known = set(initial_atoms)
    step_id = 0
    
    for depth in range(p["max_depth"]):
        fired = False
        
        for node in [n for n in nodes if n["type"] == "rule"]:
            rule_nid = node["nid"]
            body = node["body_atoms"]
            head = node["head_atom"]
            
            # Can fire if all body atoms known and head not yet derived
            if all(a in known for a in body) and head not in known:
                body_nids = [fact_map[a] for a in body]
                head_nid = fact_map[head]
                
                proof_steps.append({
                    "step_id": step_id,
                    "derived_node": head_nid,
                    "used_rule": rule_nid,
                    "premises": body_nids
                })
                
                known.add(head)
                step_id += 1
                fired = True
        
        if not fired:
            break
    
    # Metadata
    metadata = {
        "difficulty": difficulty.value,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_initial_facts": len(initial_atoms),
        "n_rules": len([n for n in nodes if n["type"] == "rule"]),
        "proof_length": len(proof_steps),
        "source": "forward_chaining_v3_fixed"
    }
    
    return {
        "id": instance_id,
        "nodes": nodes,
        "edges": edges,
        "proof_steps": proof_steps,
        "metadata": metadata
    }


def generate_dataset(
    output_dir: str,
    n_per_difficulty: Dict[Difficulty, int],
    seed: int = 42
):
    """Generate complete dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "by_difficulty": {}}
    
    for diff, count in n_per_difficulty.items():
        diff_dir = output_dir / diff.value
        diff_dir.mkdir(exist_ok=True)
        
        print(f"Generating {count} {diff.value} instances...")
        
        for i in range(count):
            inst = generate_horn_instance(
                f"{diff.value}_{i}",
                diff,
                seed + i + hash(diff.value) % 10000
            )
            
            path = diff_dir / f"{diff.value}_{i}.json"
            with open(path, 'w') as f:
                json.dump(inst, f, indent=2)
        
        stats["by_difficulty"][diff.value] = count
        stats["total"] += count
        print(f"  ✓ {count} instances")
    
    print(f"\nTotal: {stats['total']} instances")
    return stats


if __name__ == "__main__":
    generate_dataset(
        "data/horn",
        {
            Difficulty.EASY: 400,
            Difficulty.MEDIUM: 400,
            Difficulty.HARD: 300,
            Difficulty.VERY_HARD: 300,
            # Difficulty.EXTREME_HARD: 200
        },
        seed=42
    )