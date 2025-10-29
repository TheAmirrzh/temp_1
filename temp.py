import json

# Corrected hypothetical invalid sample (missing body premise)
sample_json = '''
{
  "id": "test_invalid",
  "nodes": [
    {"nid": "f1", "type": "fact", "atom": "A", "is_initial": true},
    {"nid": "r1", "type": "rule", "body_atoms": ["B"], "head_atom": "C"},  # Missing "B"
    {"nid": "f2", "type": "fact", "atom": "C"}
  ],
  "edges": [],
  "proof_steps": [{"derived_node": "f2", "used_rule": "r1", "premises": ["f1"]}]
}
'''
data = json.loads(sample_json)
nodes = data["nodes"]
proof_steps = data["proof_steps"]

# Simplified verifier
known_facts = set()
for node in nodes:
    if node.get("is_initial", False) and node["type"] == "fact":
        known_facts.add(node["atom"])

for step in proof_steps:
    rule_nid = step["used_rule"]
    for node in nodes:
        if node["nid"] == rule_nid and node["type"] == "rule":
            body = set(node["body_atoms"])
            head = node["head_atom"]
            if not body.issubset(known_facts):
                print("Missing premises:", body - known_facts)  # Expected: {'B'}
            if head in known_facts:
                print("Head already known")
            known_facts.add(head)
print("Known facts after proof:", known_facts)