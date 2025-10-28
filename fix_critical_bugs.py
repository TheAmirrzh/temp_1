import torch
from model import FixedTemporalSpectralGNN, TacticGuidedGNN

# Test single-graph value shape
model = FixedTemporalSpectralGNN(in_dim=22, hidden_dim=64, k=16)
model.eval()

x = torch.randn(10, 22)
edge_index = torch.randint(0, 10, (2, 20))
derived_mask = torch.randint(0, 2, (10,)).bool()
step_numbers = torch.randint(0, 10, (10,))
eigvecs = torch.randn(10, 16)
eigvals = torch.randn(16)

# Test with batch=None (single graph)
scores, emb, value = model(x, edge_index, derived_mask, step_numbers, 
                            eigvecs, eigvals, batch=None)

print(f"Single graph:")
print(f"  scores shape: {scores.shape}")  # Should be [10]
print(f"  embeddings shape: {emb.shape}")  # Should be [10, 192]
print(f"  value shape: {value.shape}")  # Should be [1] ← THIS WAS FAILING

assert value.shape == torch.Size([1]), f"Expected [1], got {value.shape}"

# Test with batch (2 graphs)
batch = torch.cat([torch.zeros(5), torch.ones(5)]).long()
scores_b, emb_b, value_b = model(x, edge_index, derived_mask, step_numbers,
                                   eigvecs, eigvals, batch=batch)

print(f"\nBatched (2 graphs):")
print(f"  scores shape: {scores_b.shape}")  # Should be [10]
print(f"  embeddings shape: {emb_b.shape}")  # Should be [10, 192]
print(f"  value shape: {value_b.shape}")  # Should be [2]

assert value_b.shape == torch.Size([2]), f"Expected [2], got {value_b.shape}"

print("\n✅ All shape tests passed!")