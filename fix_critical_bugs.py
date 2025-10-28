# fix_critical_bugs.py - Apply all critical fixes

import re

# Fix 1: dataset.py - instance_id_for_spectral
with open('dataset.py', 'r') as f:
    content = f.read()

# Find and replace the __getitem__ method's beginning
old_pattern = r"def __getitem__\(self, idx: int\) -> Data:\s+inst, step_idx, has_spectral, metadata = self\.samples\[idx\]"
new_code = """def __getitem__(self, idx: int) -> Data:
        inst, step_idx, has_spectral, metadata = self.samples[idx]
        
        # FIX: Define instance_id_for_spectral FIRST
        instance_id_for_spectral = inst.get("id")
        if not instance_id_for_spectral:
            instance_id_for_spectral = metadata.get("id", f"unknown_instance_{idx}")"""

content = re.sub(old_pattern, new_code, content)

with open('dataset.py', 'w') as f:
    f.write(content)

print("✓ Fixed dataset.py")

# Fix 2: model.py - Spectral filter dimension
with open('model.py', 'r') as f:
    model_content = f.read()

# Replace filter_generator output dimension
model_content = model_content.replace(
    "nn.Linear(64, in_dim),",
    "nn.Linear(64, 1),"
)

# Fix the forward pass
old_filter_apply = "filtered_freq = filters * x_freq"
new_filter_apply = "filtered_freq = filters.squeeze(-1).unsqueeze(-1) * x_freq"
model_content = model_content.replace(old_filter_apply, new_filter_apply)

with open('model.py', 'w') as f:
    f.write(model_content)

print("✓ Fixed model.py")

# Fix 3: temporal_encoder.py - Attention dimension check
# (Manual edit required - complex regex)
print("⚠ temporal_encoder.py requires manual edit - see above")

print("\n✅ Applied critical fixes")
print("⚠ Still need to manually fix ProofFrontierAttention.forward()")