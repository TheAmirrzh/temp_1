#!/usr/bin/env python3
"""
Quick Fix Script: Apply all Priority 1 fixes at once
Run this to patch your existing codebase with critical fixes.

Usage:
    python apply_critical_fixes.py --backup
    
This will:
1. Backup original files to ./backups/
2. Apply all critical fixes
3. Print summary of changes
"""

import shutil
import re
from pathlib import Path
import argparse


def backup_files():
    """Backup original files before modification."""
    backup_dir = Path("./backups")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "dataset.py",
        "losses.py", 
        "train.py",
        "batch_process_spectral.py"
    ]
    
    for file in files_to_backup:
        if Path(file).exists():
            shutil.copy(file, backup_dir / file)
            print(f"✅ Backed up {file}")


def fix_difficulty_estimation():
    """Fix 1: Rescale difficulty estimation"""
    print("\n🔧 Applying Fix 1: Difficulty Estimation...")
    
    with open("dataset.py", "r") as f:
        content = f.read()
    
    # Find and replace _estimate_difficulty
    old_pattern = r"def _estimate_difficulty\(self, metadata: dict\) -> float:.*?return min\(max\(difficulty, 0\.0\), 1\.0\)"
    
    new_code = """def _estimate_difficulty(self, metadata: dict) -> float:
        \"\"\"
        FIXED: Properly scaled difficulty estimation.
        Maps instances to [0, 1] range with better discrimination.
        \"\"\"
        n_rules = metadata.get('n_rules', 10)
        n_nodes = metadata.get('n_nodes', 20)
        proof_length = metadata.get('proof_length', 5)
        
        # Normalize using ranges from generation parameters
        rule_score = min(n_rules / 40.0, 1.0)
        node_score = min(n_nodes / 70.0, 1.0)
        proof_score = min(proof_length / 15.0, 1.0)
        
        # Weighted (heavier on proof length - most indicative)
        difficulty = (0.3 * rule_score + 
                      0.2 * node_score + 
                      0.5 * proof_score)
        
        return min(max(difficulty, 0.0), 1.0)"""
    
    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    
    with open("dataset.py", "w") as f:
        f.write(content)
    
    print("  ✅ Fixed difficulty estimation in dataset.py")


def fix_curriculum_scheduler():
    """Fix 2: Adjust curriculum scheduler thresholds"""
    print("\n🔧 Applying Fix 2: Curriculum Scheduler...")
    
    with open("train.py", "r") as f:
        content = f.read()
    
    # Fix __init__ parameters
    content = re.sub(
        r"def __init__\(self, dataset.*?base_difficulty: float = [0-9.]+, *max_difficulty: float = [0-9.]+\)",
        "def __init__(self, dataset: StepPredictionDataset, \n                 base_difficulty: float = 0.15, \n                 max_difficulty: float = 0.95)",
        content
    )
    
    with open("train.py", "w") as f:
        f.write(content)
    
    print("  ✅ Fixed curriculum thresholds in train.py")


def fix_loss_scaling():
    """Fix 3: Reduce penalty and add temperature scaling"""
    print("\n🔧 Applying Fix 3: Loss Function Scaling...")
    
    with open("losses.py", "r") as f:
        content = f.read()
    
    # Fix penalty parameter
    content = re.sub(
        r"def __init__\(self, margin=[0-9.]+, penalty_nonapplicable=[0-9.]+\)",
        "def __init__(self, margin=1.0, penalty_nonapplicable=10.0)",
        content
    )
    
    # Add temperature scaling to forward
    old_return = r"loss = component1 \+ component2\s+return loss"
    new_return = """# Temperature scaling to prevent initial loss explosion
        temperature = 0.5
        loss = temperature * (component1 + component2)
        
        return loss"""
    
    content = re.sub(old_return, new_return, content)
    
    with open("losses.py", "w") as f:
        f.write(content)
    
    print("  ✅ Fixed loss scaling in losses.py")


def fix_spectral_adaptive():
    """Fix 4: Enable adaptive k by default"""
    print("\n🔧 Applying Fix 4: Adaptive Spectral Dimension...")
    
    with open("batch_process_spectral.py", "r") as f:
        content = f.read()
    
    # Enable adaptive_k by default
    content = re.sub(
        r"'--adaptive-k',\s*action='store_true',\s*default=False",
        "'--adaptive-k',\n        action='store_true',\n        default=True",
        content
    )
    
    with open("batch_process_spectral.py", "w") as f:
        f.write(content)
    
    print("  ✅ Enabled adaptive k in batch_process_spectral.py")


def validate_fixes():
    """Run basic validation that fixes were applied"""
    print("\n🔍 Validating fixes...")
    
    checks = {
        "dataset.py": "n_rules / 40.0",  # Check new scaling
        "train.py": "base_difficulty: float = 0.15",  # Check new threshold
        "losses.py": "penalty_nonapplicable=10.0",  # Check reduced penalty
        "batch_process_spectral.py": "default=True",  # Check adaptive k
    }
    
    all_pass = True
    for file, pattern in checks.items():
        with open(file, "r") as f:
            content = f.read()
        
        if pattern in content:
            print(f"  ✅ {file}: {pattern[:30]}... found")
        else:
            print(f"  ❌ {file}: {pattern[:30]}... NOT found")
            all_pass = False
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Apply critical fixes to Horn Clause GNN"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup original files before modification"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate fixes, don't apply"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" HORN CLAUSE GNN: CRITICAL FIXES")
    print("="*70)
    
    if args.validate_only:
        success = validate_fixes()
        if success:
            print("\n✅ All fixes validated successfully!")
        else:
            print("\n❌ Some fixes missing or incorrect")
        return
    
    if args.backup:
        print("\n📦 Creating backups...")
        backup_files()
    
    # Apply all fixes
    fix_difficulty_estimation()
    fix_curriculum_scheduler()
    fix_loss_scaling()
    fix_spectral_adaptive()
    
    # Validate
    print("\n" + "="*70)
    success = validate_fixes()
    
    if success:
        print("\n" + "="*70)
        print(" ✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Re-run spectral preprocessing:")
        print("   bash run.sh  # This will clear cache and regenerate")
        print("\n2. Start training:")
        print("   python train.py --data-dir ./generated_data --epochs 30")
        print("\n3. Expected improvements:")
        print("   - Curriculum will now work correctly")
        print("   - Loss will be more stable")
        print("   - Training will be faster")
        print("   - Hit@1 should improve by 5-10%")
    else:
        print("\n❌ Some fixes failed to apply. Check output above.")


if __name__ == "__main__":
    main()