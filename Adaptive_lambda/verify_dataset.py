"""
verify_dataset.py
=================
Run this FIRST (locally and on Colab) before training.
It checks the dataset structure, counts pairs, and loads one batch
to make sure everything is wired correctly.

Usage:
    python verify_dataset.py --root ../Dataset/Training_Dataset
    python verify_dataset.py --root ../Dataset/Test_Dataset --test
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (
    DualDegradationDataset,
    DualDegradationTestDataset,
    CATEGORY_FOLDERS,
    CLEAN_FILENAME,
    DEGRADED_INDICES,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",       type=str, required=True,
                   help="Path to Training_Dataset or Test_Dataset")
    p.add_argument("--test",       action="store_true",
                   help="Verify test dataset (no crop/augment)")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f" Dataset verification")
    print(f" Root: {args.root}")
    print(f"{'='*60}\n")

    # ── Raw structure check ───────────────────────────────────────────────────
    if not os.path.isdir(args.root):
        print(f"ERROR: Root directory does not exist:\n  {args.root}")
        sys.exit(1)

    total_scenes = 0
    total_pairs  = 0

    for cat in CATEGORY_FOLDERS:
        cat_path = os.path.join(args.root, cat)
        if not os.path.isdir(cat_path):
            print(f"  [MISSING]  {cat}")
            continue

        scenes     = sorted(d for d in os.listdir(cat_path)
                           if os.path.isdir(os.path.join(cat_path, d)))
        n_scenes   = len(scenes)
        n_pairs    = 0
        no_clean   = []
        missing_deg = []

        for scene in scenes:
            scene_path = os.path.join(cat_path, scene)
            clean_p    = os.path.join(scene_path, CLEAN_FILENAME)
            if not os.path.exists(clean_p):
                no_clean.append(scene)
                continue
            found = [i for i in DEGRADED_INDICES
                     if os.path.exists(os.path.join(scene_path, f"{i}.png"))]
            if not found:
                missing_deg.append(scene)
            n_pairs += len(found)

        total_scenes += n_scenes
        total_pairs  += n_pairs

        print(f"  [OK]  {cat}")
        print(f"        scenes        : {n_scenes}")
        print(f"        pairs         : {n_pairs}")
        if no_clean:
            print(f"        [WARN] missing {CLEAN_FILENAME}: {len(no_clean)} scenes")
        if missing_deg:
            print(f"        [WARN] missing deg images  : {len(missing_deg)} scenes")
        print()

    print(f"  Total scenes : {total_scenes}")
    print(f"  Total pairs  : {total_pairs}")

    if total_pairs == 0:
        print("\nERROR: No pairs found. Check directory path and structure.")
        sys.exit(1)

    # ── Dataset class check ───────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(" Loading via Dataset class...")

    try:
        if args.test:
            ds = DualDegradationTestDataset(args.root)
            item = ds[0]
            deg, clean, cat, scene, level = item
            print(f"\n  Test dataset item:")
            print(f"    category  : {cat}")
            print(f"    scene     : {scene}")
            print(f"    deg level : {level}")
            print(f"    deg shape : {tuple(deg.shape)}")
            print(f"    clean shape: {tuple(clean.shape)}")
        else:
            ds = DualDegradationDataset(args.root, patch_size=args.patch_size)
            loader = DataLoader(ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
            batch_deg, batch_clean = next(iter(loader))
            print(f"\n  Training batch shapes:")
            print(f"    degraded : {tuple(batch_deg.shape)}")
            print(f"    clean    : {tuple(batch_clean.shape)}")
            print(f"    deg  value range: [{batch_deg.min():.3f}, {batch_deg.max():.3f}]")
            print(f"    clean value range: [{batch_clean.min():.3f}, {batch_clean.max():.3f}]")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(" Verification PASSED — dataset is ready.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
