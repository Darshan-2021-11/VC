"""
DualDegradationDataset
======================

Dataset structure (what we discovered from tree.txt):

  Dataset/
  ├── Training_Dataset/
  │   ├── Cloudy_to_Rainy/
  │   │   ├── 294 to 1/          ← scene group folder
  │   │   │   ├── 0.png          ← CLEAN ground truth (undegraded)
  │   │   │   ├── 1.png          ← degraded (light)
  │   │   │   ├── ...
  │   │   │   └── 9.png          ← degraded (heavy)
  │   │   └── 301 to 8/
  │   │       └── ...
  │   ├── Sunny_to_Foggy/
  │   └── Sunny_to_Rainy/
  └── Test_Dataset/
      └── (same structure)

Each (k.png, 0.png) for k in 1..9 is one training pair.
So one scene folder yields 9 pairs.
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

CATEGORY_FOLDERS  = ["Cloudy_to_Rainy", "Sunny_to_Foggy", "Sunny_to_Rainy"]
DEGRADED_INDICES  = [str(i) for i in range(1, 10)]   # "1".."9"
CLEAN_FILENAME    = "0.png"
IMG_EXTENSIONS    = (".png", ".jpg", ".jpeg", ".bmp")


def _collect_pairs(root_dir: str):
    """
    Walk root_dir/<category>/<scene_group>/ and collect
    (degraded_path, clean_path, category, scene_group, degradation_level) tuples.

    Returns list of dicts for clarity.
    """
    pairs = []
    missing_cats = []

    for cat in CATEGORY_FOLDERS:
        cat_path = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_path):
            missing_cats.append(cat)
            continue

        scene_groups = sorted([
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        ])

        if not scene_groups:
            print(f"[WARN] No scene-group subfolders found in {cat_path}")
            continue

        for scene in scene_groups:
            scene_path = os.path.join(cat_path, scene)
            clean_path = os.path.join(scene_path, CLEAN_FILENAME)

            if not os.path.exists(clean_path):
                print(f"[WARN] No {CLEAN_FILENAME} in {scene_path}, skipping.")
                continue

            for deg_idx in DEGRADED_INDICES:
                deg_path = os.path.join(scene_path, f"{deg_idx}.png")
                if os.path.exists(deg_path):
                    pairs.append({
                        "degraded":  deg_path,
                        "clean":     clean_path,
                        "category":  cat,
                        "scene":     scene,
                        "deg_level": int(deg_idx),
                    })

    if missing_cats:
        print(f"[WARN] Missing category folders: {missing_cats}")
        print(f"       Expected under: {root_dir}")

    return pairs


class DualDegradationDataset(Dataset):
    """
    Training dataset.
    Returns random (patch_size x patch_size) crops with horizontal/vertical flip.
    Both degraded and clean patches use the same random crop coordinates.
    """

    def __init__(self, root_dir: str, patch_size: int = 256):
        self.patch_size = patch_size
        self.pairs      = _collect_pairs(root_dir)

        if not self.pairs:
            raise RuntimeError(
                f"No image pairs found under: {root_dir}\n"
                f"Expected structure: {root_dir}/<category>/<scene>/{CLEAN_FILENAME}\n"
                f"                   {root_dir}/<category>/<scene>/1.png .. 9.png\n"
                f"Categories looked for: {CATEGORY_FOLDERS}"
            )

        self._print_stats()

    def _print_stats(self):
        from collections import Counter
        counts = Counter(p["category"] for p in self.pairs)
        print("[DualDegradationDataset] Pairs loaded:")
        for cat in CATEGORY_FOLDERS:
            print(f"  {cat:<30}: {counts.get(cat, 0):>5} pairs")
        print(f"  {'TOTAL':<30}: {len(self.pairs):>5} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        deg   = Image.open(pair["degraded"]).convert("RGB")
        clean = Image.open(pair["clean"]).convert("RGB")

        # Upscale if image is smaller than patch_size
        W, H = deg.size
        ps   = self.patch_size
        if W < ps or H < ps:
            scale = max(ps / W, ps / H) + 0.01
            nW, nH = int(W * scale), int(H * scale)
            deg   = deg.resize((nW, nH),   Image.BICUBIC)
            clean = clean.resize((nW, nH), Image.BICUBIC)
            W, H  = deg.size

        # Random crop — same coords for both images
        x = random.randint(0, W - ps)
        y = random.randint(0, H - ps)
        deg   = TF.crop(deg,   y, x, ps, ps)
        clean = TF.crop(clean, y, x, ps, ps)

        # Augmentation
        if random.random() > 0.5:
            deg, clean = TF.hflip(deg), TF.hflip(clean)
        if random.random() > 0.5:
            deg, clean = TF.vflip(deg), TF.vflip(clean)

        return TF.to_tensor(deg), TF.to_tensor(clean)


class DualDegradationTestDataset(Dataset):
    """
    Test / validation dataset.
    Returns full images — no crop, no augmentation.
    Also returns metadata for per-category and per-degradation-level reporting.
    """

    def __init__(self, root_dir: str):
        self.pairs = _collect_pairs(root_dir)
        if not self.pairs:
            raise RuntimeError(f"No test pairs found under: {root_dir}")
        print(f"[DualDegradationTestDataset] {len(self.pairs)} pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair  = self.pairs[idx]
        deg   = TF.to_tensor(Image.open(pair["degraded"]).convert("RGB"))
        clean = TF.to_tensor(Image.open(pair["clean"]).convert("RGB"))
        return deg, clean, pair["category"], pair["scene"], pair["deg_level"]
