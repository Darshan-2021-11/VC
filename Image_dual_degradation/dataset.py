import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CATEGORY_FOLDERS  = ["Cloudy_to_Rainy", "Sunny_to_Foggy", "Sunny_to_Rainy"]
DEGRADED_INDICES  = [str(i) for i in range(1, 10)]   # "1".."9"
CLEAN_FILENAME    = "0.png"


# ─────────────────────────────────────────────────────────────
# COLLECT DATA
# ─────────────────────────────────────────────────────────────

def _collect_pairs(root_dir: str):
    """
    Collect:
    (degraded_path, clean_path, category, scene, level)
    """
    pairs = []

    for cat in CATEGORY_FOLDERS:
        cat_path = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_path):
            print(f"[WARN] Missing category: {cat}")
            continue

        scenes = sorted([
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
            ])

        for scene in scenes:
            scene_path = os.path.join(cat_path, scene)
            clean_path = os.path.join(scene_path, CLEAN_FILENAME)

            if not os.path.exists(clean_path):
                print(f"[WARN] Missing clean image in {scene_path}")
                continue

            for level in DEGRADED_INDICES:
                deg_path = os.path.join(scene_path, f"{level}.png")
                if os.path.exists(deg_path):
                    pairs.append({
                        "deg": deg_path,
                        "clean": clean_path,
                        "cat": cat,
                        "scene": scene,
                        "level": level
                        })

    if len(pairs) == 0:
        raise RuntimeError(f"No dataset found at {root_dir}")

    print(f"[Dataset] Total pairs: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────────────────────
# TRAIN DATASET (OTS STYLE)
# ─────────────────────────────────────────────────────────────

class DualDegradationDataset(Dataset):
    def __init__(self, root_dir: str, patch_size: int = 256):
        self.patch_size = patch_size
        self.pairs = _collect_pairs(root_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        deg   = Image.open(p["deg"]).convert("RGB")
        clean = Image.open(p["clean"]).convert("RGB")

        # resize if too small
        W, H = deg.size
        ps = self.patch_size
        if W < ps or H < ps:
            scale = max(ps / W, ps / H) + 0.01
            nW, nH = int(W * scale), int(H * scale)
            deg   = deg.resize((nW, nH), Image.BICUBIC)
            clean = clean.resize((nW, nH), Image.BICUBIC)
            W, H = deg.size

        # random crop
        x = random.randint(0, W - ps)
        y = random.randint(0, H - ps)
        deg   = TF.crop(deg,   y, x, ps, ps)
        clean = TF.crop(clean, y, x, ps, ps)

        # augment
        if random.random() > 0.5:
            deg, clean = TF.hflip(deg), TF.hflip(clean)
        if random.random() > 0.5:
            deg, clean = TF.vflip(deg), TF.vflip(clean)

        return TF.to_tensor(deg), TF.to_tensor(clean)


# ─────────────────────────────────────────────────────────────
# TEST DATASET (OTS STYLE)
# ─────────────────────────────────────────────────────────────

class DualDegradationTestDataset(Dataset):
    def __init__(self, root_dir: str):
        self.pairs = _collect_pairs(root_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        deg   = TF.to_tensor(Image.open(p["deg"]).convert("RGB"))
        clean = TF.to_tensor(Image.open(p["clean"]).convert("RGB"))

        # 🔥 OTS-style filename
        name = f"{p['cat']}_{p['scene']}_{p['level']}.png"

        return deg, clean, name
