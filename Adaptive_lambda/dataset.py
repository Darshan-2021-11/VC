import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# =========================
# TRAIN DATASET
# =========================

class DualDegradationDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for cat in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_path):
                continue

            for scene in os.listdir(cat_path):
                scene_path = os.path.join(cat_path, scene)
                if not os.path.isdir(scene_path):
                    continue

                input_path = os.path.join(scene_path, "0.png")
                target_path = os.path.join(scene_path, "1.png")  # TRAIN: 0 → 1

                if os.path.exists(input_path) and os.path.exists(target_path):
                    self.samples.append((input_path, target_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No training pairs found under: {root_dir}")

        print(f"[DualDegradationDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]

        deg = Image.open(input_path).convert("RGB")
        clean = Image.open(target_path).convert("RGB")

        deg = torch.from_numpy(np.array(deg)).permute(2,0,1).float() / 255.
        clean = torch.from_numpy(np.array(clean)).permute(2,0,1).float() / 255.

        return deg, clean


# =========================
# TEST DATASET
# =========================

class DualDegradationTestDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        expected_categories = ["Cloudy_to_Rainy", "Sunny_to_Foggy", "Sunny_to_Rainy"]
        missing = []

        for cat in expected_categories:
            cat_path = os.path.join(root_dir, cat)

            if not os.path.exists(cat_path):
                missing.append(cat)
                continue

            for scene in os.listdir(cat_path):
                scene_path = os.path.join(cat_path, scene)

                if not os.path.isdir(scene_path):
                    continue

                input_path = os.path.join(scene_path, "0.png")
                target_path = os.path.join(scene_path, "1.png")  # TEST: 0 → 1 (FIXED)

                if os.path.exists(input_path) and os.path.exists(target_path):
                    self.samples.append((input_path, target_path, cat, scene))

        if missing:
            print(f"[WARN] Missing category folders: {missing}")
            print(f"       Expected under: {root_dir}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No test pairs found under: {root_dir}")

        print(f"[TestDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path, cat, scene = self.samples[idx]

        deg = Image.open(input_path).convert("RGB")
        clean = Image.open(target_path).convert("RGB")

        deg = torch.from_numpy(np.array(deg)).permute(2,0,1).float() / 255.
        clean = torch.from_numpy(np.array(clean)).permute(2,0,1).float() / 255.

        return deg, clean, cat, scene, idx
