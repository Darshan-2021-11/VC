"""
test.py — ConvIR Dual Degradation Evaluation
=============================================

Reports PSNR and SSIM:
  - Per degradation category (Cloudy_to_Rainy, Sunny_to_Foggy, Sunny_to_Rainy)
  - Per degradation level    (1 = light  ...  9 = heavy)
  - Overall

Usage:
    python test.py \
        --test_dir  ../Dataset/Test_Dataset \
        --model_path ./checkpoints/best_model.pth \
        --model_size S

    # if GPU OOM on large images, tile inference:
    python test.py ... --tile 512

    # save restored images:
    python test.py ... --save_images
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ConvIR import ConvIR
from dataset import DualDegradationTestDataset, CATEGORY_FOLDERS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir",    type=str, required=True)
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--model_size",  type=str, default="S", choices=["S", "B", "L"])
    p.add_argument("--save_images", action="store_true",
                   help="Save restored images to ./results/<category>/")
    p.add_argument("--tile",        type=int, default=0,
                   help="Tile size for inference on large images (0 = full image)")
    return p.parse_args()


def infer(model, inp: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Forward pass, with optional tiled inference for large images."""
    if tile_size <= 0:
        with torch.no_grad():
            out = model(inp)
        return out[-1] if isinstance(out, (list, tuple)) else out

    # Tiled inference — avoids GPU OOM on high-res images
    _, _, H, W = inp.shape
    stride = tile_size // 2
    out = torch.zeros_like(inp)
    cnt = torch.zeros_like(inp)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            ys = min(y, H - tile_size) if H > tile_size else 0
            xs = min(x, W - tile_size) if W > tile_size else 0
            ye = min(ys + tile_size, H)
            xe = min(xs + tile_size, W)
            tile = inp[:, :, ys:ye, xs:xe]
            with torch.no_grad():
                r = model(tile)
                r = r[-1] if isinstance(r, (list, tuple)) else r
            out[:, :, ys:ye, xs:xe] += r
            cnt[:, :, ys:ye, xs:xe] += 1
    return out / cnt.clamp(min=1)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze(0).clamp(0, 1).permute(1, 2, 0)
             .cpu().numpy() * 255).astype(np.uint8)


def print_table(cat_data: dict, level_data: dict, model_tag: str):
    W = 58
    print("\n" + "=" * W)
    print(f"  {model_tag}  —  Dual Degradation Results")
    print("=" * W)

    # Per category
    print(f"\n  {'Category':<32} {'PSNR':>6} {'SSIM':>7}")
    print("  " + "-" * (W - 2))
    all_p, all_s = [], []
    for cat in CATEGORY_FOLDERS:
        if cat not in cat_data:
            continue
        ps, ss = cat_data[cat]["psnr"], cat_data[cat]["ssim"]
        all_p.extend(ps); all_s.extend(ss)
        print(f"  {cat:<32} {np.mean(ps):>6.2f} {np.mean(ss):>7.4f}")
    print("  " + "-" * (W - 2))
    print(f"  {'Overall':<32} {np.mean(all_p):>6.2f} {np.mean(all_s):>7.4f}")

    # Per degradation level
    print(f"\n  {'Deg level':<12} {'PSNR':>6} {'SSIM':>7}  {'N pairs':>8}")
    print("  " + "-" * (W - 2))
    for lv in sorted(level_data.keys()):
        ps, ss = level_data[lv]["psnr"], level_data[lv]["ssim"]
        print(f"  {lv:<12} {np.mean(ps):>6.2f} {np.mean(ss):>7.4f}  {len(ps):>8}")
    print("=" * W + "\n")


def main():
    args = get_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_RES = {"S": 8, "B": 16, "L": 20}[args.model_size]

    # Load model
    model = ConvIR(num_res=NUM_RES).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded  : {args.model_path}")
    print(f"Device  : {device}")

    dataset = DualDegradationTestDataset(args.test_dir)

    cat_data   = defaultdict(lambda: {"psnr": [], "ssim": []})
    level_data = defaultdict(lambda: {"psnr": [], "ssim": []})

    for idx in range(len(dataset)):
        deg, clean, cat, scene, deg_level = dataset[idx]

        restored = infer(model, deg.unsqueeze(0).to(device), args.tile)

        r_u8 = tensor_to_uint8(restored)
        c_u8 = tensor_to_uint8(clean.unsqueeze(0))

        p = calc_psnr(c_u8, r_u8, data_range=255)
        s = calc_ssim(c_u8, r_u8, data_range=255, channel_axis=2)

        cat_data[cat]["psnr"].append(p)
        cat_data[cat]["ssim"].append(s)
        level_data[deg_level]["psnr"].append(p)
        level_data[deg_level]["ssim"].append(s)

        if args.save_images:
            save_dir = os.path.join("./results", cat, scene)
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(r_u8).save(
                os.path.join(save_dir, f"{deg_level}_restored.png")
            )

        if (idx + 1) % 50 == 0:
            print(f"  Evaluated {idx+1}/{len(dataset)}")

    print_table(cat_data, level_data,
                f"ConvIR-{args.model_size}")


if __name__ == "__main__":
    main()
