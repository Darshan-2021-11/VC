import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F   # ✅ ADDED
from PIL import Image
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ConvIR import ConvIR
from dataset import DualDegradationTestDataset, CATEGORY_FOLDERS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_size", type=str, default="S", choices=["S","B","L"])
    p.add_argument("--result_dir", type=str, default="./results")
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--tile", type=int, default=0)
    return p.parse_args()


def pad_to_multiple(x, factor=8):
    _, _, h, w = x.shape
    new_h = (h + factor - 1) // factor * factor
    new_w = (w + factor - 1) // factor * factor

    pad_h = new_h - h
    pad_w = new_w - w

    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w


def infer(model, inp, tile):
    if tile <= 0:
        with torch.no_grad():
            out = model(inp)
        return out[-1] if isinstance(out,(list,tuple)) else out

    _, _, H, W = inp.shape
    stride = tile // 2
    out = torch.zeros_like(inp)
    cnt = torch.zeros_like(inp)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            ys = min(y, H - tile) if H > tile else 0
            xs = min(x, W - tile) if W > tile else 0
            ye = min(ys + tile, H)
            xe = min(xs + tile, W)

            patch = inp[:, :, ys:ye, xs:xe]
            with torch.no_grad():
                r = model(patch)
                r = r[-1] if isinstance(r,(list,tuple)) else r

            out[:, :, ys:ye, xs:xe] += r
            cnt[:, :, ys:ye, xs:xe] += 1

    return out / cnt.clamp(min=1)


def tensor_to_uint8(t):
    return (t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_RES = {"S":8,"B":16,"L":20}[args.model_size]

    model = ConvIR(num_res=NUM_RES).to(device)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print(f"Loaded : {args.model_path}")
    print(f"Device : {device}")

    os.makedirs(args.result_dir, exist_ok=True)

    dataset = DualDegradationTestDataset(args.test_dir)

    cat_data = defaultdict(lambda: {"psnr":[], "ssim":[]})
    level_data = defaultdict(lambda: {"psnr":[], "ssim":[]})

    for idx in range(len(dataset)):
        deg, clean, cat, scene, level = dataset[idx]

        inp = deg.unsqueeze(0).to(device)
        inp, orig_h, orig_w = pad_to_multiple(inp, 8)

        restored = infer(model, inp, args.tile)

        restored = restored[:, :, :orig_h, :orig_w]

        r_u8 = tensor_to_uint8(restored)
        c_u8 = tensor_to_uint8(clean.unsqueeze(0))

        psnr = calc_psnr(c_u8, r_u8, data_range=255)
        ssim = calc_ssim(c_u8, r_u8, data_range=255, channel_axis=2)

        cat_data[cat]["psnr"].append(psnr)
        cat_data[cat]["ssim"].append(ssim)
        level_data[level]["psnr"].append(psnr)
        level_data[level]["ssim"].append(ssim)

        if args.save_images:
            save_dir = os.path.join(args.result_dir, cat, scene)
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(r_u8).save(
                os.path.join(save_dir, f"{level}_restored.png")
            )

        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1}/{len(dataset)}")

    all_psnr = []
    all_ssim = []

    for cat in cat_data:
        all_psnr += cat_data[cat]["psnr"]
        all_ssim += cat_data[cat]["ssim"]

    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)

    print(f"\nFINAL RESULTS:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")

    with open(os.path.join(args.result_dir, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")


if __name__ == "__main__":
    main()
