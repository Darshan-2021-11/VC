import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ConvIR import ConvIR
from dataset import DualDegradationTestDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_size", type=str, default="S", choices=["S","B","L"])
    p.add_argument("--result_dir", type=str, default="./results")
    p.add_argument("--save_images", action="store_true")
    return p.parse_args()


def infer(model, inp):
    with torch.no_grad():
        out = model(inp)
    return out[-1] if isinstance(out,(list,tuple)) else out


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

    psnr_list = []
    ssim_list = []

    for idx in range(len(dataset)):
        deg, clean, cat, scene, level = dataset[idx]

        inp = deg.unsqueeze(0).to(device)

        # 🔥 FIX: resize to training resolution (256)
        orig_h, orig_w = inp.shape[-2:]
        inp_resized = F.interpolate(inp, size=(256, 256), mode='bilinear', align_corners=False)

        # inference
        restored = infer(model, inp_resized)

        # 🔥 resize back
        restored = F.interpolate(restored, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        r_u8 = tensor_to_uint8(restored)
        c_u8 = tensor_to_uint8(clean.unsqueeze(0))

        psnr = calc_psnr(c_u8, r_u8, data_range=255)
        ssim = calc_ssim(c_u8, r_u8, data_range=255, channel_axis=2)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        if args.save_images:
            save_dir = os.path.join(args.result_dir, cat, scene)
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(r_u8).save(
                    os.path.join(save_dir, f"{level}_restored.png")
                    )

        if (idx+1) % 20 == 0:
            print(f"Processed {idx+1}/{len(dataset)}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\nFINAL RESULTS:")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")

    with open(os.path.join(args.result_dir, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")


if __name__ == "__main__":
    main()
