import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.ConvIR import ConvIR
from dataset import DualDegradationTestDataset


def tensor_to_img(t):
    return (t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_size", type=str, default="S", choices=["S","B","L"])
    p.add_argument("--result_dir", type=str, default="./results")
    p.add_argument("--save_images", action="store_true")

    return p.parse_args()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_RES = {"S":8,"B":16,"L":20}[args.model_size]
    model = ConvIR(num_res=NUM_RES).to(device)

    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    dataset = DualDegradationTestDataset(args.test_dir)

    os.makedirs(args.result_dir, exist_ok=True)

    # 🔥 CSV logging
    csv_path = os.path.join(args.result_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "psnr", "ssim"])

    psnr_list = []
    ssim_list = []

    print("\n========== TESTING START ==========\n")

    for i in range(len(dataset)):
        deg, clean, cat, scene, level = dataset[i]

        inp = deg.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)[-1]

        r = tensor_to_img(out)
        c = tensor_to_img(clean.unsqueeze(0))

        p = psnr(c, r, data_range=255)
        s = ssim(c, r, channel_axis=2, data_range=255)

        psnr_list.append(p)
        ssim_list.append(s)

        image_name = f"{cat}_{scene}_{level}"

        # save per-image metrics
        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([image_name, p, s])

        # optionally save images
        if args.save_images:
            save_path = os.path.join(args.result_dir, image_name + ".png")
            Image.fromarray(r).save(save_path)

        if i % 20 == 0:
            print(f"[{i}/{len(dataset)}] {image_name} | PSNR: {p:.2f} | SSIM: {s:.3f}")

    # 🔥 Final metrics
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)

    print("\n========== FINAL RESULTS ==========")
    print(f"PSNR: {mean_psnr:.4f}")
    print(f"SSIM: {mean_ssim:.4f}")

    # 🔥 Save summary
    summary_path = os.path.join(args.result_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("FINAL RESULTS\n")
        f.write(f"PSNR: {mean_psnr:.4f}\n")
        f.write(f"SSIM: {mean_ssim:.4f}\n")

    print(f"\nSaved results to: {args.result_dir}")


if __name__ == "__main__":
    main()
