import os
import csv
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.ConvIR import ConvIR
from dataset import DualDegradationTestDataset


def tensor_to_img(t):
    return (t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvIR(num_res=8).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    dataset = DualDegradationTestDataset("Dataset/Test_Dataset")

    os.makedirs("results", exist_ok=True)

    log_file = "results/test_metrics.csv"

    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "psnr", "ssim"])

    psnr_list = []
    ssim_list = []

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

        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([f"{cat}_{scene}_{level}", p, s])

        Image.fromarray(r).save(f"results/{cat}_{scene}_{level}.png")

    print("\nFINAL:")
    print("PSNR:", np.mean(psnr_list))
    print("SSIM:", np.mean(ssim_list))

    with open("results/summary.txt", "w") as f:
        f.write(f"PSNR: {np.mean(psnr_list)}\n")
        f.write(f"SSIM: {np.mean(ssim_list)}\n")


if __name__ == "__main__":
    main()
