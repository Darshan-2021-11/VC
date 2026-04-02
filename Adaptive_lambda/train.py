import os
import sys
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ConvIR import ConvIR
from dataset import DualDegradationDataset
from losses import dual_domain_loss


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--model_size", type=str, default="S", choices=["S","B","L"])

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/ConvIR_checkpoints/adaptive_lambda")  # Google Drive path
    p.add_argument("--save_every", type=int, default=1)

    p.add_argument("--fixed_lambda", action="store_true")

    return p.parse_args()


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
        }, path)


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_RES = {"S":8,"B":16,"L":20}[args.model_size]
    model = ConvIR(num_res=NUM_RES).to(device)

    dataset = DualDegradationDataset(args.train_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.drive_dir:
        os.makedirs(args.drive_dir, exist_ok=True)

    log_file = os.path.join(args.save_dir, "train_log.csv")

    # create CSV header
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "lambda"])

    for epoch in range(1, args.num_epochs+1):
        model.train()
        total_loss = 0
        total_lambda = 0

        for deg, clean in loader:
            deg, clean = deg.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(deg)

            if isinstance(pred, (list, tuple)):
                loss = 0
                lam_val = 0

                for p in pred:
                    target_resized = F.interpolate(clean, size=p.shape[-2:], mode='bilinear')
                    l, lam = dual_domain_loss(p, target_resized, adaptive=not args.fixed_lambda)
                    loss += l
                    lam_val += lam

                loss /= len(pred)
                lam_val /= len(pred)
            else:
                loss, lam_val = dual_domain_loss(pred, clean, adaptive=not args.fixed_lambda)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_lambda += lam_val

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_lambda = total_lambda / len(loader)

        print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Lambda {avg_lambda:.4f}")

        # save log
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_lambda])

        # save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

            if args.drive_dir:
                drive_path = os.path.join(args.drive_dir, f"epoch_{epoch}.pth")
                save_checkpoint(model, optimizer, epoch, avg_loss, drive_path)
                print(f"Saved to Drive: {drive_path}")

    # final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))


if __name__ == "__main__":
    main()
