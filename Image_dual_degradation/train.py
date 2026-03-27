"""
train.py — ConvIR Dual Degradation Training
============================================

Run locally to verify:
    python train.py --train_dir ../Dataset/Training_Dataset --num_epochs 2 --batch_size 2

Run on Colab (called from notebook):
    python train.py \
        --train_dir  /content/drive/MyDrive/Dataset/Training_Dataset \
        --drive_ckpt /content/drive/MyDrive/ConvIR_checkpoints/dual_degradation \
        --model_size S \
        --num_epochs 200

Checkpoint strategy:
  - Saved locally every --save_every epochs
  - Copied to --drive_ckpt immediately after saving (never lose more than save_every epochs)
  - best_model.pth always mirrors the lowest-loss checkpoint
  - On re-run, auto-resumes from the latest epoch_XXXX.pth in drive_ckpt
"""

import os
import sys
import shutil
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add repo root to path so 'models' is importable when run from task folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ConvIR import ConvIR
from dataset import DualDegradationDataset

try:
    from warmup_scheduler import GradualWarmupScheduler
    HAS_WARMUP = True
except ImportError:
    HAS_WARMUP = False
    print("[INFO] warmup_scheduler not found — using plain cosine annealing.")


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="ConvIR dual-degradation trainer")

    # Paths
    p.add_argument("--train_dir",  type=str, required=True,
                   help="Path to Training_Dataset folder")
    p.add_argument("--local_ckpt", type=str, default="./checkpoints",
                   help="Local folder to write checkpoint files")
    p.add_argument("--drive_ckpt", type=str, default="/content/drive/MyDrive/ConvIR_checkpoints/dual_degradation",
                   help="Google Drive folder to mirror checkpoints into. "
                        "Leave empty when running locally.")
    p.add_argument("--resume",     type=str, default="",
                   help="Explicit checkpoint path to resume from. "
                        "If empty, auto-detects latest in drive_ckpt.")

    # Model
    # num_res = residual blocks per EBlock/DBlock (ConvIR.__init__ param)
    # default in repo is 16. We expose it directly.
    # Recommended: S=8 (fast), B=16 (default/paper), L=20 (slower, stronger)
    p.add_argument("--model_size", type=str, default="B",
                   choices=["S", "B", "L"],
                   help="S=small(num_res=8)  B=base(num_res=16)  L=large(num_res=20)")

    # Training hypers
    p.add_argument("--patch_size",  type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--num_epochs",  type=int,   default=200)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_min",      type=float, default=1e-6)
    p.add_argument("--save_every",  type=int,   default=5,
                   help="Save checkpoint every N epochs (also saves on best loss)")

    return p.parse_args()


# ── Loss ─────────────────────────────────────────────────────────────────────

LAMBDA_FREQ = 0.1   # paper eq.13

def dual_domain_loss(pred, target):
    """Spatial L1 + Frequency L1 (paper Section III.D)."""
    l_spatial = F.l1_loss(pred, target)

    pf = torch.fft.rfft2(pred,   norm="backward")
    tf = torch.fft.rfft2(target, norm="backward")
    pred_ri   = torch.cat([pf.real, pf.imag], dim=1)
    target_ri = torch.cat([tf.real, tf.imag], dim=1)
    l_freq = F.l1_loss(pred_ri, target_ri) / pred_ri.numel()

    return l_spatial + LAMBDA_FREQ * l_freq


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _drive_copy(src: str, drive_dir: str):
    """Copy a file to Drive. Silently skips if drive_dir is empty."""
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    dst = os.path.join(drive_dir, os.path.basename(src))
    shutil.copy2(src, dst)


def save_checkpoint(state: dict, epoch: int, loss: float,
                    local_dir: str, drive_dir: str, is_best: bool):
    os.makedirs(local_dir, exist_ok=True)

    # ── Save locally ─────────────────────────────
    epoch_name = f"epoch_{epoch:04d}.pth"
    local_path = os.path.join(local_dir, epoch_name)
    torch.save(state, local_path)

    print(f"  [ckpt] epoch {epoch:04d} saved locally (loss={loss:.5f})")

    # ── Save directly to Drive ───────────────────
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        drive_path = os.path.join(drive_dir, epoch_name)
        torch.save(state, drive_path)
        print(f"  [drive] saved → {drive_path}")

    # ── Best model ───────────────────────────────
    if is_best:
        best_local = os.path.join(local_dir, "best_model.pth")
        torch.save(state["model"], best_local)

        print(f"  [best] new best loss={loss:.5f} (local)")

        if drive_dir:
            best_drive = os.path.join(drive_dir, "best_model.pth")
            torch.save(state["model"], best_drive)
            print(f"  [drive] saved best → {best_drive}")


def find_latest_checkpoint(drive_dir: str, local_dir: str) -> str:
    """Return path of the most recent epoch_XXXX.pth, checking Drive then local."""
    for search_dir in [drive_dir, local_dir]:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        ckpts = sorted([
            f for f in os.listdir(search_dir)
            if f.startswith("epoch_") and f.endswith(".pth")
        ])
        if ckpts:
            return os.path.join(search_dir, ckpts[-1])
    return ""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # ── Model
    # ConvIR(num_res) — num_res = residual blocks per EBlock/DBlock stage
    # S=8 (fast local test), B=16 (paper default), L=20 (strongest)
    NUM_RES = {"S": 8, "B": 16, "L": 20}[args.model_size]
    model   = ConvIR(num_res=NUM_RES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model      : ConvIR-{args.model_size}  num_res={NUM_RES}  ({n_params/1e6:.2f}M params)")

    # ── Dataset
    print("\nLoading training dataset...")
    train_set = DualDegradationDataset(args.train_dir, patch_size=args.patch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Steps/epoch: {len(train_loader)}")

    # ── Optimizer + scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    cosine    = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr_min)

    if HAS_WARMUP:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=3, after_scheduler=cosine
        )
    else:
        scheduler = cosine

    # ── Resume
    start_epoch = 1
    best_loss   = float("inf")

    resume_path = args.resume or find_latest_checkpoint(args.drive_ckpt, args.local_ckpt)
    print("resume_path: " + resume_path)
    if resume_path:
        print(f"\nResuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt["model"])
            if "optimizer"   in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler"   in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss   = ckpt.get("best_loss", float("inf"))
        else:
            model.load_state_dict(ckpt)
        print(f"  Resuming epoch {start_epoch}  |  best_loss so far: {best_loss:.5f}")
    else:
        print("\nNo checkpoint found — starting fresh.")

    # ── Training loop
    print(f"\n{'='*60}")
    print(f" Training  ConvIR-{args.model_size}  "
          f"epochs {start_epoch}..{args.num_epochs}  "
          f"bs={args.batch_size}  patch={args.patch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, (deg, clean) in enumerate(train_loader, 1):
            deg, clean = deg.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(deg)

            # Model may return list (multi-output strategy, paper Section III.A)
            if isinstance(pred, (list, tuple)):
                loss = 0
                for p in pred:
                    target_resized = F.interpolate(
                            clean,
                            size=p.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                            )
                    loss += dual_domain_loss(p, target_resized)
                loss /= len(pred)
            else:
                loss = dual_domain_loss(pred, clean)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"  Epoch [{epoch:>4}/{args.num_epochs}] "
                      f"Step [{step:>4}/{len(train_loader)}] "
                      f"loss={loss.item():.5f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        is_best  = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        print(f"\nEpoch {epoch:>4} | avg_loss={avg_loss:.5f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"best={best_loss:.5f}\n")

        # Save on schedule or when best
        if epoch % args.save_every == 0 or is_best:
            state = {
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "loss":       avg_loss,
                "best_loss":  best_loss,
                "model_size": args.model_size,
                "num_res":    NUM_RES,
            }
            save_checkpoint(state, epoch, avg_loss,
                            args.local_ckpt, args.drive_ckpt, is_best)

    print(f"\nTraining complete.  Best loss: {best_loss:.5f}")
    print(f"Best model saved at: {os.path.join(args.local_ckpt, 'best_model.pth')}")


if __name__ == "__main__":
    main()
