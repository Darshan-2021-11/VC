import torch
import torch.nn.functional as F


def compute_spatial_lambda(inp):
    mean = F.avg_pool2d(inp, kernel_size=7, stride=1, padding=3)
    var = F.avg_pool2d((inp - mean) ** 2, kernel_size=7, stride=1, padding=3)

    var = var.mean(dim=1, keepdim=True)
    var = var / (var.max().detach() + 1e-6)

    lam_map = 0.05 + 0.15 * var
    return lam_map


def dual_domain_loss(pred, target, adaptive=True):
    l_spatial = F.l1_loss(pred, target)

    pf = torch.fft.rfft2(pred, norm="backward")
    tf = torch.fft.rfft2(target, norm="backward")

    pred_ri = torch.cat([pf.real, pf.imag], dim=1)
    target_ri = torch.cat([tf.real, tf.imag], dim=1)

    l_freq = F.l1_loss(pred_ri, target_ri) / pred_ri.numel()

    if adaptive:
        lam_map = compute_spatial_lambda(pred.detach())
        lam = lam_map.mean().clamp(0.05, 0.15)
    else:
        lam = 0.1

    return l_spatial + lam * l_freq, lam.item()
