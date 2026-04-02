import torch
import torch.nn.functional as F


def compute_adaptive_lambda(pred, target):
    """
    Adaptive lambda based on reconstruction error.
    If high-frequency error is large → increase lambda
    """

    # spatial error
    err = torch.abs(pred - target)

    # focus on high-frequency error via Laplacian
    kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=pred.dtype,
            device=pred.device
            ).view(1, 1, 3, 3)

    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)

    hf_err = F.conv2d(err, kernel, padding=1, groups=pred.shape[1])
    hf_err = torch.abs(hf_err)

    # global magnitude of HF error
    hf_score = hf_err.mean(dim=[1,2,3], keepdim=True)

    # normalize across batch
    hf_min = hf_score.min().detach()
    hf_max = hf_score.max().detach()

    hf_norm = (hf_score - hf_min) / (hf_max - hf_min + 1e-6)

    # map to λ range
    lam = 0.05 + 0.25 * hf_norm   # adaptive

    return lam


def dual_domain_loss(pred, target, adaptive=True):
    l_spatial = F.l1_loss(pred, target)

    pf = torch.fft.rfft2(pred, norm="backward")
    tf = torch.fft.rfft2(target, norm="backward")

    pred_ri = torch.cat([pf.real, pf.imag], dim=1)
    target_ri = torch.cat([tf.real, tf.imag], dim=1)

    l_freq = F.l1_loss(pred_ri, target_ri) / pred_ri.numel()

    if adaptive:
        lam = compute_adaptive_lambda(pred.detach(), target.detach())
        lam = lam.mean().clamp(0.05, 0.30)
    else:
        lam = 0.1

    return l_spatial + lam * l_freq, lam.item()
