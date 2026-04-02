import torch
import matplotlib.pyplot as plt


def fft_vis(img):
    fft = torch.fft.fft2(img)
    return torch.log(1 + torch.abs(fft))


def show_fft(inp, out, gt):
    inp_f = fft_vis(inp)
    out_f = fft_vis(out)
    gt_f  = fft_vis(gt)

    fig, ax = plt.subplots(3,2, figsize=(8,10))

    ax[0,0].imshow(inp.permute(1,2,0).cpu())
    ax[0,1].imshow(inp_f.mean(0).cpu(), cmap='gray')

    ax[1,0].imshow(out.permute(1,2,0).cpu())
    ax[1,1].imshow(out_f.mean(0).cpu(), cmap='gray')

    ax[2,0].imshow(gt.permute(1,2,0).cpu())
    ax[2,1].imshow(gt_f.mean(0).cpu(), cmap='gray')

    for a in ax.flatten():
        a.axis('off')

    plt.savefig("fft.png")
    plt.show()
