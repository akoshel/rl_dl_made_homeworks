import matplotlib.pyplot as plt
import torch


def display_progress(cond: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, figsize=(10, 12)) -> None:
    for i in range(len(cond)):
        cond_show = cond[i].detach().cpu().permute(1, 2, 0)
        fake_show = fake[i].detach().cpu().permute(1, 2, 0)
        real_show = real[i].detach().cpu().permute(1, 2, 0)
        fig, ax = plt.subplots(i, 3, figsize=(10, 15))
        ax[i][0].imshow(cond_show)
        ax[i][2].imshow(fake_show)
        ax[i][1].imshow(real_show)
    plt.show()
    