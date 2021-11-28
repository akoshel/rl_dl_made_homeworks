import matplotlib.pyplot as plt
import torch


def display_progress(cond: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, figsize=(10, 12)) -> None:
    fig, ax = plt.subplots(len(cond), 3, figsize=(10, 3 * len(cond)))
    for i in range(len(cond)):
        ax[i][0].imshow(cond[i].permute(1, 2, 0))
        ax[i][2].imshow(fake[i].permute(1, 2, 0))
        ax[i][1].imshow(real[i].permute(1, 2, 0))
    plt.show()
