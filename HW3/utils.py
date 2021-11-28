import matplotlib.pyplot as plt
import torch


def display_progress(cond: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, figsize=(10, 12)) -> None:
    for i in range(len(cond)):
        cond = cond.detach().cpu().permute(1, 2, 0)
        fake = fake.detach().cpu().permute(1, 2, 0)
        real = real.detach().cpu().permute(1, 2, 0)
        fig, ax = plt.subplots(i, 3, figsize=(10, 15))
        ax[i][0].imshow(cond)
        ax[i][2].imshow(fake)
        ax[i][1].imshow(real)
    plt.show()
