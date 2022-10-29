import torch
import numpy as np
import matplotlib.pyplot as plt


def keep_numbers(numbers: list[int, ...], dataset) -> torch.Tensor:
    a = np.array([(dataset.targets == i).numpy() for i in numbers])
    keeps = [False for i in range(len(a[0]))]
    for data_point in range(len(a[0])):
        for number in range(len(numbers)):
            if a[:, data_point][number]:
                keeps[data_point] = True
    return torch.tensor(keeps, dtype=None)


def save_image_grid(images: torch.Tensor, epoch: int):
    _rows = 6
    _cols = 6
    _fig, _ax = plt.subplots(_rows, _cols, figsize= (8,8))
    counter = 0
    for _row in range(_rows):
        for _col in range(_cols):
            _X_hat = images[counter].reshape((28,28)).detach().numpy()
            _ax[_row][_col].set_xticks([])
            _ax[_row][_col].set_yticks([])
            plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0,hspace=0)
            _ax[_row][_col].imshow(_X_hat, cmap="gray")
            counter += 1
    plt.savefig("Training_pictures/epoch_"+str(epoch)+".pdf")
    plt.close()