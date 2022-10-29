import torch
import numpy as np


def keep_numbers(numbers: list[int, ...], dataset) -> torch.Tensor:
    a = np.array([(dataset.targets == i).numpy() for i in numbers])
    keeps = [False for i in range(len(a[0]))]
    for data_point in range(len(a[0])):
        for number in range(len(numbers)):
            if a[:, data_point][number]:
                keeps[data_point] = True
    return torch.tensor(keeps, dtype=None)
