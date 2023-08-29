# import torch
# from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


training_data = datasets.KMNIST(
    root="D:\ProgramData\data", train=True, download=False, transform=ToTensor()
)

test_data = datasets.KMNIST(
    root="D:\ProgramData\data", train=False, download=False, transform=ToTensor()
)

for img,lbl in training_data:
    plt.imshow(img)
    plt.title(str(lbl))
    break

plt.show()