# import torch
# from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt


training_data = datasets.KMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.KMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)