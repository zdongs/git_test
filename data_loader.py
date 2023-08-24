# import torch
# from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# import matplotlib.pyplot as plt


training_data = datasets.KMNIST(
    root="D:\ProgramData\data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.KMNIST(
    root="D:\ProgramData\data", train=False, download=True, transform=ToTensor()
)
