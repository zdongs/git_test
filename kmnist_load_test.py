import torch
from torch import nn
from kmnist_torch import NeuralNetwork
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

test_data = datasets.KMNIST(
    root="D:\ProgramData\data", train=False, download=False, transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=128)

loss_fn = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork(28 * 28, 512, 10).to(device)
model.load_state_dict(torch.load("kmnist_model.pth"))

size = len(test_dataloader.dataset)
num_batches = len(test_dataloader)
model.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= num_batches
correct /= size

print(
    f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"  # noqa: E501
)
