# import torch
from torch import nn, cuda, optim, save
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from kmnist_torch import train, test


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ReLU_stack = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(3920, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out = self.conv_ReLU_stack(x)
        return out


if __name__ == "__main__":
    device = "cuda" if cuda.is_available() else "cpu"

    training_data = datasets.KMNIST(
        root="data", train=True, download=False, transform=ToTensor()
    )

    test_data = datasets.KMNIST(
        root="data", train=False, download=False, transform=ToTensor()
    )

    batch_size = 128

    train_dataloader = DataLoader(
        training_data,  # 数据集对象，如 torch.utils.data.Dataset 的实例
        batch_size=batch_size,  # 批次大小，即每次加载的样本数
        shuffle=True,  # 是否在每个 epoch 时对数据进行洗牌
        pin_memory=True,  # 是否将加载的数据放入 CUDA 固定内存中（适用于 GPU 训练）
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    mymodel = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mymodel.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, mymodel, loss_fn, optimizer, device)

    test_loss, correct = test(test_dataloader, mymodel, loss_fn, device)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"  # noqa: E501
    )
    save(mymodel, "git_test/models/kmnist_conv.pth")
