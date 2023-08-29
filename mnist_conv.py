from torch import nn, cuda, optim, save
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from kmnist_torch import train, test
from kmnist_conv import CNN


if __name__ == "__main__":
    device = "cuda" if cuda.is_available() else "cpu"

    training_data = MNIST(
        root="D:\ProgramData\data", train=True, download=True, transform=ToTensor()
    )

    test_data = MNIST(
        root="D:\ProgramData\data", train=False, download=True, transform=ToTensor()
    )

    batch_size = 128

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True)

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
    save(mymodel, "models/mnist_conv.pth")