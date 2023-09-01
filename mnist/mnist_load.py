from numpy import newaxis
from torch import load
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from kmnist_conv import CNN  # noqa: F401

mymodel = load('models\mnist_conv.pth').to('cpu')

one_test = MNIST(root='D:\ProgramData\data',train=False,download=False,transform=ToTensor())  # noqa: E501

img,label = one_test[56]

preds = mymodel(img[newaxis,:])
pred = preds.argmax(1).item()

print(pred == label)
