from torch import nn, cuda, no_grad, optim, save
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor,Compose,ColorJitter
from myfunction.alexnet import AlexNet
from tqdm import tqdm

batch = 128
epochs = 10
device = 'cuda' if cuda.is_available() else 'cpu'

transform = Compose([ToTensor(),ColorJitter((1,10),(1,10),(1,10),(0.2,0.4))])

# 数据准备
dataset_train = CIFAR10(root='data',transform=transform,download = True)
dataset_test = CIFAR10(root='data',transform=transform,download=True,train=False)

# 数据封装
train_loader = DataLoader(dataset_train,batch_size=batch,shuffle=True)
test_loader = DataLoader(dataset_test,batch)

# 模型构建
mymodel = AlexNet().to(device)

loss_fn = nn.CrossEntropyLoss()
opti = optim.Adam(mymodel.parameters())

best_accuracy = 0

# 运算
for e in range(epochs):
    tpbar = tqdm(train_loader)
    mymodel.train()
    for batch,(X,y) in enumerate(tpbar):
        X,y = X.to(device),y.to(device)
        pred = mymodel(X)
        loss = loss_fn(pred,y)

        loss.backward()
        opti.step()
        mymodel.zero_grad()
        
        tpbar.set_description(
        f'epoch:{e+1} batch:{batch+1:>5d} loss:{loss.item():>7f}'
        )
    mymodel.eval()
    test_loss,correct = 0,0
    size = len(test_loader.dataset)
    num_batch = len(test_loader)
    for X,y in test_loader:
        with no_grad():
            X,y = X.to(device),y.to(device)
            pred = mymodel(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= num_batch
    correct /= size
    print(f'loss:{test_loss} Accuracy:{correct*100:>4f}%')

    if correct > best_accuracy:
        save(mymodel,'git_test/models/alexnet_CIFAR10.pth')
        
print('训练完成！')