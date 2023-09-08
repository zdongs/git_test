import torch
import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from rnn_models import RNNBasic_Model, RNN_Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'使用{device}训练模型')

# mnist数据集
train_ds = ds.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_ds = ds.MNIST(root='data', train=False, download=True, transform=ToTensor())
# dataloader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=100, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=100, shuffle=True)
# model
rnn = RNN_Model()
# 模型注册设备
rnn.to(device)
# loss
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer
optm = torch.optim.Adam(rnn.parameters(), lr=0.001)


def train():
    for i, (X, y) in enumerate(train_dl):
        X,y = X.to(device), y.to(device)
        X = X.reshape(-1, 28, 28)
        pred = rnn(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optm.step()
        rnn.zero_grad()

        if i % 100 == 99:
            loss_val = loss.item()
            print(f'epoch:{epoch} loss:{loss_val:.4f}')

def test():
    # 样本数量
    nums = len(test_dl.dataset)
    # 批次数量
    num_batchs = len(test_dl)
    correct,test_loss = 0,0
    # 不使用梯度更新
    with torch.no_grad():
        for X,y in test_dl:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, 28, 28)
            pred = rnn(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # 平均损失和准确率
        test_loss /= num_batchs
        correct /= nums
    print(f'avg loss:{test_loss:.4f} correct:{(correct *100):.2f}%')

for epoch in range(2):
    test()
    train()

# torch.save()
torch.save(rnn,'rnn_mode.pth')

# torch.load()
rnn = torch.load('rnn_mode.pth')
test()

