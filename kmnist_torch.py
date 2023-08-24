import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# import matplotlib.pyplot as plt

# 下载的数据集文件默认保存到当前用户工作目录的data子目录中。
# 下载路径：root
# 例如"D:\\datasets\\fashionMNIST\\"一类的绝对路径
training_data = datasets.KMNIST(
    root="D:\ProgramData\data",
    train=True,
    download=False,
    transform=ToTensor()
)


test_data = datasets.KMNIST(
    root="D:\ProgramData\data",
    train=False,
    download=False,
    transform=ToTensor()
)

batch_size = 128

train_dataloader = DataLoader(
    training_data,  # 数据集对象，如 torch.utils.data.Dataset 的实例
    batch_size=batch_size,  # 批次大小，即每次加载的样本数
    shuffle=True,  # 是否在每个 epoch 时对数据进行洗牌
    pin_memory=True,  # 是否将加载的数据放入 CUDA 固定内存中（适用于 GPU 训练）
)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 检验可以使用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"尊敬的主人mgzn，这是您使用的设备：{device}")

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 实例化终极降维操作层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 隐藏层的线性单元
            nn.ReLU(),  # 隐藏层的非线性单元
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),  # 输出层
        )

    def forward(self, x):
        x = self.flatten(x)  # 应用终极降维
        out = self.linear_relu_stack(x)
        return out


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练数据样本总量
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 张量加载到设备

        # 计算预测的误差
        pred = model(X)  # 调用模型获得结果(forward时被自动调用)
        loss = loss_fn(pred, y)  # 计算损失

        # 反向传播 Backpropagation
        model.zero_grad()  # 重置模型中参数的梯度值为0
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型中参数的梯度值

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 模型设置为评估模式，代码等效于 model.train(False)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


if __name__ == "__main__":
    mymodel = NeuralNetwork(28 * 28, 512, 10).to(device)  # 转到gpu

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mymodel.parameters())

    t = 0
    while True:
        t += 1
        print(f"Epoch {t}\n-------------------------------")
        train(train_dataloader, mymodel, loss_fn, optimizer)
        test_loss, correct = test(test_dataloader, mymodel, loss_fn)
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"  # noqa: E501
        )
        if correct >= 0.8:
            break

    print("训练完成!")

    # 保存完整模型（包括权重和结构）
    torch.save(mymodel, "complete_model.pth")
