from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn, optim, save, no_grad
from rnn_models import RNN_Model
import sys,os

myfunction_dire = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(myfunction_dire)
from myfunction.cudas import devices
from tqdm import tqdm

train_sets = KMNIST(root="data", train=True, transform=ToTensor())
test_sets = KMNIST(root="data", train=False, transform=ToTensor())
class_target = train_sets.classes
print(class_target)

batch_size = 108
epochs = 40
device = devices()
train_dataloader = DataLoader(train_sets, batch_size, shuffle=True)
test_dataloader = DataLoader(test_sets, batch_size)


# 初始化模型
input_size = 28  # 输入维度（28*28像素）
seq_len = 28
hidden_size = 100
num_layer = 4  # 2层RNN
output_size = 10  # 输出维度 （0-9 10个数）
lr = 0.001  # 学习率

# 模型实例化
model = RNN_Model(input_size, hidden_size, num_layer, output_size).to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    tqbar = tqdm(train_dataloader)
    for i, (X, y) in enumerate(tqbar):
        X, y = X.to(device), y.to(device)
        model.train()
        X = X.reshape(-1, seq_len, input_size)
        # 前向传播
        outputs = model(X)
        # 计算损失
        loss = loss_fn(outputs, y)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

        tqbar.set_description(f"epoch:{epoch+1:>3},loss:{loss.item():.4f}")
    model.eval()
    nums = len(test_dataloader.dataset)
    num_batchs = len(test_dataloader)
    correct, test_loss = 0, 0
    # 不使用梯度更新
    with no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, 28, 28)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
        # 平均损失和准确率
        test_loss /= num_batchs
        correct /= nums
    print(f"avg loss:{test_loss:.4f} correct:{(correct *100):.2f}%")


save(model, "git_test/models/rnn_kmnist.pth")
