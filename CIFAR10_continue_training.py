from torch import nn, cuda, no_grad, optim, save, load
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, ColorJitter
from alexnet import AlexNet
from tqdm import tqdm

batch = 128
device = "cuda" if cuda.is_available() else "cpu"

transform = Compose([ToTensor(), ColorJitter((1, 10), (1, 10), (1, 10), (0.2, 0.4))])

# 数据准备
dataset_train = CIFAR10(root="models", transform=transform, download=True)
dataset_test = CIFAR10(root="models", transform=transform, download=True, train=False)

# 数据封装
train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
test_loader = DataLoader(dataset_test, batch)

# 模型构建
mymodel = load("git_test/models/best_CIFAR10.pth").to(device)

loss_fn = nn.CrossEntropyLoss()
opti = optim.Adam(mymodel.parameters())

mymodel.eval()
test_loss, correct = 0, 0
size = len(test_loader.dataset)
for X, y in test_loader:
    with no_grad():
        X, y = X.to(device), y.to(device)
        pred = mymodel(X)
        correct += (pred.argmax(1) == y).sum().item()
correct /= size
best_accuracy = correct

e = 0
# 运算
while True:
    e += 1
    tpbar = tqdm(train_loader)
    mymodel.train()
    for batch, (X, y) in enumerate(tpbar):
        X, y = X.to(device), y.to(device)
        pred = mymodel(X)
        loss = loss_fn(pred, y)

        loss.backward()
        opti.step()
        mymodel.zero_grad()

        tpbar.set_description(f"epoch:{e} batch:{batch+1:>5d} loss:{loss.item():>7f}")
    mymodel.eval()
    test_loss, correct = 0, 0
    size = len(test_loader.dataset)
    num_batch = len(test_loader)
    for X, y in test_loader:
        with no_grad():
            X, y = X.to(device), y.to(device)
            pred = mymodel(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= num_batch
    correct /= size
    print(f"loss:{test_loss} Accuracy:{correct*100:>4f}%")

    if correct > best_accuracy:
        save(mymodel, "git_test/models/best_CIFAR10.pth")

    if correct > 0.9:
        break

# 保存说明文件
with open("git_test/models/best_training_summary.txt", "w") as f:
    f.write("训练总结\n")
    f.write("模型信息:\n")
    f.write(str(mymodel))  # 将模型信息写入文件
    f.write("\n")
    f.write(f"总训练轮数: {e}\n")
    f.write(f"最终测试准确率: {correct*100:.2f}%\n")
    f.write(f"最终测试损失值: {test_loss:.4f}\n")
    f.write("训练完成！")

print("训练总结已保存到 training_summary.txt 文件中")

