from torchvision.models import vgg11_bn
from torch.utils.data import DataLoader
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch import nn,optim
# from tqdm import tqdm
import sys
sys.path.append('D:/demodesk/git_test')

from myfunction.cudas import devices
from kmnist.kmnist_torch import train

'''
1. 加载预训练模型
2. 修改模型架构（换头）
3. 冻结卷积层
4. 重新训练模型
'''

batch_size = 16
epochs = 10
device = devices()
best_acc = 0

# 定制经典模型结构（加载预训练参数）
mymodel = vgg11_bn(weights = 'DEFAULT')

print(mymodel)

# 定制新头
classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Linear(4096,2048),
    nn.ReLU(),
    nn.Linear(2048,2048),
    nn.ReLU(),
    nn.Linear(2048,101)
)

# 冻结身体
mymodel.features.training = False
# 换头
mymodel.classifier = classifier

# 组合转换方法
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((15,50)),
        transforms.Compose([
            transforms.RandomCrop((200,200)),
            transforms.Resize((300,300))
        ])
    ])
])

# 数据集加载与封装
dataset = Caltech101(root='data',transform=transform,download=True)
data_loader = DataLoader(dataset,batch_size,True)

mymodel.to(device)
loss_fn = nn.CrossEntropyLoss()
optims = optim.Adam(mymodel.parameters())

for e in range(epochs):
    print(f'epoch:{e+1:>2d}')
    train(data_loader,mymodel,loss_fn,optims,device)