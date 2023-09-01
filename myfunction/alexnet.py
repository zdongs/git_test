from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.AlexNet_stack=nn.Sequential(
            nn.Conv2d(3,96,11,4,0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,84,3,1,1),
            nn.ReLU(),
            nn.Conv2d(84,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10))
    
    def forward(self,x):
        return self.AlexNet_stack(x)