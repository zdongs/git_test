from torch import nn,randn,cuda
device = "cuda" if cuda.is_available() else "cpu"


class RNN_Model(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,output_size):
        # 调用父类初始方法
        super().__init__()
        # 属性通过模型getparameters()获取内部参数
        self.rnn = nn.RNN(
            input_size=input_size,       # X 特性数量
            hidden_size=hidden_size,       # 隐层大小 Wih[28,20]
            batch_first=True,
            num_layers = num_layers
        )
        # 通过Linear线性层推理
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output,hidden = self.rnn(x)
        output = self.out(hidden[-1,:,:])
        return output


if __name__ == '__main__':
    rnn = RNN_Model(28,100,2,10).to(device)
    # 模拟输入X
    x = randn((3,28,28)).to(device)
    out = rnn(x)
    print(out.shape)