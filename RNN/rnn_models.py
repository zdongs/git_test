from torch import randn,cuda
from torch.nn import RNN,LSTM,GRU,Linear,Module
device = "cuda" if cuda.is_available() else "cpu"


class RNN_Model(Module):

    def __init__(self,input_size,hidden_size,num_layers,output_size,rnn_type):
        # 调用父类初始方法
        super().__init__()
        # 属性通过模型getparameters()获取内部参数
        if rnn_type == 'RNN':
            self.rnn = RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers
            )
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers
            )
        elif rnn_type == 'GRU':
            self.rnn = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                num_layers=num_layers
            )
        else:
            # 引起一个错误
            raise ValueError("Unsupported RNN type")
        # 通过Linear线性层推理
        self.out = Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output,hidden = self.rnn(x)
        output = self.out(hidden[-1,:,:])
        return output


if __name__ == '__main__':
    rnn = RNN_Model(28,100,2,10,'RNN').to(device)
    # 模拟输入X
    x = randn((3,28,28)).to(device)
    out = rnn(x)
    print(out.shape)