import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
global Linear
class _Linear(nn.Module):
    """The class for meta linear"""
    def __init__(self, input_dim, output_dim, bias=True):
        super(_Linear, self).__init__()

        self.weight = nn.Parameter(torch.empty((output_dim, input_dim)))
        self.weight.requires_grad=False
        self.mtl_weight = nn.Parameter(torch.empty(1))

        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
            self.bias.requires_grad=False
            self.mtl_bias = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.mtl_weight.data = torch.tensor(1.0)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            self.mtl_bias.data = torch.tensor(0.0)
class Linear(_Linear):
    """The class for meta linear"""
    def __init__(self, input_dim, output_dim, bias = True):
        super(Linear, self).__init__(input_dim, output_dim, bias)
        self.has_bias = bias

    def forward(self, inp):
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        new_weight = self.weight.mul(new_mtl_weight)
        if self.has_bias:
            new_bias = self.bias.add(self.mtl_bias)
        else:
            new_bias = None
        return F.linear(inp, new_weight, new_bias)
    
    def gen_new_weight(self):
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        self.weight.data = self.weight.mul(new_mtl_weight).data.clone()
        if self.has_bias:
            self.bias.data = self.bias.add(self.mtl_bias).data.clone()

    
    def freeze_meta_parameters(self, mode = True):
        self.weight.requires_grad_(mode)
        self.mtl_weight.requires_grad_(not mode)

        if self.has_bias:
            self.bias.requires_grad_(mode)
            self.mtl_bias.requires_grad_(not mode)
    
    def freeze_all_parameters(self, mode = True):
        self.weight.requires_grad_(not mode)
        self.mtl_weight.requires_grad_(not mode)

        if self.has_bias:
            self.bias.requires_grad_(not mode)
            self.mtl_bias.requires_grad_(not mode)
    
    def reset_meta_parameters(self):
        self.mtl_weight.data = torch.tensor(1.0)
        self.mtl_bias.data = torch.tensor(0.0)
    
    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

class Mclr_Logistic(nn.Module):

    def __init__(self, input_dim=60, output_dim=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class DNN(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super(DNN, self).__init__()
        # define network layers
        self.features = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_dim, 128)),
                ('relu1', nn.ReLU()),

                ('fc2', nn.Linear(128, 10)),
                ('relu2', nn.ReLU()),

                ('fc3', nn.Linear(10, 128)),
                ('relu3', nn.ReLU()),
            ]),
        )
        self.classifier = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


class DNN_BN(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super(DNN_BN, self).__init__()
        # define network layers
        self.features = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_dim, mid_dim)),
                ('bn1', nn.BatchNorm1d(mid_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(mid_dim, 50)),
                ('bn2', nn.BatchNorm1d(50)),
                ('relu2', nn.ReLU()),
            ]),
        )
        self.classifier = nn.Linear(50, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output

class DNN_Init(nn.Module):

    def __init__(self, input_dim= 784, mid_dim = 100, output_dim = 10):
        super(DNN_Init, self).__init__()

        self.features = nn.Sequential(
            Linear(input_dim, 512),
            nn.ReLU(),

            Linear(512, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x):#moxingjisuanguocheng qianxiangchuanbo
        x = torch.flatten(x, 1)#jiang x biaoshichengyiweishuzu
        out = self.features(x)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)
    
    def gen_new_feature_weights(self):
        for params in self.features:
            if hasattr(params, 'gen_new_weight'):
                params.gen_new_weight()
    
    def freeze_meta_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear):
                params.freeze_meta_parameters(mode)
    
    def reset_meta_parameters(self):
        for params in self.features:
            if isinstance(params, Linear):
                params.reset_meta_parameters()


    def freeze_all_parameters(self, mode):
        for params in self.features:
            print(type(params))
            if isinstance(params, Linear):
                params.freeze_all_parameters(mode)

class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn = \
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        return output

def get_number_of_trained_params(net):
    res = 0
    for params in net.parameters():
        if params.requires_grad:
            res += params.numel()
    return res


if __name__ == "__main__":
    from torchstat import stat
    from torchsummary import summary
    # stat(net, (1, 28, 28))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fedavg = DNN()
    peravg = DNN()
    fedbn = DNN_BN()
    pfedinit = DNN_Init()
    
    summary(fedavg.to(device), (1, 28, 28))








    # print(get_number_of_trained_params(pfedinit))