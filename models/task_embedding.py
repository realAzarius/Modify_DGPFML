import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Autoencoder(nn.Module):

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        # Encoder specification
        self.enc_linear_1 = nn.Linear(input_size, 128)
        self.enc_linear_2 = nn.Linear(128, self.embedding_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.embedding_size, 128)
        self.dec_linear_2 = nn.Linear(128, input_size)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, code):
        #code = code.view(-1, 784)  # 或者 code = code.reshape(-1, 784)
        code = code.to(self.enc_linear_1.weight.dtype)
        code = F.relu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    def decode(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        return out
'''


class Autoencoder(nn.Module):

    def __init__(self, input_size, embedding_size):  # 784 64
        super(Autoencoder, self).__init__()
        self.embedding_size = embedding_size

        # Encoder specification
        self.enc_linear_1 = nn.Linear(input_size, 128)
        self.enc_bn_1 = nn.BatchNorm1d(128)  # 添加批归一化层
        self.enc_linear_2 = nn.Linear(128, self.embedding_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.embedding_size, 128)
        self.dec_bn_1 = nn.BatchNorm1d(128)  # 添加批归一化层
        self.dec_linear_2 = nn.Linear(128, input_size)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    # def encode(self, code):
    #     code = code.view(-1, 784)  # 或者 code = code.reshape(-1, 784)
    #     code = code.to(self.enc_linear_1.weight.dtype)
    #     code = F.relu(self.enc_linear_1(code))
    #     code = self.enc_linear_2(code)
    #     return code

    def encode(self, code):
        code = F.relu(self.enc_linear_1(code))
        code = self.enc_bn_1(code)  # 应用批归一化层
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = self.dec_bn_1(out)  # 应用批归一化层
        out = F.sigmoid(self.dec_linear_2(out))
        return out
