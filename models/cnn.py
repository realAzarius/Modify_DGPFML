import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models1 import Linear
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from collections import OrderedDict


class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        self.weight.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad = False
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.has_bias = bias
        super(Conv2dMtl, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inp):
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        new_weight = self.weight.mul(new_mtl_weight)
        if self.bias is not None:
            new_bias = self.bias + self.mtl_bias
        else:
            new_bias = None
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def freeze_meta_parameters(self, mode=True):
        self.weight.requires_grad_(mode)
        self.mtl_weight.requires_grad_(not mode)

        if self.has_bias:
            self.bias.requires_grad_(mode)
            self.mtl_bias.requires_grad_(not mode)

    def freeze_all_parameters(self, mode=True):
        self.weight.requires_grad_(not mode)
        self.mtl_weight.requires_grad_(not mode)

        if self.has_bias:
            self.bias.requires_grad_(not mode)
            self.mtl_bias.requires_grad_(not mode)

    def reset_meta_parameters(self):
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.mtl_bias.data.uniform_(0, 0)

    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class LeNet_Init(nn.Module):
    def __init__(self, in_channels=1, input_shape=(32, 32)):
        super(LeNet_Init, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels

        self.features = nn.Sequential(
            Conv2dMtl(in_channels, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            Conv2dMtl(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            Flatten(),
            # cifar
        )

        self.classifier = nn.Sequential(
            # mnist
            # nn.Linear(16 * 4 * 4, 120),
            # cifar
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 100),
        )

    def forward(self, img):
        # print(img.shape) # [5,3072]
        # feature
        img = img.view(img.shape[0], self.in_channels, self.input_shape[0], self.input_shape[1])
        output = self.features(img)
        output = self.classifier(output)
        return output

    def gen_new_feature_weights(self):
        for params in self.features:
            if hasattr(params, 'gen_new_weight'):
                params.gen_new_weight()

    def freeze_meta_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_meta_parameters(mode)

    def reset_meta_parameters(self):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.reset_meta_parameters()

    def freeze_all_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_all_parameters(mode)


class LeNet_BN(nn.Module):
    def __init__(self, in_channels=1):
        super(LeNet_BN, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels, 6, 5)),
                ('bn1', nn.BatchNorm2d(6)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2, 2)),

                ('conv2', nn.Conv2d(6, 16, 5)),
                ('bn2', nn.BatchNorm2d(16)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2, 2)),
            ]),
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(16 * 5 * 5, 120)),
                ('bn3', nn.BatchNorm1d(120)),
                ('relu3', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(120, 84)),
                ('bn4', nn.BatchNorm1d(84)),
                ('relu4', nn.ReLU(inplace=True)),
                ('classifier', nn.Linear(84, 10))
            ]),

        )

    def forward(self, img):
        img = img.view(img.shape[0], 3, 32, 32)
        output = self.features(img)
        output = torch.flatten(output, 1)
        return output


class LeNet(nn.Module):
    def __init__(self, in_channels=1):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            # nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        # img= img.view(img.shape[0], 3, 32, 32)
        img = img.view(img.shape[0], 1, 28, 28)
        output = self.features(img)
        output = self.classifier(output)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockMtl(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockMtl, self).__init__()
        self.conv1 = conv3x3mtl(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckMtl(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckMtl, self).__init__()
        self.conv1 = Conv2dMtl(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMtl(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2dMtl(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetMtl(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], mtl=True):
        super(ResNetMtl, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
            block = BasicBlockMtl
        else:
            self.Conv2d = nn.Conv2d
            block = BasicBlock
        # cfg = [160, 320, 640]
        cfg = [64, 128, 256, 512]
        self.inplanes = 64
        self.conv1 = self.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[0], layers[0])
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, cfg[3], layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, 65)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def gen_new_feature_weights(self):
        for params in self.parameters():
            if hasattr(params, 'gen_new_weight'):
                params.gen_new_weight()

    def freeze_meta_parameters(self, mode):
        for m in self.modules():
            if isinstance(m, (Linear, Conv2dMtl)):
                m.freeze_meta_parameters(mode)
        for k, v in self.named_parameters():
            if 'bn' in k:
                v.requires_grad_(mode)

    def reset_meta_parameters(self):
        for m in self.modules():
            if isinstance(m, (Linear, Conv2dMtl)):
                m.reset_meta_parameters()

    def freeze_all_parameters(self, mode):
        for m in self.modules():
            if isinstance(m, (Linear, Conv2dMtl)):
                m.freeze_all_parameters(mode)
        for k, v in self.named_parameters():
            if 'bn' in k:
                v.requires_grad_(mode)


class Conv4_Init(nn.Module):
    def __init__(self, in_channels=3, input_shape=(32, 32)):
        super(Conv4_Init, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels

        self.features = nn.Sequential(
            Conv2dMtl(in_channels, 32, 3, 1, 0),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            Conv2dMtl(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            Conv2dMtl(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            Conv2dMtl(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 1),  # kernel_size, stride
            Flatten(),
        )

        self.classifier = nn.Linear(32 * 23 * 23, 65)

    def forward(self, img):
        # feature
        img = img.view(img.shape[0], self.in_channels, self.input_shape[0], self.input_shape[1])
        output = self.features(img)
        output = self.classifier(output)
        return output

    def gen_new_feature_weights(self):
        for params in self.features:
            if hasattr(params, 'gen_new_weight'):
                params.gen_new_weight()

    def freeze_meta_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_meta_parameters(mode)

    def reset_meta_parameters(self):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.reset_meta_parameters()

    def freeze_all_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_all_parameters(mode)


class Conv4(nn.Module):
    def __init__(self, in_channels=3, input_shape=(32, 32)):
        super(Conv4, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 0),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(32, 32, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 1),  # kernel_size, stride
            Flatten(),
        )

        self.classifier = nn.Linear(32 * 27 * 27, 10)

    def forward(self, img):
        # feature
        img = img.view(img.shape[0], self.in_channels, self.input_shape[0], self.input_shape[1])
        output = self.features(img)
        output = self.classifier(output)
        return output


class Conv4_BN(nn.Module):
    def __init__(self, in_channels=3, input_shape=(32, 32)):
        super(Conv4_BN, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels

        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels, 32, 3, 1, 0)),
                ('relu1', nn.ReLU(inplace=True)),
                ('bn1', nn.BatchNorm2d(32)),
                ('pool1', nn.MaxPool2d(2, 2)),

                ('conv2', nn.Conv2d(32, 32, 3, 1, 0)),
                ('relu2', nn.ReLU(inplace=True)),
                ('bn2', nn.BatchNorm2d(32)),
                ('pool2', nn.MaxPool2d(2, 2)),

                ('conv3', nn.Conv2d(32, 32, 3, 1, 0)),
                ('relu3', nn.ReLU(inplace=True)),
                ('bn3', nn.BatchNorm2d(32)),
                ('pool3', nn.MaxPool2d(2, 2)),

                ('conv4', nn.Conv2d(32, 32, 3, 1, 0)),
                ('relu4', nn.ReLU(inplace=True)),
                ('bn4', nn.BatchNorm2d(32)),
                ('pool4', nn.MaxPool2d(2, 1)),
            ])
        )

        self.classifier = nn.Linear(32 * 23 * 23, 65)

    def forward(self, img):
        # feature
        img = img.view(img.shape[0], self.in_channels, self.input_shape[0], self.input_shape[1])
        output = self.features(img)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output


class AlexNet_Init(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, num_classes=1000, input_shape=(32, 32)):
        super(AlexNet_Init, self).__init__()
        self.input_shape = input_shape
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', Conv2dMtl(3, 64, kernel_size=11, stride=4, padding=2)),
                # ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', Conv2dMtl(64, 192, kernel_size=5, padding=2)),
                # ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', Conv2dMtl(192, 384, kernel_size=3, padding=1)),
                # ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', Conv2dMtl(384, 256, kernel_size=3, padding=1)),
                # ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', Conv2dMtl(256, 256, kernel_size=3, padding=1)),
                # ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('avgpool', nn.AdaptiveAvgPool2d((6, 6))),
                ('flatten', Flatten()),
                ('fc1', Linear(256 * 6 * 6, 4096)),
                # ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(4096, 4096)),
                # ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = x.view(x.shape[0], 3, self.input_shape[0], self.input_shape[1])
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def gen_new_feature_weights(self):
        for params in self.features:
            if hasattr(params, 'gen_new_weight'):
                params.gen_new_weight()

    def freeze_meta_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_meta_parameters(mode)

    def reset_meta_parameters(self):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.reset_meta_parameters()

    def freeze_all_parameters(self, mode):
        for params in self.features:
            if isinstance(params, Linear) or isinstance(params, Conv2dMtl):
                params.freeze_all_parameters(mode)


if __name__ == "__main__":
    # net = Conv4_Init(input_shape=(224, 224))
    import torchvision.models as models
    from torchstat import stat

    # net = models.alexnet(pretrained=True)
    net = LeNet()
    print(stat(net, (1, 28, 28)))
    # exit(0)
    # net_iter = net.named_parameters()

    # net1 = AlexNet_Init()
    # print(net1)
    # net1_iter = iter(net1.items())
    # x = next(net1_iter)

    # while True:
    #     try:
    #         w1 = next(net1_iter)
    #         if 'mtl' in w1[0]:
    #             continue
    #         print('target:', w1[0])
    #         w2 = next(net_iter)
    #         print('source:', w2[0])
    #         w1[1].data = w2[1].data.clone()
    #     except StopIteration:
    #         break
    # print('ok')
    # input = torch.randn(2, 3, 224, 224)
    # y = torch.LongTensor([0, 1])

    # loss = nn.CrossEntropyLoss()
    # l = loss(net(input), y)
    # l.backward()
    # print(l)
    # # for (k, v), (dk, dv) in zip(net.named_parameters(), resnet18.named_parameters()):
    # #     if 'mtl' not in k:
    # #         print(k, dk)
    # for k, v in resnet18.named_parameters():
    #     print(k)
