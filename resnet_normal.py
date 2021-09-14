import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)




# class MetaLinear(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         ignore = nn.Linear(*args, **kwargs)
#
#         self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
#         self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
#
#     def forward(self, x):
#         return F.linear(x, self.weight, self.bias)
#
#     def named_leaves(self):
#         return [('weight', self.weight), ('bias', self.bias)]


# class MetaConv2d(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         ignore = nn.Conv2d(*args, **kwargs)
#
#         self.in_channels = ignore.in_channels
#         self.out_channels = ignore.out_channels
#         self.stride = ignore.stride
#         self.padding = ignore.padding
#         self.dilation = ignore.dilation
#         self.groups = ignore.groups
#         self.kernel_size = ignore.kernel_size
#
#         self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
#
#         if ignore.bias is not None:
#             self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
#         else:
#             self.register_buffer('bias', None)
#
#     def forward(self, x):
#         return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#     def named_leaves(self):
#         return [('weight', self.weight), ('bias', self.bias)]


#
# class MetaBatchNorm2d(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         ignore = nn.BatchNorm2d(*args, **kwargs)
#
#         self.num_features = ignore.num_features
#         self.eps = ignore.eps
#         self.momentum = ignore.momentum
#         self.affine = ignore.affine
#         self.track_running_stats = ignore.track_running_stats
#
#         if self.affine:
#             self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
#             self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
#
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(self.num_features))
#             self.register_buffer('running_var', torch.ones(self.num_features))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#
#     def forward(self, x):
#         return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
#                             self.training or not self.track_running_stats, self.momentum, self.eps)
#
#     def named_leaves(self):
#         return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)
        # self.linear3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)
