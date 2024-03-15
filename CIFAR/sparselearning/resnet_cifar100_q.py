'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sparselearning_path = os.path.join(parent_dir, 'sparselearning')
sys.path.insert(0, sparselearning_path)
from quantization import Pact,Conv2dQuantized

pact=Pact.apply

K = 32


_QUANT_ACT = False
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dQuantized(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dQuantized(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.alpha2 = nn.Parameter(torch.tensor(10.))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dQuantized(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if _QUANT_ACT:
            out = pact(self.bn1(self.conv1(x)),self.alpha1,K)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.pact(out,self.alpha2,K)
        else:
            out = pact(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2dQuantized(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dQuantized(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2dQuantized(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.alpha1 = nn.Parameter(torch.tensor(10.))
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.alpha3 = nn.Parameter(torch.tensor(10.))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dQuantized(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if _QUANT_ACT:
            out = pact(self.bn1(self.conv1(x)),self.alpha1,K)
            out = pact(self.bn2(self.conv2(out)),self.alpha2,K)
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = pact(out,self.alpha3,K)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2dQuantized(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)

        self.alpha = nn.Parameter(torch.tensor(10.))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if _QUANT_ACT:
            out = pact(self.bn1(self.conv1(x)),self.alpha,K)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out


def ResNet18(c=1000):
    return ResNet(BasicBlock, [2,2,2,2],c)

def ResNet34(c=10):
    return ResNet(BasicBlock, [3,4,6,3],c)

def ResNet50(c=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], c)

def ResNet101(c=10):
    return ResNet(Bottleneck, [3,4,23,3],c)

def ResNet152(c=10):
    return ResNet(Bottleneck, [3,8,36,3],c)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
