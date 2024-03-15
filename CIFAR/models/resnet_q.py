import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sparselearning_path = os.path.join(parent_dir, 'sparselearning')
sys.path.insert(0, sparselearning_path)
from quantization import conv2d_quantize_fn, Pact


K = 32

from .init_utils import weights_init

__all__ = ['resnet']  # , 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

_AFFINE = True
# _AFFINE = False

_QUANT_ACT = False
# _QUANT_ACT = True

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)




def clip_relu(x, clip_value=10):
    return F.relu(x).clip(max=clip_value)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', name=None, mask=None, inner=None):
        assert name is not None, 'name is empty'
        assert mask is not None, 'mask is empty'
        assert inner is not None, f'inner is empty'

        super(BasicBlock, self).__init__()
        conv2d = conv2d_quantize_fn(mask[f'{name}.{inner}.conv1.weight'])
        self.conv1 = conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # batch_norm = bn_quantize_fn(mask[f'{name}.{inner}.bn1.weight'])#? check the name here
        # self.bn1 = batch_norm(planes)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.alpha1 = nn.Parameter(torch.tensor(10.))

        conv2d = conv2d_quantize_fn(mask[f'{name}.{inner}.conv2.weight'])
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # batch_norm = bn_quantize_fn(mask[f'{name}.{inner}.bn2.weight'])#? check the name here
        # self.bn2 = batch_norm(planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.quant_act = _QUANT_ACT
        self.pact = Pact.apply
        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            conv2d = conv2d_quantize_fn(mask[f'{name}.{inner}.conv1.weight'])  # ? check the name here
            self.downsample = conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

    def forward(self, x):
        # x: batch_size * in_c * h * w
        if self.quant_act:
            residual = x
            out = self.pact(self.bn1(self.conv1(x, )), self.alpha1, K)
            out = self.bn2(self.conv2(out, ))
            if self.downsample is not None:
                residual = self.bn3(self.downsample(x, ))
            out += residual
            out = self.pact(out, self.alpha2, K)
        else:
            residual = x
            out = F.relu(self.bn1(self.conv1(x, )))
            out = self.bn2(self.conv2(out, ))
            if self.downsample is not None:
                residual = self.bn3(self.downsample(x, ))
            out += residual
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mask=None):
        assert mask is not None, 'mask is empty'
        super(ResNet, self).__init__()
        _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]
        self.quant_act = _QUANT_ACT

        conv2d = conv2d_quantize_fn(mask['conv1.weight'])
        self.conv1 = conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.pact = Pact.apply
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1, mask=mask, name="layer1")
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2, mask=mask, name="layer2")
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2, mask=mask, name="layer3")
        self.linear = nn.Linear(_outputs[2], num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, name, mask=None):
        assert name is not None, 'name is empty'
        assert mask is not None, 'mask is empty'

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for inner_idx, stride in enumerate(strides):
            layers.append(
                block(self.in_planes, planes, stride, name=name, inner=inner_idx, mask=mask, use_pact=self.use_pact))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.quant_act:
            out = self.pact(self.bn(self.conv1(x, )), self.alpha1, K)
        else:
            out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(depth=32, dataset='cifar10'):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth
    n = (depth - 2) // 6
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError('Dataset [%s] is not supported.' % dataset)
    return ResNet(BasicBlock, [n] * 3, num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()