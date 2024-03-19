import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Function, Variable

torch.set_default_dtype(torch.float64)



_FULL_QUANT = False
# _FULL_QUANT = True


class Pact(Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        # print('pact!!!')
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(x, min=0, max=alpha.item())
        scale = (2 ** k - 1) / alpha
        y_q = torch.round(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        x, alpha, = ctx.saved_tensors
        lower_bound = x < 0
        upper_bound = x > alpha
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None

class dorefaQuantizer:
    @classmethod
    def quantize_weight(cls, weight, bit_mask, fn, ):


        weight_q = weight.clone()

        weight_q = torch.where(bit_mask <= 32, cls.quantize_weight_middle(weight=weight, fn=fn), weight_q)
        weight_q = torch.where(bit_mask == 0, torch.zeros_like(weight), weight_q)

        return weight_q

    @classmethod
    def quantize_weight_middle(cls, weight, fn):
        """
            only when bit in [2, 4, 8, 16]
        """
        weight = weight.tanh()
        weight = weight / (2 * weight.abs().max()) + 0.5
        weight = fn(weight)
        weight = 2 * weight - 1

        return weight

    @classmethod
    def quantized_weight_to_one_bit(cls, weight):
        return torch.sign(weight) * torch.mean(torch.abs(weight))

    @classmethod
    def quantize_k(cls, input_ri, bit_mask):


        bit_mask = bit_mask.clip(1)
        scale=torch.pow(2,bit_mask)-1
        scale=scale.to(input_ri.device)
        # print ("scale device", scale.device)
        # print ("input_ri device", input_ri.device)
        out=torch.round(input_ri*scale)/scale

        return out.detach()



def uniform_quantize(bit_mask):
    class QuantizeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return dorefaQuantizer.quantize_k(input, bit_mask)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return QuantizeFunction.apply


# class WeightQuantizeFn(nn.Module):
#     def __init__(self, bit_mask):
#         super(WeightQuantizeFn, self).__init__()
#         self.bit_mask = bit_mask
#         self.uniform_quantize = uniform_quantize(bit_mask)
#
#     def forward(self, x):
#         return dorefaQuantizer.quantize_weight(x, self.bit_mask, self.uniform_quantize)

class WeightQuantizeFn(nn.Module):
    def __init__(self, bit_mask):
        super(WeightQuantizeFn, self).__init__()
        # self.bit_mask = bit_mask
        self.uniform_quantize = uniform_quantize(bit_mask)

    def forward(self, x, bit_mask):
        return dorefaQuantizer.quantize_weight(x, bit_mask, self.uniform_quantize)


class Conv2dQuantized(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dQuantized, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias)
        # self.register_buffer('mask', torch.full_like(self.weight, fill_value=32, dtype=torch.int))
        # self.quantize_fn = WeightQuantizeFn(bit_mask=self.mask)

    # def forward(self, input, ):
    #     # weight_q = self.quantize_fn(self.weight, self.mask)
    #     # return F.conv2d(input, weight_q, self.bias, self.stride,
    #     #                 self.padding, self.dilation, self.groups)
    #
    #     return super(Conv2dQuantized,self).forward(input)

# def bn_quantize_fn(bit_mask):
#     class BatchNormQuantized(nn.BatchNorm2d):
#         def __init__(self, num_features, eps=1e-7, momentum: float = 0.9, ) -> None:
#             super().__init__(num_features, eps=eps, momentum=momentum)
#             assert False, 'this function is depreciate due to slope performance'
#             self.bit_mask = bit_mask
#             self.quantize_fn = WeightQuantizeFn(bit_mask=bit_mask)

#         def forward(self, inputs):
#             weight_q = self.quantize_fn(self.weight)
#             return F.batch_norm(
#                 inputs, running_mean=self.running_mean, running_var=self.running_var,
#                 weight=weight_q, bias=self.bias, momentum=self.momentum,
#                 eps=self.eps
#             )

#     return BatchNormQuantized


# def linear_quantize_fn(bit_mask):
#     class LinearQuantized(torch.nn.Linear):
#         def __init__(
#                 self,
#                 in_channel: int,
#                 out_channel: int,
#                 bias: bool = True,
#         ):
#             super(LinearQuantized, self).__init__(in_channel, out_channel, bias)
#             self.quantized_weight = WeightQuantizeFn(bit_mask=bit_mask)

#         def forward(self, x):
#             w_q = self.quantized_weight(self.weight)
#             return F.linear(x, w_q, self.bias)

#     return LinearQuantized


# if __name__ == '__main__':
#     for scale in [32, 16, 8, 4, 2, 1, 0]:
#         conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#         img = torch.ones(1, 256, 56, 56)

#         mask = torch.ones_like(conv.weight) * scale
#         conv2d = conv2d_quantize_fn(bit_mask=mask)
#         conv1 = conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

#         with torch.no_grad():
#             out = conv(img)
#             print('base', out.max().item(), out.min().item())

#             out = conv1(img)
#             print(scale, out.max().item(), out.min().item())