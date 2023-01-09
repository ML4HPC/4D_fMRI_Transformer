import torch
import torch.nn as nn
from collections import OrderedDict


import math
import torch.nn.functional as F
from torch.autograd import Variable

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def tuple_prod(x):
    prod = 1
    for xx in x:
        prod *= xx
    return prod

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,drop_rate=0.4):
        """
        green_block(inp, filters, name=None)
        ------------------------------------
        Implementation of the special residual block used in the paper. The block
        consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
        units, with a residual connection from the input `inp` to the output. Used
        internally in the model. Can be used independently as well.
        Note that images must come with dimensions "c, H, W, D"
        Parameters
        ----------
        `inp`: An keras.layers.layer instance, required
            The keras layer just preceding the green block.
        `out_channels`: integer, required
            No. of filters to use in the 3D convolutional block. The output
            layer of this green block will have this many no. of channels.
        Returns
        -------
        `out`: A keras.layers.Layer instance
            The output of the green block. Has no. of channels equal to `filters`.
            The size of the rest of the dimensions remains same as in `inp`.
        """
        super(GreenBlock, self).__init__()
        self.Drop_Rate = drop_rate
        # Define block
        # Batchnorm -> GroupNorm
        self.block = nn.Sequential(OrderedDict([
            ('group_norm0', nn.GroupNorm(num_channels=in_channels, num_groups=in_channels // 4)),
            #('norm0', nn.BatchNorm3d(num_features=in_channels)),
            ('relu0', nn.LeakyReLU(inplace=True)),
            ('conv0', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('group_norm1', nn.GroupNorm(num_channels=out_channels, num_groups=in_channels // 4)),
            #('norm1', nn.BatchNorm3d(num_features=out_channels)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ]))

    def forward(self, inputs):
        x_res = inputs
        #torch.cuda.nvtx.range_push("groupnorm0")
        #x = self.group_norm0(inputs)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("relu0")
        #x = self.relu0(x)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("conv0")
        #x = self.conv0(x)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("groupnorm1")
        #x = self.group_norm1(x)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("relu1")
        #x = self.relu1(x)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("conv2")
        #x = self.conv2(x)
        #torch.cuda.nvtx.range_pop()
        #torch.cuda.nvtx.range_push("dropout")
        #x = torch.nn.functional.dropout(x, p=self.Drop_Rate, training=self.training)
        #torch.cuda.nvtx.range_pop()
        x = torch.nn.functional.dropout(self.block(inputs), p=self.Drop_Rate, training=self.training)
        return x + x_res



class UpGreenBlock(nn.Sequential):
    def __init__(self, in_features, out_features, shape, Drop_Rate):
        super(UpGreenBlock, self).__init__()

        self.add_module('conv', nn.Conv3d(in_features, out_features, kernel_size=1, stride=1))
        self.add_module('up', nn.Upsample(size=shape))
        self.add_module('green', GreenBlock(out_features, out_features, Drop_Rate))






### MobileNet v2###
def conv_bn(inp, oup, stride, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1x1_bn(inp, oup, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, last_channel, sample_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 4
        last_channel = last_channel
        # interverted_residual_setting = [
        #     # t, c, n, s
        #     [1,  8, 1, (1,1,1)],
        #     [6,  12, 2, (2,2,2)],
        #     [6,  16, 2, (2,2,2)],
        # ]
        # interverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 8, 1, (1, 1, 1)],
        #     [6, 12, 2, (2, 2, 2)],
        #     [6, 16, 2, (2, 2, 2)],
        #     [6, 32, 3, (2, 2, 2)],
        #     [6, 48, 2, (1, 1, 1)],
        # ]

        interverted_residual_setting = [
        #     # t, c, n, s
            [1, 8, 1, (1, 1, 1)],
            [6, 12, 2, (2, 2, 2)],
            [6, 16, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 48, 2, (1, 1, 1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult) # input_channel: 4
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (2, 2, 2), nlin_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel, nlin_layer=nn.ReLU))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

### MobileNet v3###
##https://github.com/kuan-wang/pytorch-mobilenet-v3

def conv_bn(inp, oup, stride, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1x1_bn(inp, oup, conv_layer=nn.Conv3d, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE',drop_rate=0.4, norm_layer=nn.BatchNorm3d, nlin_layer=nn.ReLU):
        super(MobileBottleneck, self).__init__()
        self.Drop_Rate = drop_rate
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv3d
        norm_layer = norm_layer
        if nl == 'RE':
            nlin_layer = nlin_layer # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + torch.nn.functional.dropout(self.conv(x),p=self.Drop_Rate, training=self.training)
        else:
            return torch.nn.functional.dropout(self.conv(x),p=self.Drop_Rate, training=self.training)



