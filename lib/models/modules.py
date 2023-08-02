import torch
import torch.nn as nn
import torch.nn.functional as F
import math


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    
    
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None):
        super(GhostModule, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()
        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )       

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True, mode='original')

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False, mode='original')
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class CompactBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, kernel_size=3):
        super(CompactBasicBlock, self).__init__()

        self.oup = planes
        init_channels = math.ceil(planes / 2)
        new_channels = init_channels
        dw_size =3
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inplanes, init_channels, kernel_size, stride, kernel_size//2,
                      bias=False),
            nn.BatchNorm2d(init_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(init_channels, init_channels, kernel_size, 1,
                      kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels, momentum=bn_mom)

        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels, momentum=bn_mom),
            nn.Conv2d(new_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels, momentum=bn_mom)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        x1 = self.primary_conv(x)

        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        out = out[:,:self.oup,:,:]

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out
        
