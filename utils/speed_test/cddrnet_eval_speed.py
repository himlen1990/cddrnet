import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from modules import CompactBasicBlock, BasicBlock, Bottleneck, DAPPM, segmenthead, GhostBottleneck

bn_mom = 0.1
BatchNorm2d = nn.BatchNorm2d


class CompactDualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True):
        super(CompactDualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          #BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          #BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(CompactBasicBlock, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_ghost_bottleneck(GhostBottleneck, highres_planes , highres_planes, 1)
        
        self.layer5 =  self._make_ghost_bottleneck(GhostBottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)            

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def _make_ghost_bottleneck(self, block, inplanes, planes, blocks, stride=1):
        if stride != 1 or inplanes != planes * 2:
            out_channel = planes * 2
        else:
            out_channel = planes

        cfg = [[3,  96, out_channel, 0, 1]] # k, t, c, SE, s             
        input_channel = inplanes
        layers = []
        for k, exp_size, c, se_ratio, s in cfg:
            output_channel = c
            hidden_channel = self._make_divisible(exp_size, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio))
            input_channel = output_channel
        return nn.Sequential(*layers)
    

    
    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.final_layer(x + x_)

        if self.augment: 
            x_extra = self.seghead_extra(temp)
            return [x_extra, x_]
        else:
            return x_      

def get_seg_model(cfg, **kwargs):
    model = CompactDualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True)
    return model


if __name__ == '__main__':
   import time
   device = torch.device('cuda')
   #torch.backends.cudnn.enabled = True
   #torch.backends.cudnn.benchmark = True
   model = CompactDualResNet(BasicBlock, [2, 2, 2, 2], num_classes=11, planes=32, spp_planes=128, head_planes=64)
   model.eval()
   model.to(device)
   iterations = None
   #input = torch.randn(1, 3, 1024, 2048).cuda()
   input = torch.randn(1, 3, 720, 960).cuda()
   with torch.no_grad():
       for _ in range(10):
           model(input)
           
       if iterations is None:
           elapsed_time = 0
           iterations = 100
           while elapsed_time < 1:
               torch.cuda.synchronize()
               torch.cuda.synchronize()
               t_start = time.time()
               for _ in range(iterations):
                   model(input)
               torch.cuda.synchronize()
               torch.cuda.synchronize()
               elapsed_time = time.time() - t_start
               iterations *= 2
           FPS = iterations / elapsed_time
           iterations = int(FPS * 6)
            
       print('=========Speed Testing=========')
       torch.cuda.synchronize()
       torch.cuda.synchronize()
       t_start = time.time()
       for _ in range(iterations):
           model(input)
       torch.cuda.synchronize()
       torch.cuda.synchronize()
       elapsed_time = time.time() - t_start
       latency = elapsed_time / iterations * 1000
   torch.cuda.empty_cache()
   FPS = 1000 / latency
   print(FPS)
