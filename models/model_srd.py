import sys
sys.path.append(".")
sys.path.append("..")
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.expansion = 1
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample
    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        expansion = 4
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * expansion)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out

class SSI(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSI, self).__init__()
        self.conv_Low = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_conv_Low = nn.BatchNorm2d(out_channels)
        self.conv_Super = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_conv_Super = nn.BatchNorm2d(out_channels)
        self.conv_to_Low = torch.nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn_conv_to_Low = nn.BatchNorm2d(in_channels)
        self.conv_to_Super = torch.nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn_conv_to_Super = nn.BatchNorm2d(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def adjust_resolution(self, f_to_low, f_to_super, f_low_shape, f_super_shape):
        if f_to_low.shape == f_low_shape and f_to_super.shape == f_super_shape:
            return f_to_low, f_to_super
        else:
            f_to_low = torch.nn.functional.interpolate(f_to_low, size=(f_low_shape[2], f_low_shape[3]), mode='bilinear')
            f_to_super = torch.nn.functional.interpolate(f_to_super, size=(f_super_shape[2], f_super_shape[3]), mode='bilinear')
            return f_to_low, f_to_super

    def forward(self, f_low, f_super):
        bsz = f_low.shape[0]
        f_low_shape = f_low.shape
        f_super_shape = f_super.shape
        # the 1*1 conv mapping for each branch's activation, then reshape into bsz*chw
        f_low = self.bn_conv_Low(self.conv_Low(f_low)).view(bsz, -1)
        f_super = self.bn_conv_Super(self.conv_Super(f_super)).view(bsz, -1)
        # obtain the self-similarity matrix for each branch
        A_low = F.normalize(torch.mm(f_low, torch.t(f_low)),dim=1)# add normalization on A 
        A_super = F.normalize(torch.mm(f_super, torch.t(f_super)),dim=1)

        M_super2low = F.normalize(torch.mm(A_low, torch.t(A_super)), dim=1)
        M_low2super = F.normalize(torch.mm(A_super, torch.t(A_low)), dim=1)
        
        f_to_low = self.bn_conv_to_Low(self.conv_to_Low(
            torch.mm(
                M_super2low, f_super
                ).view(f_super_shape)
            ))
        f_to_super = self.bn_conv_to_Super(self.conv_to_Super(
            torch.mm(
                M_low2super, f_low
                ).view(f_low_shape)
            ))
        # concat + bottleneck
        f_to_low, f_to_super = self.adjust_resolution(f_to_low, f_to_super, f_low_shape, f_super_shape)
        return f_to_low, f_to_super, \
            A_low, A_super, \
            M_super2low, M_low2super


class DualClassifiersSSI(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(DualClassifiersSSI, self).__init__()
        self.inplanes = 64

        m_low = OrderedDict()
        m_low['low_conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m_low['low_bn1'] = nn.BatchNorm2d(64)
        m_low['low_relu1'] = nn.ReLU(inplace=True)
        m_low['low_maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1_low= nn.Sequential(m_low)

        self.layer1_low = self._make_layer(block, 64, layers['0_low'])
        self.layer2_low = self._make_layer(block, 128, layers['1_low'], stride=2)
        self.layer3_low = self._make_layer(block, 256, layers['2_low'], stride=2)
        self.layer4_low = self._make_layer(block, 512, layers['3_low'], stride=2)
        self._reset_inplanes()
        ##

        m_super = OrderedDict()
        m_super['super_conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m_super['super_bn1'] = nn.BatchNorm2d(64)
        m_super['super_relu1'] = nn.ReLU(inplace=True)
        m_super['super_maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1_super= nn.Sequential(m_super)

        self.layer1_super = self._make_layer(block, 64, layers['0_super'])
        self.layer2_super = self._make_layer(block, 128, layers['1_super'], stride=2)
        self.layer3_super = self._make_layer(block, 256, layers['2_super'], stride=2)
        self.layer4_super = self._make_layer(block, 512, layers['3_super'], stride=2)
        self._reset_inplanes()

        self.avgpool_low = nn.AdaptiveAvgPool2d((1, 1))
        self.group2_low = nn.Sequential(
            OrderedDict([
                ('fc_low', nn.Linear(512 * block.expansion, num_classes))
            ])
        )
        self.avgpool_super = nn.AdaptiveAvgPool2d((1, 1))
        self.group2_super = nn.Sequential(
            OrderedDict([
                ('fc_super', nn.Linear(512 * block.expansion, num_classes))
            ])
        )
        
        self.ssi_layer1 = SSI(in_channels=64, out_channels=64)
        self.ssi_layer2 = SSI(in_channels=128, out_channels=128)
        self.ssi_layer3 = SSI(in_channels=256, out_channels=256)
        self.ssi_layer4 = SSI(in_channels=512, out_channels=512)

        self.bottle_1_Low = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_1_Low_relu = nn.ReLU(inplace=True)
        self.bn_bottle_1_Low = nn.BatchNorm2d(64)
        self.bottle_1_Super = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_1_Super_relu = nn.ReLU(inplace=True)
        self.bn_bottle_1_Super = nn.BatchNorm2d(64)

        self.bottle_2_Low = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_2_Low_relu = nn.ReLU(inplace=True)
        self.bn_bottle_2_Low = nn.BatchNorm2d(128)
        self.bottle_2_Super = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_2_Super_relu = nn.ReLU(inplace=True)
        self.bn_bottle_2_Super = nn.BatchNorm2d(128)

        self.bottle_3_Low = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_3_Low_relu = nn.ReLU(inplace=True)
        self.bn_bottle_3_Low = nn.BatchNorm2d(256)
        self.bottle_3_Super = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_3_Super_relu = nn.ReLU(inplace=True)
        self.bn_bottle_3_Super = nn.BatchNorm2d(256)

        self.bottle_4_Low = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_4_Low_relu = nn.ReLU(inplace=True)
        self.bn_bottle_4_Low = nn.BatchNorm2d(512)
        self.bottle_4_Super = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.bottle_4_Super_relu = nn.ReLU(inplace=True)
        self.bn_bottle_4_Super = nn.BatchNorm2d(512)

        self.ensemble_layer1 = nn.Linear(1024, 32)
        self.ensemble_layer2 = nn.Linear(32, num_classes)

        # soft-gate for final decision
        self.gate_fc1 = nn.Linear(1024, 32, bias=True)
        self.bn_gate_fc1 = nn.BatchNorm1d(32)
        self.gate_fc2 = nn.Linear(32, 3, bias=True)
        self.bn_gate_fc2 = nn.BatchNorm1d(3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_low, x_super):
        fg_low = self.group1_low(x_low)
        fg_super = self.group1_super(x_super)

        f1_low = self.layer1_low(fg_low)
        f1_super = self.layer1_super(fg_super)
        f1_to_low, f1_to_super, A1_low, A1_super, M1_s2l, M1_l2s = self.ssi_layer1(f1_low, f1_super)
        f1_low = self.bn_bottle_1_Low(self.bottle_1_Low_relu(self.bottle_1_Low(torch.cat([f1_low, f1_to_low], dim=1))))
        f1_super = self.bn_bottle_1_Super(self.bottle_1_Super_relu(self.bottle_1_Super(torch.cat([f1_super, f1_to_super], dim=1))))

        f2_low = self.layer2_low(f1_low)
        f2_super = self.layer2_super(f1_super)
        f2_to_low, f2_to_super, A2_low, A2_super, M2_s2l, M2_l2s = self.ssi_layer2(f2_low, f2_super)
        f2_low = self.bn_bottle_2_Low(self.bottle_2_Low_relu(self.bottle_2_Low(torch.cat([f2_low, f2_to_low], dim=1))))
        f2_super = self.bn_bottle_2_Super(self.bottle_2_Super_relu(self.bottle_2_Super(torch.cat([f2_super, f2_to_super], dim=1))))

        f3_low = self.layer3_low(f2_low)
        f3_super = self.layer3_super(f2_super)
        f3_to_low, f3_to_super, A3_low, A3_super, M3_s2l, M3_l2s = self.ssi_layer3(f3_low, f3_super)
        f3_low = self.bn_bottle_3_Low(self.bottle_3_Low_relu(self.bottle_3_Low(torch.cat([f3_low, f3_to_low], dim=1))))
        f3_super = self.bn_bottle_3_Super(self.bottle_3_Super_relu(self.bottle_3_Super(torch.cat([f3_super, f3_to_super], dim=1))))

        f4_low = self.layer4_low(f3_low)
        f4_super = self.layer4_super(f3_super)
        f4_to_low, f4_to_super, A4_low, A4_super, M4_s2l, M4_l2s = self.ssi_layer4(f4_low, f4_super)
        f4_low = self.bn_bottle_4_Low(self.bottle_4_Low_relu(self.bottle_4_Low(torch.cat([f4_low, f4_to_low], dim=1))))
        f4_super = self.bn_bottle_4_Super(self.bottle_4_Super_relu(self.bottle_4_Super(torch.cat([f4_super, f4_to_super], dim=1))))
        
        f_low = self.avgpool_low(f4_low)
        f_low = f_low.view(f_low.size(0), -1)
        output_low = self.group2_low(f_low)
               
        
        f_super = self.avgpool_super(f4_super)
        f_super = f_super.view(f_super.size(0), -1)
        output_super = self.group2_super(f_super)

        gate_weight = F.relu(
            torch.tanh(self.bn_gate_fc2(self.gate_fc2(
                F.relu(self.bn_gate_fc1(self.gate_fc1(
                    torch.cat([f_low, f_super], dim=1)
                    )))
                )))
        )
        output_aux = self.ensemble_layer2(
            F.relu(self.ensemble_layer1(
                torch.cat([f_low, f_super], dim=1)
                ))
            )
        output_ensemble = output_low * gate_weight[:, 0].view(-1, 1) + \
            output_super * gate_weight[:, 1].view(-1, 1) + \
            output_aux * gate_weight[:, 2].view(-1, 1)
        return output_low, output_super, output_aux, output_ensemble, \
            [M1_s2l, M2_s2l, M3_s2l, M4_s2l], [M1_l2s, M2_l2s, M3_l2s, M4_l2s],\
            [A1_low, A2_low, A3_low, A4_low], [A1_super, A2_super, A3_super, A4_super]

    def _reset_inplanes(self, inplanes=64):
        self.inplanes = inplanes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes)) 
        return nn.Sequential(*layers)

##
def dualnet18_18_ssi(model_root=None, **kwargs):
    layers_dict = { '0_low':2, '1_low':2, '2_low':2, '3_low':2,
        '0_super':2, '1_super':2, '2_super':2, '3_super':2, }
    model = DualClassifiersSSI(BasicBlock, layers_dict, **kwargs) #[2, 2, 2, 2]
    return model

class DualNet18_18(nn.Module):
    def __init__(self, sr_net):
        super(DualNet18_18, self).__init__()
        self.add_module('sr_net', sr_net)
        self.add_module('dual_classifiers', dualnet18_18_ssi())

    def forward(self, x_low):
        x_super, _, _, _ = self.sr_net(x_low) # 
        output_low, output_super, output_aux, output_ensemble, M_s2l_list, M_l2s_list, A_low_list, A_super_list = self.dual_classifiers(x_low, x_super)
        return output_low, output_super, output_aux, output_ensemble, M_s2l_list, M_l2s_list, A_low_list, A_super_list
        
   