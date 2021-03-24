import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init2(m[-1], val=0)
    else:
        constant_init2(m, val=0)

def kaiming_init2(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init2(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class RCF(nn.Module):
    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=2):
        super(RCF, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        self.se_backone = SELayer(channel=inplanes)
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init2(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, backbone, bypass):
        batch, channel, height, width = backbone.size()
        if self.pool == 'att':
            input_x = bypass
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(backbone)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(bypass)
        return context

    def forward(self, backbone, bypass):
        # [N, C, 1, 1]
        context = self.spatial_pool(backbone, bypass)
        backbone = self.se_backone(backbone) 
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = backbone * channel_mul_term
        else:
            out = backbone
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            backbone = backbone + channel_add_term
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_NoBN(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_NoBN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out

class Bottleneck_NoBN(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_NoBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.relu1 = nn.PReLU(num_parameters=planes)
        self.relu2 = nn.PReLU(num_parameters=planes)
        self.relu3 = nn.PReLU(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        return out

class MRC_Stage(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(MRC_Stage, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.rcf1 = RCF(inplanes=num_channels[0], planes=num_channels[0])
        self.rcf2 = RCF(inplanes=num_channels[1], planes=num_channels[1])

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride
                )
            )
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1): 
            fuse_layer = []
            for j in range(num_branches):
                if j > i: 
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0
                            ),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i: 
                    fuse_layer.append(None)
                else: 
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1
                                    )
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1
                                    ),
                                    nn.PReLU(num_parameters=num_outchannels_conv3x3)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)): 
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0]) 
            for j in range(1, self.num_branches):
                if i == j: 
                    y = self.rcf2(backbone=x[j], bypass=y)
                else: 
                    y = self.rcf1(backbone=y, bypass=self.fuse_layers[i][j](x[j]))
            x_fuse.append(self.relu(y))
        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock_NoBN,
    'BOTTLENECK': Bottleneck_NoBN
}

class MRC_Module(nn.Module):
    def __init__(self, trans_in_channel=32):
        super(MRC_Module, self).__init__()
        self.trans_in_channel = trans_in_channel
        # 2
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([self.trans_in_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4,4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=True)
        # 3
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4, 4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=True)
        # 4
        num_channels = [32, 32]
        block = blocks_dict['BASIC']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            num_modules=1,num_branches=2,num_blocks=[4, 4],num_channels=num_channels,block=block,fuse_method='SUM', num_inchannels=num_channels,
            multi_scale_output=False)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1
                            ),
                            nn.PReLU(num_parameters=num_channels_cur_layer[i])
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1
                            ),
                            nn.PReLU(num_parameters=outchannels)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, num_modules,num_branches,num_blocks,num_channels,block,fuse_method, 
                    num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                MRC_Stage(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x_list = []
        for i in range(2):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(2):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list[0]

class Net_Recursive(nn.Module):
    def __init__(self, interpolate='nearest'):
        super(Net_Recursive, self).__init__()
        self.interpolate = interpolate
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.resbk1 = BasicBlock_NoBN(inplanes=64, planes=64, stride=1, downsample=None)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) # channel reduction
        self.resbk2 = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)                            
        self.relu1 = nn.PReLU(num_parameters=64)
        self.relu2 = nn.PReLU(num_parameters=32)

        self.mrc = MRC_Module()

        self.ps1_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)
        self.conv_inter_ps1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps1 = nn.PReLU(num_parameters=32)
        self.resbk_inter_ps = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None) 
        self.conv_inter_ps2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps2 = nn.PReLU(num_parameters=32)
        self.ps2_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)

        self.conv_sr1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_sr2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu_sr1 = nn.PReLU(num_parameters=32)
        self.relu_sr2 = nn.PReLU(num_parameters=32)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # deep LSTM
        self.conv_concate1 = nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0, bias=True)
        self.relu_concate1 = nn.PReLU(num_parameters=32)
        self.resbk_concate = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)
        self.conv_concate2 = nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1, bias=True)
        self.relu_concate2 = nn.PReLU(num_parameters=32)
        self.conv_i = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
    
    def forward(self, input, tmp_FLAG=False):
        tmp_list = []
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        y = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        x = self.relu1(self.conv1(input))
        x = self.resbk1(x)
        x = self.relu2(self.conv2(x))
        x = self.resbk2(x)

        # loop mechanism, with conv-lstm at first
        for loop in range(3):
            concat_feature = self.relu_concate2(self.conv_concate2(
                self.resbk_concate(
                    self.relu_concate1(self.conv_concate1(torch.cat([x, y], dim=1)))
                )
            ))
            i = self.conv_i(concat_feature)
            f = self.conv_f(concat_feature)
            g = self.conv_g(concat_feature)
            o = self.conv_o(concat_feature)
            c = f * c + i * g # c: hidden state
            h = o * torch.tanh(c) # h: LSTM output
            y = self.mrc(h) + h 
            y_upsampled = F.pixel_shuffle(self.ps2_conv(
                self.relu_inter_ps2(self.conv_inter_ps2(
                    self.resbk_inter_ps(
                        self.relu_inter_ps1(self.conv_inter_ps1(
                        F.pixel_shuffle(self.ps1_conv(y), upscale_factor=2)
                        )
                    ))
                ))
                ), upscale_factor=2)
            output = self.conv_final(
                self.relu_sr2(self.conv_sr2(
                    self.relu_sr1(self.conv_sr1(y_upsampled))))) + torch.nn.functional.interpolate(input, scale_factor=4, mode=self.interpolate)
            if tmp_FLAG:
                tmp_list.append(output)      

        if tmp_FLAG:
            return output, tmp_list[:-1]
        else:
            return output

class Net_Loop_Teacher(nn.Module):
    def __init__(self, interpolate='nearest'):
        super(Net_Loop_Teacher, self).__init__()
        self.interpolate = interpolate
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.resbk1 = BasicBlock_NoBN(inplanes=64, planes=64, stride=1, downsample=None) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) # channel reduction
        self.resbk2 = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)                            
        self.relu1 = nn.PReLU(num_parameters=64)
        self.relu2 = nn.PReLU(num_parameters=32)

        self.mrc = MRC_Module()

        self.ps1_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)
        self.conv_inter_ps1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps1 = nn.PReLU(num_parameters=32)
        self.resbk_inter_ps = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None) 
        self.conv_inter_ps2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps2 = nn.PReLU(num_parameters=32)
        self.ps2_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)

        self.conv_sr1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_sr2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu_sr1 = nn.PReLU(num_parameters=32)
        self.relu_sr2 = nn.PReLU(num_parameters=32)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # deep LSTM
        self.conv_concate1 = nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0, bias=True)
        self.relu_concate1 = nn.PReLU(num_parameters=32)
        self.resbk_concate = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)
        self.conv_concate2 = nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1, bias=True)
        self.relu_concate2 = nn.PReLU(num_parameters=32)
        self.conv_i = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3) 
        y = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda()

        x = self.relu1(self.conv1(input))
        x = self.resbk1(x)
        x = self.relu2(self.conv2(x))
        x = self.resbk2(x)

        # loop mechanism, with conv-lstm at first
        for loop in range(3):
            concat_feature = self.relu_concate2(self.conv_concate2(
                self.resbk_concate(
                    self.relu_concate1(self.conv_concate1(torch.cat([x, y], dim=1)))
                )
            ))
            i = self.conv_i(concat_feature)
            f = self.conv_f(concat_feature)
            g = self.conv_g(concat_feature)
            o = self.conv_o(concat_feature)

            c = f * c + i * g #
            h = o * torch.tanh(c) # 

            y = self.mrc(h) + h 
            y_upsampled = F.pixel_shuffle(self.ps2_conv(
                self.relu_inter_ps2(self.conv_inter_ps2(
                    self.resbk_inter_ps(
                        self.relu_inter_ps1(self.conv_inter_ps1(
                        F.pixel_shuffle(self.ps1_conv(y), upscale_factor=2)
                        )
                    ))
                ))
                ), upscale_factor=2)
            output = self.conv_final(
                self.relu_sr2(self.conv_sr2(
                    self.relu_sr1(self.conv_sr1(y_upsampled))))) + torch.nn.functional.interpolate(input, scale_factor=4, mode=self.interpolate)
        return output, y_upsampled, y, h

# distill knowledge from lstm output, sr features and final output
class Net_Student(nn.Module):
    def __init__(self, interpolate='nearest'):
        super(Net_Student, self).__init__()
        self.interpolate = interpolate
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.resbk1 = BasicBlock_NoBN(inplanes=64, planes=64, stride=1, downsample=None) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) # channel reduction
        self.resbk2 = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)                            
        self.relu1 = nn.PReLU(num_parameters=64)
        self.relu2 = nn.PReLU(num_parameters=32)

        self.mrc = MRC_Module()

        self.ps1_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)
        self.conv_inter_ps1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps1 = nn.PReLU(num_parameters=32)
        self.resbk_inter_ps = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None) 
        self.conv_inter_ps2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu_inter_ps2 = nn.PReLU(num_parameters=32)
        self.ps2_conv = nn.Conv2d(32, 32*4, kernel_size=3, stride=1, padding=1)

        self.conv_sr1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_sr2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu_sr1 = nn.PReLU(num_parameters=32)
        self.relu_sr2 = nn.PReLU(num_parameters=32)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # keep the struture, without recursion
        self.conv_concate1 = nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0, bias=True)
        self.relu_concate1 = nn.PReLU(num_parameters=32)
        self.resbk_concate = BasicBlock_NoBN(inplanes=32, planes=32, stride=1, downsample=None)
        self.conv_concate2 = nn.Conv2d(32, 32, kernel_size=3,stride=1,padding=1, bias=True)
        self.relu_concate2 = nn.PReLU(num_parameters=32)
        self.conv_i = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
    
    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        y = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = torch.autograd.Variable(torch.zeros(batch_size, 32, row, col)).cuda()

        x = self.relu1(self.conv1(input))
        x = self.resbk1(x)
        x = self.relu2(self.conv2(x))
        x = self.resbk2(x)

        # no recursion
        concat_feature = self.relu_concate2(self.conv_concate2(
            self.resbk_concate(
                self.relu_concate1(self.conv_concate1(torch.cat([x, y], dim=1)))
            )
        ))
        i = self.conv_i(concat_feature)
        f = self.conv_f(concat_feature)
        g = self.conv_g(concat_feature)
        o = self.conv_o(concat_feature)

        c = f * c + i * g # 
        h = o * torch.tanh(c) # 
        y = self.mrc(h) + h 
        y_upsampled = F.pixel_shuffle(self.ps2_conv(
            self.relu_inter_ps2(self.conv_inter_ps2(
                self.resbk_inter_ps(
                    self.relu_inter_ps1(self.conv_inter_ps1(
                    F.pixel_shuffle(self.ps1_conv(y), upscale_factor=2)
                    )
                ))
            ))
            ), upscale_factor=2)
        output = self.conv_final(
            self.relu_sr2(self.conv_sr2(
                self.relu_sr1(self.conv_sr1(y_upsampled))))) + torch.nn.functional.interpolate(input, scale_factor=4, mode=self.interpolate)
         
        return output, y_upsampled, y, h 




    