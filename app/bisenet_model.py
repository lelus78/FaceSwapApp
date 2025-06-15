# app/bisenet_model.py
# Questo file contiene la definizione del modello BiSeNet per il face parsing,
# eliminando la necessità di installare la libreria 'facelib'.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_first = nn.BatchNorm2d(out_channels)
        self.conv_3x3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_first(x)
        x = self.bn_first(x)
        x = self.conv_3x3(x)
        x_attn = F.adaptive_avg_pool2d(x, 1)
        x_attn = self.conv_1x1(x_attn)
        x_attn = self.sigmoid(x_attn)
        x = x * x_attn
        return x

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_sp, x_cp):
        x = torch.cat([x_sp, x_cp], dim=1)
        x = self.conv_block(x)
        x_attn = F.adaptive_avg_pool2d(x, 1)
        x_attn = self.conv1x1(x_attn)
        x_attn = self.relu(x_attn)
        x_attn = self.conv_final(x_attn)
        x_attn = self.sigmoid(x_attn)
        x = x * x_attn + x
        return x

class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
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

class BiSeNet(torch.nn.Module):
    def __init__(self, n_classes, n_res_layers=18):
        super(BiSeNet, self).__init__()
        if n_res_layers == 18:
            self.context_path = ResNet(BasicBlock, [2, 2, 2, 2])
        else: # Add more if needed, e.g., ResNet34
            raise ValueError("Unsupported ResNet layer count.")

        self.spatial_path = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=1, stride=1, padding=0)
        )
        
        self.arm1 = AttentionRefinementModule(256, 128)
        self.arm2 = AttentionRefinementModule(512, 128)
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_head = ConvBlock(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(128, n_classes, kernel_size=1)
        self.up_billing = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        H, W = x.size()[2:]
        # Context path
        x_context = self.context_path.conv1(x)
        x_context = self.context_path.bn1(x_context)
        x_context = self.context_path.relu(x_context)
        x_context = self.context_path.maxpool(x_context)
        c1 = self.context_path.layer1(x_context)
        c2 = self.context_path.layer2(c1)
        c3 = self.context_path.layer3(c2)
        c4 = self.context_path.layer4(c3)
        
        avg_pool = F.adaptive_avg_pool2d(c4, 1)
        
        # ARM
        c3_arm = self.arm1(c3)
        c4_arm = self.arm2(c4)
        c4_arm = self.up_billing(c4_arm)
        
        context_features = torch.cat([c3_arm, c4_arm], dim=1)
        
        # Spatial path
        x_sp = self.spatial_path(x)
        
        # FFM
        x_fuse = self.ffm(x_sp, context_features)
        
        # Head
        x_head = self.conv_head(x_fuse)
        x_head = self.up_billing(x_head)
        x_head = self.up_billing(x_head)
        
        out = self.conv_out(x_head)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        return out, None, None # Return tuple to match expected structure