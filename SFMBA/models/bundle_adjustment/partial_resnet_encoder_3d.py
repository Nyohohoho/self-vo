import torch.nn as nn
from ..layers import PartialConv3d, PartialConvBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                         bias=False, multi_channel=True, return_mask=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, x_mask):
        residual = x
        residual_mask = x_mask

        out, out_mask = self.conv1(x, x_mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)

        if self.downsample is not None:
            residual, residual_mask = self.downsample(x, x_mask)

        out += residual
        out_mask += residual_mask
        out = self.relu(out)

        return out, out_mask


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = PartialConv3d(inplanes, planes, kernel_size=1,
                                   bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = PartialConv3d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                   bias=False, multi_channel=True, return_mask=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = PartialConv3d(planes, planes * self.expansion, kernel_size=1,
                                   bias=False, multi_channel=True, return_mask=True)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, x_mask):
        residual = x
        residual_mask = x_mask

        out, out_mask = self.conv1(x, x_mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, out_mask = self.conv3(out, out_mask)
        out = self.bn3(out)

        if self.downsample is not None:
            residual, residual_mask = self.downsample(x, x_mask)

        out += residual
        out_mask += residual_mask
        out = self.relu(out)

        return out, out_mask


class PartialResNet3d(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(PartialResNet3d, self).__init__()
        self.conv1 = PartialConv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=1,
                                   bias=False, multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=(1, 2, 2))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1, 2, 2))

        for m in self.modules():
            if isinstance(m, PartialConv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = PartialConvBlock(
                conv_layer=PartialConv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                         bias=False, multi_channel=True, return_mask=True),
                batch_norm=nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)

    def forward(self, x, x_mask):
        features = []
        masks = []

        x, x_mask = self.conv1(x, x_mask)
        x = self.bn1(x)
        x = self.relu(x)
        # Save the feature and mask from conv1
        features.append(x)
        masks.append(x_mask)

        for block in self.layer1:
            x, x_mask = block(x, x_mask)
        # Save the feature and mask from layer1
        features.append(x)
        masks.append(x_mask)

        for block in self.layer2:
            x, x_mask = block(x, x_mask)
        # Save the feature and mask from layer2
        features.append(x)
        masks.append(x_mask)

        for block in self.layer3:
            x, x_mask = block(x, x_mask)
        # Save the feature and mask from layer3
        features.append(x)
        masks.append(x_mask)

        for block in self.layer4:
            x, x_mask = block(x, x_mask)
        # Save the feature and mask from layer4
        features.append(x)
        masks.append(x_mask)

        return features, masks


class PartialResNetEncoder(nn.Module):

    def __init__(self):
        super(PartialResNetEncoder, self).__init__()
        # Note that it is ResNet-18
        self.model = PartialResNet3d(BasicBlock, [2, 2, 2, 2])

    def forward(self, x, x_mask):
        features = self.model(x, x_mask)
        return features


if __name__ == '__main__':
    import torch
    import torch.cuda

    net = PartialResNetEncoder().cuda()
    image = torch.rand(1, 3, 6, 256, 256).cuda()
    mask = (torch.rand(1, 3, 6,  256, 256).cuda() > 0.99).float()

    features, masks = net(image, mask)
    for f in features:
        print(f.shape)
