import torch.nn as nn
import torch
class Bottleneck(nn.Module):


    def __init__(self, in_channel, out_channel,expansion, stride=1,  downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
    
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=width,
                               kernel_size=5, stride=1, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv4 = nn.Conv2d(in_channels=width, out_channels=width, groups=width,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.bn4 = nn.BatchNorm2d(width)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*expansion)
        self.Mish = nn.Mish(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Mish(out)
        f=out

        out1 = self.conv4(f)#3*3
        out1 = self.bn4(out1)
        out1 = self.pool(out1)
        
        out2 = self.conv2(f)#5*5
        out2 = self.bn2(out2)
        out2 = self.pool(out2)
        
        out3 = self.pool(f)#maxpool
        
        out = out1+out2+out3

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.Mish(out)
        
        out = out + identity


        return out
class Remulbnet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(Remulbnet, self).__init__()
        self.include_top = include_top
        self.in_channel = 32

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=5, stride=2,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.Mish = nn.Mish(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 54, blocks_num[0],expansion=1)
        self.layer2 = self._make_layer(block, 126, blocks_num[1], stride=2,expansion=2)
        self.layer3 = self._make_layer(block, 258, blocks_num[2], stride=2,expansion=3)
        self.layer4 = self._make_layer(block, 586, blocks_num[3], stride=2,expansion=2)
        if self.include_top:

            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))  # output size = (1, 1)

            self.fc = nn.Linear(586*2, num_classes)
            

            

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, expansion, stride=1):
        downsample = None   #跨层例（64->128）
        if stride != 1 or self.in_channel != channel *expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel *expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            expansion,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                expansion,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.Mish(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:

            f = x
            x = self.avg_pool(x)
            f = self.max_pool(f)
            x = x + f
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
def remulbnet(num_classes=1000, include_top=True):

    '''groups = 32
    width_per_group = 4'''
    return Remulbnet(Bottleneck, [1, 1, 2, 1],num_classes=num_classes,include_top=include_top)
