import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1. 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2. 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 3. 处理 Shortcut (捷径)
        # 如果输入和输出的维度不一样（比如 stride=2 导致图片变小了，或者通道数变了）
        # 我们需要用 1x1 卷积把 x 的维度调整成和 out 一样，才能相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 核心魔法在这里： F(x) + x
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # --- 针对 CIFAR-10 的修改 ---
        # 原版是 Conv 7x7, stride 2, padding 3 + MaxPool
        # 这里改为 Conv 3x3, stride 1, padding 1 (保持 32x32 尺寸)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # ---------------------------

        # 这里的 layer1, 2, 3, 4 分别对应 ResNet 的四个 Stage
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # 例如 stride=2: [2, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # 注意：CIFAR 版通常去掉了这里的 MaxPool
        
        out = self.layer1(out) # 32x32
        out = self.layer2(out) # 16x16
        out = self.layer3(out) # 8x8
        out = self.layer4(out) # 4x4
        
        out = F.avg_pool2d(out, 4) # 全局平均池化 -> 1x1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 方便调用的函数：ResNet-18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])