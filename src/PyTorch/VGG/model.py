import torch
import torch.nn as nn

# VGG16 的配置列表
# 数字代表卷积核个数 (Channels)
# 'M' 代表最大池化层 (Maxpool)
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        
        # CIFAR-10 经过 5 次池化后，32x32 的图片会变成 1x1
        # 所以这里的输入维度是 512 * 1 * 1
        self.classifier = nn.Sequential(
            nn.Linear(4*4*512, 4*4*512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4*4*512, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # RGB 图片输入通道为 3
        
        for v in cfg:
            if v == 'M':
                # 如果是 'M'，添加一个 MaxPool2d (kernel=2, stride=2)
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 如果是数字 v，添加一个 Conv2d
                # 重点：VGG 统一使用 kernel_size=3, padding=1
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                
                # 完整的层通常是: Conv -> BatchNorm(可选) -> ReLU
                # 我们这里简化版：Conv -> ReLU
                layers += [conv2d, 
                      nn.BatchNorm2d(v),
                      nn.ReLU(inplace=True)]
                
                # 关键：更新 in_channels，以便下一层知道输入是多少
                in_channels = v 
                
        return nn.Sequential(*layers)