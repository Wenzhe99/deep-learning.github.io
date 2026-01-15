from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_cifar_dataloader(batch_size=16):
    # 定义数据转换
    data_transform = transforms.Compose([
        # 1. 调整大小 (Resize) - 填空！
        transforms.Resize((128, 128)),
        
        # 2. 转为张量
        transforms.ToTensor(),
        
        # 3. 标准化 (Normalize)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载并加载数据 (注意这里是 CIFAR10)
    train_dataset = CIFAR10(root='./data', train=True, transform=data_transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=data_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader