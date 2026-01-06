import sys, os
sys.path.insert(0, os.getcwd())

from torchvision.datasets import MNIST
import PIL
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=64, train=True):
    # 定义数据转换
    data_transform = transforms.Compose([
        # 调整大小
        transforms.Resize((32, 32),),
        # 转为张量
        transforms.ToTensor(),
        # 归一化
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 1. 准备数据集 (Prepare Datasets)
    train_dataset = MNIST(root='././././data', train=True, transform=data_transform, download=True)
    test_dataset = MNIST(root='././././data', train=False, transform=data_transform, download=True)

    # 2. 准备数据加载器 (Prepare DataLoaders)
    # 设定 Batch Size
    BATCH_SIZE = 32

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 检查一下数据 (Check the data)
    print("Train set size:", len(train_dataset))
    print("Test set size:", len(test_dataset))

    return train_loader, test_loader