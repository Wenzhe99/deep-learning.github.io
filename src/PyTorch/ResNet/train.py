import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from model import ResNet18
from dataset import get_cifar_dataloader

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train() # 切换到训练模式 (Switch to training mode)
    
    # 进度条 (Progress bar)
    loop = tqdm(train_loader, desc='Train')
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Zero  gradients
        optimizer.zero_grad()
        # 2. Forward pass
        outputs = model(images)
        # 3. Compute Loss
        loss = criterion(outputs, labels)
        # 4. Backward pass
        loss.backward()
        #5. Update weights
        optimizer.step()
        
        # 打印当前 loss (Optional: update progress bar)
        loop.set_postfix(loss=loss.item())

def evaluate(model, test_loader, device):
    model.eval() # 切换到评估模式 (Switch to evaluation mode)
    
    correct = 0
    total = 0
    
    # 1.在这里填入禁止计算梯度的上下文管理器 (Fill in the context manager to disable gradient calculation here)
    # Hint: with torch. ? :
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # 2. 获取预测的类别 (Get the predicted class)
            # outputs shape: [Batch_Size, 10]
            # 我们需要找到每一行最大值的索引 (We need the index of the max value in each row)
            # Hint: torch.max() or .argmax()
            _, predicted = torch.max(outputs, 1) 
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    return acc

if __name__ == '__main__':
    # 超参数 (Hyperparameters)
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 16
    
    # 设备配置 (Device configuration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器 (Get data loaders)
    train_loader, test_loader = get_cifar_dataloader(batch_size)
    
    # 初始化模型、损失函数和优化器 (Initialize model, loss function, and optimizer)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练和评估循环 (Training and evaluation loop)
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f'Accuracy on test set: {acc:.2f}%\n')
