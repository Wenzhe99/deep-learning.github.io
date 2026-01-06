import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from model import LeNet5
from my_dataset import get_mnist_dataloader



def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train() # 切换到训练模式 (Switch to training mode)
    
    # 进度条 (Progress bar)
    loop = tqdm(train_loader, desc='Train')
    
    for images, labels in loop:
        # 将数据送到显卡/CPU (Move data to device)
        # (假设我们稍后会定义 device，现在先不管它)
        images, labels = images.to(device), labels.to(device)
        
        # -------------------------------------------
        # 请在这里写出那 5 个关键步骤 (Write the 5 key steps here)
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
        # -------------------------------------------
        
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

if __name__ == "__main__":
    # 1. 实例化模型 (Instantiate Model)
    model = LeNet5()
    # 2. 定义损失函数 (Define Loss Function)
    criterion = nn.CrossEntropyLoss()# Hint: nn.CrossEntropyLoss()
    # 3. 定义优化器 (Define Optimizer)
    # 注意：你需要告诉优化器我们要更新哪些参数 (model.parameters())
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    # 4. 准备数据加载器 (Prepare DataLoader)
    train_loader, test_loader = get_mnist_dataloader(batch_size=32, train=True)
    # 5. define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 6. 开始训练 (Start Training)
    # 设定总轮数 (Set total epochs)
    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # 1. 运行训练 (Run Training)
        # Hint: train_one_epoch(...)
        # 注意参数顺序 (Check parameter order): model, train_loader, optimizer, criterion, device
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        # 2. 运行评估 (Run Evaluation)
        # Hint: evaluate(...)
        accuracy = evaluate(model, test_loader, device)
        
        print(f"Test Accuracy: {accuracy:.2f}%")

    # 保存模型 (Save Model)
    # torch.save(model.state_dict(), "lenet5_mnist.pth")
    # print("Model saved to lenet5_mnist.pth")