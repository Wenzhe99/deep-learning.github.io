import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import ResNet18

# 1. 设置配置
MODEL_PATH = "../../../results/resnet18.pth"  # 确保路径正确
IMAGE_PATH = "../../../testdata/dog.jpg"             # 你的测试图片
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 定义 Grad-CAM 工具类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子 (Hooks)
        # 向前传播时：记录特征图
        self.target_layer.register_forward_hook(self.save_activation)
        # 向后传播时：记录梯度
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output 是一个 tuple，取第一项
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # A. 前向传播
        output = self.model(x)
        
        if class_idx is None:
            # 如果没指定看哪一类，就看概率最大的那一类
            class_idx = torch.argmax(output, dim=1).item()

        # B. 反向传播 (计算梯度)
        self.model.zero_grad()
        # 这里的 1.0 是反向传播的起始梯度
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)

        # C. 生成 CAM
        # 1. 对梯度求全局平均 (Global Average Pooling) -> 得到每个通道的权重
        # gradients shape: [1, 512, 4, 4] -> weights shape: [1, 512, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 2. 权重 * 特征图
        # activations shape: [1, 512, 4, 4]
        # cam shape: [1, 512, 4, 4]
        cam = self.activations * weights

        # 3. 对所有通道求和 -> 压缩成一张图 [1, 4, 4]
        cam = torch.sum(cam, dim=1).squeeze()

        # 4. ReLU (只保留正向激活，负数说明抑制，不重要)
        cam = F.relu(cam)

        # 5. 归一化到 0-1 之间，方便画图
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7) # 加个小树防止除零

        return cam.data.cpu().numpy(), class_idx

# 3. 图像融合与显示
def show_cam_on_image(img_path, mask):
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255

    # 将 mask (4x4) 放大到原图大小
    heatmap = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # 上色 (将灰度图变成热力图)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # 叠加: 原图 * 0.5 + 热力图 * 0.5
    cam_img = heatmap * 0.5 + img * 0.5
    cam_img = cam_img / np.max(cam_img)

    # 画图
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cam_img)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.show()

# 4. 主程序
if __name__ == '__main__':
    # 加载模型
    model = ResNet18().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ 模型加载成功")
    except:
        print("❌ 模型加载失败，请检查路径")
        exit()
    
    model.eval()

    # 准备图片输入 (32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    raw_img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    # 🔥 核心：初始化 GradCAM
    # 我们要看 ResNet 的最后一层卷积层：layer4
    # layer4 是最后一个 Residual Block，我们取它的最后一层
    target_layer = model.layer4[-1] 
    
    grad_cam = GradCAM(model, target_layer)

    # 生成热力图
    print(f"🤖 正在分析图片: {IMAGE_PATH} ...")
    mask, class_idx = grad_cam(input_tensor)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"🔍 模型关注点分析完成。预测类别: {classes[class_idx]}")

    # 显示结果
    show_cam_on_image(IMAGE_PATH, mask)