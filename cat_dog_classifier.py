"""
猫狗图像二分类任务
使用3层CNN卷积神经网络进行分类
作者：zk
环境：Python虚拟环境 zk
框架：PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置参数 ====================
# 数据集路径配置
DATA_DIR = './data'  # 数据集根目录
# 如果数据结构是 data/cat 和 data/dog，设置为True
# 如果数据结构是 data/training_set 和 data/test_set，设置为False
USE_FLAT_STRUCTURE = False

# 训练参数
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATE_EVERY = 1  # 每1个epoch验证一次
TRAIN_RATIO = 0.8   # 训练集比例（用于自动划分）

# 图像预处理参数
IMG_SIZE = 128      # 统一图像大小

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ==================== 数据预处理 ====================
# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像大小
    transforms.ToTensor(),                     # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5])   # 归一化到[-1, 1]
])

# ==================== 数据加载 ====================
def load_datasets():
    """加载数据集并划分训练集和验证集"""
    
    if USE_FLAT_STRUCTURE:
        # 方式1: data/cat 和 data/dog 结构，自动划分训练集和验证集
        print("使用 data/cat 和 data/dog 结构...")
        full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        
        # 自动划分训练集和验证集
        train_size = int(TRAIN_RATIO * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
        )
    else:
        # 方式2: data/training_set 和 data/test_set 结构
        print("使用 data/training_set 和 data/test_set 结构...")
        train_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, 'training_set'), 
            transform=transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, 'test_set'), 
            transform=transform
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,   # 训练集打乱
        num_workers=0   # Windows下设为0避免多进程问题
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # 验证集不打乱
        num_workers=0
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")
    
    return train_loader, val_loader

# ==================== 模型定义 ====================
class CatDogCNN(nn.Module):
    """
    3层CNN卷积神经网络
    结构: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC
    """
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # 第一层卷积块: 输入3通道(RGB) -> 32通道
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        
        # 第二层卷积块: 32通道 -> 64通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # 第三层卷积块: 64通道 -> 128通道
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 128通道 * 16x16特征图
        self.fc2 = nn.Linear(512, 2)              # 输出2个类别（猫、狗）
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一层: 卷积 -> ReLU -> 池化
        x = self.pool1(torch.relu(self.conv1(x)))
        
        # 第二层: 卷积 -> ReLU -> 池化
        x = self.pool2(torch.relu(self.conv2(x)))
        
        # 第三层: 卷积 -> ReLU -> 池化
        x = self.pool3(torch.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# ==================== 验证函数 ====================
def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# ==================== 绘图函数 ====================
def plot_metrics(train_losses, train_accs, val_accs):
    """
    在同一张图中绘制训练损失、训练精度和验证精度曲线
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制训练损失曲线
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    # 绘制训练精度曲线
    plt.plot(epochs, train_accs, 'g-', label='Training Accuracy', linewidth=2)
    
    # 绘制验证精度曲线
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    
    # 添加图例
    plt.legend(loc='upper right', fontsize=12)
    
    # 添加坐标轴标签
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    
    # 添加标题
    plt.title('Training Progress: Loss and Accuracy', fontsize=16)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("训练曲线图已保存为: training_metrics.png")
    
    # 显示图像
    plt.show()

# ==================== 模型保存函数 ====================
def save_model(model, filepath):
    """保存模型"""
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存: {filepath}")

# ==================== 转换为ONNX格式 ====================
def convert_to_onnx(model, onnx_path, input_size=(1, 3, 128, 128)):
    """
    将PyTorch模型转换为ONNX格式
    """
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(input_size).to(DEVICE)
    
    # 导出为ONNX格式
    torch.onnx.export(
        model,                          # 模型
        dummy_input,                    # 模型输入
        onnx_path,                      # 输出文件名
        export_params=True,             # 存储训练好的参数
        opset_version=11,               # ONNX算子版本
        do_constant_folding=True,       # 优化常量折叠
        input_names=['input'],          # 输入节点名称
        output_names=['output'],        # 输出节点名称
        dynamic_axes={                  # 动态batch维度
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型已保存: {onnx_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("猫狗图像二分类任务 - 开始训练")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n[1/6] 加载数据集...")
    train_loader, val_loader = load_datasets()
    
    # 2. 创建模型
    print("\n[2/6] 创建模型...")
    model = CatDogCNN().to(DEVICE)
    print(f"模型结构:\n{model}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. 训练模型
    print("\n[3/6] 开始训练...")
    print("-" * 60)
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 每5个epoch验证一次
        if epoch % VALIDATE_EVERY == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            val_accs.append(val_acc)
            
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                  f"- Train Loss: {train_loss:.4f} "
                  f"- Train Acc: {train_acc:.2f}% "
                  f"- Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, 'best_model.pth')
                print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
        else:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                  f"- Train Loss: {train_loss:.4f} "
                  f"- Train Acc: {train_acc:.2f}%")
    
    print("-" * 60)
    print(f"训练完成! 最佳验证精度: {best_val_acc:.2f}%")
    
    # 5. 保存最终模型
    print("\n[4/6] 保存模型...")
    save_model(model, 'latest_model.pth')
    
    # 6. 绘制训练曲线
    print("\n[5/6] 绘制训练曲线...")
    plot_metrics(train_losses, train_accs, val_accs)
    
    # 7. 转换最佳模型为ONNX格式
    print("\n[6/6] 转换ONNX格式...")
    
    # 加载最佳模型
    best_model = CatDogCNN().to(DEVICE)
    best_model.load_state_dict(torch.load('best_model.pth'))
    print("已加载最佳模型: best_model.pth")
    
    # 转换为ONNX
    convert_to_onnx(best_model, 'best_model.onnx')
    
    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. best_model.pth    - 验证集精度最高的模型")
    print("  2. latest_model.pth  - 最后一个epoch的模型")
    print("  3. best_model.onnx   - ONNX格式的最佳模型")
    print("  4. training_metrics.png - 训练曲线图")

if __name__ == '__main__':
    main()
