"""
猫狗图像二分类任务 - 改进版
解决过拟合问题，提升验证集准确率
改进点：
1. 数据增强（Data Augmentation）
2. Batch Normalization
3. L2正则化
4. 学习率调度
5. 早停机制
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置参数 ====================
DATA_DIR = './data'
USE_FLAT_STRUCTURE = False

# 训练参数（性能优化配置）
BATCH_SIZE = 64        # 增大batch size以提高GPU利用率
NUM_EPOCHS = 50        # 训练轮数
LEARNING_RATE = 0.001  # 学习率
VALIDATE_EVERY = 1     # 每N个epoch验证一次
TRAIN_RATIO = 0.8      # 训练集比例

# 数据加载参数（性能优化）
NUM_WORKERS = 4        # 数据加载进程数，提高CPU利用率
PIN_MEMORY = True      # 锁页内存，加速CPU到GPU传输
PREFETCH_FACTOR = 2    # 预取因子，每个worker预取的batch数

# 图像预处理参数
IMG_SIZE = 150  # 图像尺寸

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# GPU性能优化
if torch.cuda.is_available():
    # 启用cuDNN自动优化，加速卷积运算
    torch.backends.cudnn.benchmark = True
    # 启用cuDNN确定性模式（可选，如果需要可复现性）
    # torch.backends.cudnn.deterministic = True
    print("已启用cuDNN性能优化模式")

# ==================== 数据预处理（关键改进：数据增强）====================
# 训练集：使用数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),     # 先放大一点
    transforms.RandomCrop(IMG_SIZE),                        # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),                 # 随机水平翻转
    transforms.RandomRotation(15),                          # 随机旋转±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2,   # 颜色抖动
                          saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet标准化
                        std=[0.229, 0.224, 0.225])
])

# 验证集：不做数据增强，只做基本预处理
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ==================== 数据加载 ====================
def load_datasets():
    """加载数据集"""
    
    if USE_FLAT_STRUCTURE:
        print("使用 data/cat 和 data/dog 结构...")
        full_dataset = datasets.ImageFolder(root=DATA_DIR)
        
        train_size = int(TRAIN_RATIO * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # 分割后再应用不同的transform
        from torch.utils.data import Subset
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(full_dataset)))
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # 手动设置transform
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    else:
        print("使用 data/training_set 和 data/test_set 结构...")
        train_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, 'training_set'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(DATA_DIR, 'test_set'),
            transform=val_transform
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,           # 多进程数据加载
        pin_memory=PIN_MEMORY,             # 锁页内存加速传输
        prefetch_factor=PREFETCH_FACTOR,   # 预取数据
        persistent_workers=True            # 保持worker进程存活，减少启动开销
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")
    
    return train_loader, val_loader

# ==================== 改进的模型定义（添加Batch Normalization）====================
class CatDogCNN_Improved(nn.Module):
    """
    改进的3层CNN
    添加Batch Normalization层，提高训练稳定性，减少过拟合
    """
    def __init__(self):
        super(CatDogCNN_Improved, self).__init__()
        
        # 第一层卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三层卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 第四层卷积块（新增）
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 全连接层（减小神经元数量）
        self.fc1 = nn.Linear(256 * 9 * 9, 256)  # 减小全连接层
        self.fc2 = nn.Linear(256, 2)
        
        # 增加Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一层: Conv -> BN -> ReLU -> Pool
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # 第二层
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # 第三层
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # 第四层
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== 训练函数（支持混合精度）====================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """训练一个epoch（支持混合精度加速）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# ==================== 验证函数（支持混合精度）====================
def validate(model, val_loader, criterion, device, use_amp=True):
    """验证模型（支持混合精度加速）"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 混合精度推理
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# ==================== 绘图函数 ====================
def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线（两子图）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 子图1: 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 精度曲线
    ax2.plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'm-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics_improved.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存: training_metrics_improved.png")
    plt.show()

# ==================== 模型保存函数 ====================
def save_model(model, filepath):
    """保存模型"""
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存: {filepath}")

# ==================== 转换为ONNX ====================
def convert_to_onnx(model, onnx_path):
    """转换为ONNX格式"""
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX模型已保存: {onnx_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("猫狗分类 - 改进版训练（防过拟合）")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据集（使用数据增强）...")
    train_loader, val_loader = load_datasets()
    
    # 2. 创建模型
    print("\n[2/6] 创建改进模型（添加Batch Normalization）...")
    model = CatDogCNN_Improved().to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 定义损失函数和优化器（添加L2正则化）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # L2正则化
    
    # 混合精度训练：创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler is not None:
        print("已启用混合精度训练（AMP）")
    
    # 学习率调度器（验证损失不降时降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 4. 训练（添加早停机制）
    print("\n[3/6] 开始训练...")
    print("-" * 60)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10  # 早停耐心值
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练（使用混合精度）
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证（使用混合精度）
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, use_amp=True)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印进度
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% "
              f"- Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model_improved.pth')
            print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！连续{EARLY_STOP_PATIENCE}个epoch验证精度未提升")
            break
    
    print("-" * 60)
    print(f"训练完成！最佳验证精度: {best_val_acc:.2f}%")
    
    # 5. 保存最终模型
    print("\n[4/6] 保存最终模型...")
    save_model(model, 'latest_model_improved.pth')
    
    # 6. 绘制曲线
    print("\n[5/6] 绘制训练曲线...")
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    # 7. 转换ONNX
    print("\n[6/6] 转换ONNX格式...")
    best_model = CatDogCNN_Improved().to(DEVICE)
    best_model.load_state_dict(torch.load('best_model_improved.pth'))
    convert_to_onnx(best_model, 'best_model_improved.onnx')
    
    print("\n" + "=" * 60)
    print("训练完成！生成的文件：")
    print("  1. best_model_improved.pth    - 最佳模型")
    print("  2. latest_model_improved.pth  - 最终模型")
    print("  3. best_model_improved.onnx   - ONNX格式")
    print("  4. training_metrics_improved.png - 训练曲线")
    print("=" * 60)

if __name__ == '__main__':
    main()
