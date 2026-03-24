"""
猫狗图像二分类任务 - ResNet18预训练模型版本
使用PyTorch预训练的ResNet18进行迁移学习
特点：
1. 预训练模型迁移学习
2. 实时显示训练速度（samples/sec）
3. 混合精度训练加速
4. 数据增强防过拟合
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
DATA_DIR = './data'
USE_FLAT_STRUCTURE = False

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 30  # 迁移学习通常需要更少epoch
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8

# 数据加载参数
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# 图像预处理参数（ResNet标准输入尺寸）
IMG_SIZE = 224  # ResNet标准输入

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# GPU性能优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("已启用cuDNN性能优化模式")

# ==================== 数据预处理 ====================
# 训练集：数据增强
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # 先放大
    transforms.RandomCrop(IMG_SIZE),            # 随机裁剪到224x224
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.RandomRotation(15),              # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                        std=[0.229, 0.224, 0.225])
])

# 验证集：基本预处理
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),            # 中心裁剪
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
        
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, len(full_dataset))))
        
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
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
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

# ==================== 模型定义 ====================
class ResNet18Classifier(nn.Module):
    """
    基于预训练ResNet18的猫狗分类器
    迁移学习策略：冻结特征提取层，只训练分类头
    """
    def __init__(self, num_classes=2, freeze_features=True):
        super(ResNet18Classifier, self).__init__()
        
        # 加载预训练ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # 冻结特征提取层（可选）
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换最后的全连接层
        num_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==================== 训练函数（含速度统计）====================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """
    训练一个epoch
    返回: (损失, 精度, 平均速度samples/sec)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 计时器
    data_start = time.time()
    batch_times = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        
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
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 记录batch处理时间
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # 计算平均速度
    total_time = time.time() - data_start
    avg_speed = total / total_time  # samples/sec
    
    # 计算batch平均处理时间
    avg_batch_time = sum(batch_times) / len(batch_times) * 1000  # ms
    
    return epoch_loss, epoch_acc, avg_speed, avg_batch_time

# ==================== 验证函数（含速度统计）====================
def validate(model, val_loader, criterion, device, use_amp=True):
    """
    验证模型
    返回: (损失, 精度, 平均速度samples/sec)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    val_start = time.time()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
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
    
    # 计算速度
    total_time = time.time() - val_start
    avg_speed = total / total_time  # samples/sec
    
    return epoch_loss, epoch_acc, avg_speed

# ==================== 绘图函数 ====================
def plot_metrics(train_losses, train_accs, val_losses, val_accs, train_speeds, val_speeds):
    """绘制训练曲线（三子图）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 子图1: 损失曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 精度曲线
    axes[1].plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'm-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 训练速度曲线
    axes[2].plot(epochs, train_speeds, 'c-', label='Train Speed', linewidth=2, marker='o')
    axes[2].plot(epochs, val_speeds, 'orange', label='Val Speed', linewidth=2, marker='s')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Speed (samples/sec)', fontsize=12)
    axes[2].set_title('Training and Validation Speed', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics_resnet18.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存: training_metrics_resnet18.png")
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
    print("=" * 70)
    print("猫狗分类 - ResNet18预训练模型（迁移学习）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/7] 加载数据集（使用数据增强）...")
    train_loader, val_loader = load_datasets()
    
    # 2. 创建模型
    print("\n[2/7] 创建ResNet18模型（预训练权重）...")
    
    # 第一阶段：冻结特征提取层，只训练分类头
    model = ResNet18Classifier(num_classes=2, freeze_features=True).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler is not None:
        print("已启用混合精度训练（AMP）")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 4. 第一阶段训练（冻结特征层）
    print("\n[3/7] 开始训练（冻结特征提取层）...")
    print("-" * 70)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_speeds, val_speeds = [], []
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5
    FREEZE_EPOCHS = 10  # 先训练10个epoch
    
    for epoch in range(1, FREEZE_EPOCHS + 1):
        train_loss, train_acc, train_speed, batch_time = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_speeds.append(train_speed)
        
        val_loss, val_acc, val_speed = validate(
            model, val_loader, criterion, DEVICE, use_amp=True
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_speeds.append(val_speed)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch}/{FREEZE_EPOCHS}] "
              f"- Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"- Acc: {train_acc:.2f}%/{val_acc:.2f}% "
              f"- Speed: {train_speed:.1f}/{val_speed:.1f} samples/sec "
              f"- Batch: {batch_time:.1f}ms")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model_resnet18_stage1.pth')
            print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
    
    # 5. 第二阶段训练（解冻所有层，微调）
    print("\n[4/7] 解冻所有层，开始微调...")
    print("-" * 70)
    
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
    
    # 使用更小的学习率微调
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE/10, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable_params:,}")
    
    FINETUNE_EPOCHS = NUM_EPOCHS - FREEZE_EPOCHS
    
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_loss, train_acc, train_speed, batch_time = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_speeds.append(train_speed)
        
        val_loss, val_acc, val_speed = validate(
            model, val_loader, criterion, DEVICE, use_amp=True
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_speeds.append(val_speed)
        
        scheduler.step(val_loss)
        
        actual_epoch = FREEZE_EPOCHS + epoch
        print(f"Epoch [{actual_epoch}/{NUM_EPOCHS}] "
              f"- Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"- Acc: {train_acc:.2f}%/{val_acc:.2f}% "
              f"- Speed: {train_speed:.1f}/{val_speed:.1f} samples/sec "
              f"- Batch: {batch_time:.1f}ms")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model_resnet18.pth')
            print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！")
            break
    
    print("-" * 70)
    print(f"训练完成！最佳验证精度: {best_val_acc:.2f}%")
    
    # 6. 保存最终模型
    print("\n[5/7] 保存最终模型...")
    save_model(model, 'latest_model_resnet18.pth')
    
    # 7. 绘制曲线
    print("\n[6/7] 绘制训练曲线...")
    plot_metrics(train_losses, train_accs, val_losses, val_accs, 
                train_speeds, val_speeds)
    
    # 8. 转换ONNX
    print("\n[7/7] 转换ONNX格式...")
    
    # 加载最佳模型
    best_model = ResNet18Classifier(num_classes=2, freeze_features=False).to(DEVICE)
    best_model.load_state_dict(torch.load('best_model_resnet18.pth'))
    convert_to_onnx(best_model, 'best_model_resnet18.onnx')
    
    # 打印训练总结
    print("\n" + "=" * 70)
    print("训练总结:")
    print(f"  最佳验证精度: {best_val_acc:.2f}%")
    print(f"  平均训练速度: {sum(train_speeds)/len(train_speeds):.1f} samples/sec")
    print(f"  平均验证速度: {sum(val_speeds)/len(val_speeds):.1f} samples/sec")
    print("\n生成的文件:")
    print("  1. best_model_resnet18.pth    - 最佳模型")
    print("  2. latest_model_resnet18.pth  - 最终模型")
    print("  3. best_model_resnet18.onnx   - ONNX格式")
    print("  4. training_metrics_resnet18.png - 训练曲线（含速度）")
    print("=" * 70)

if __name__ == '__main__':
    main()
