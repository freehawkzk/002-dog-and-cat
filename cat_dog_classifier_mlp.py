"""
猫狗图像二分类任务 - 多层感知机（MLP）版本
使用3个隐含层的MLP模型
输入：128x128x3 的图像（展平为49152维向量）
特点：
1. 纯全连接网络
2. Dropout防止过拟合
3. Batch Normalization
4. 实时显示训练速度
5. 混合精度训练
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# ==================== 配置参数 ====================
DATA_DIR = './data'
USE_FLAT_STRUCTURE = False

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001  # MLP通常需要较小的学习率
TRAIN_RATIO = 0.8

# 数据加载参数
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# 图像预处理参数
IMG_SIZE = 128  # MLP输入尺寸128x128
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3  # 49152

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
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 验证集：基本预处理
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

# ==================== MLP模型定义 ====================
class CatDogMLP(nn.Module):
    """
    多层感知机模型
    结构：
    - 输入层：49152 (128x128x3)
    - 隐藏层1：2048神经元 + BatchNorm + Dropout
    - 隐藏层2：1024神经元 + BatchNorm + Dropout
    - 隐藏层3：512神经元 + BatchNorm + Dropout
    - 输出层：2（猫、狗）
    """
    def __init__(self, input_dim=49152, num_classes=2):
        super(CatDogMLP, self).__init__()
        
        # 输入层到第一个隐藏层
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 第二个隐藏层
        self.layer2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # 第三个隐藏层
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 输出层
        self.output = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 展平图像: (batch, 3, 128, 128) -> (batch, 49152)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        
        return x

# ==================== 训练函数（含速度统计）====================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, epoch=0, writer=None):
    """训练一个epoch（支持TensorBoard记录）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
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
        
        # TensorBoard记录（每100个batch记录一次）
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # 计算速度
    total_time = time.time() - epoch_start
    avg_speed = total / total_time  # samples/sec
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    return epoch_loss, epoch_acc, avg_speed, current_lr

# ==================== 验证函数（含速度统计）====================
def validate(model, val_loader, criterion, device, use_amp=True):
    """验证模型"""
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
    avg_speed = total / total_time
    
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
    axes[2].plot(epochs, train_speeds, 'c-', label='Train Speed', linewidth=2, marker='o', markersize=3)
    axes[2].plot(epochs, val_speeds, 'orange', label='Val Speed', linewidth=2, marker='s', markersize=3)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Speed (samples/sec)', fontsize=12)
    axes[2].set_title('Training and Validation Speed', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics_mlp.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存: training_metrics_mlp.png")
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
    print("猫狗分类 - 多层感知机（MLP）模型")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据集...")
    train_loader, val_loader = load_datasets()
    
    # 2. 创建模型
    print("\n[2/6] 创建MLP模型...")
    model = CatDogMLP(input_dim=INPUT_DIM, num_classes=2).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"输入维度: {INPUT_DIM} (128x128x3)")
    print(f"模型结构:")
    print(f"  输入层 -> 隐藏层1(2048) -> 隐藏层2(1024) -> 隐藏层3(512) -> 输出层(2)")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler is not None:
        print("已启用混合精度训练（AMP）")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 创建TensorBoard writer
    log_dir = os.path.join('runs', 'mlp_experiment')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    print("查看训练过程: tensorboard --logdir=runs")
    
    # 4. 训练
    print("\n[3/6] 开始训练...")
    print("-" * 70)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_speeds, val_speeds = [], []
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss, train_acc, train_speed, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, epoch, writer
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_speeds.append(train_speed)
        
        # 验证
        val_loss, val_acc, val_speed = validate(
            model, val_loader, criterion, DEVICE, use_amp=True
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_speeds.append(val_speed)
        
        scheduler.step(val_loss)
        
        # TensorBoard记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Speed/Train', train_speed, epoch)
        writer.add_scalar('Speed/Validation', val_speed, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 记录模型参数分布（每5个epoch记录一次）
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
        
        # 打印进度
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"- Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"- Acc: {train_acc:.2f}%/{val_acc:.2f}% "
              f"- Speed: {train_speed:.1f}/{val_speed:.1f} samples/sec "
              f"- LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model_mlp.pth')
            print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发！连续{EARLY_STOP_PATIENCE}个epoch验证精度未提升")
            break
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("-" * 70)
    print(f"训练完成！最佳验证精度: {best_val_acc:.2f}%")
    
    # 5. 保存最终模型
    print("\n[4/6] 保存最终模型...")
    save_model(model, 'latest_model_mlp.pth')
    
    # 6. 绘制曲线
    print("\n[5/6] 绘制训练曲线...")
    plot_metrics(train_losses, train_accs, val_losses, val_accs, 
                train_speeds, val_speeds)
    
    # 7. 转换ONNX
    print("\n[6/6] 转换ONNX格式...")
    
    # 加载最佳模型
    best_model = CatDogMLP(input_dim=INPUT_DIM, num_classes=2).to(DEVICE)
    best_model.load_state_dict(torch.load('best_model_mlp.pth'))
    convert_to_onnx(best_model, 'best_model_mlp.onnx')
    
    # 打印训练总结
    print("\n" + "=" * 70)
    print("训练总结:")
    print(f"  最佳验证精度: {best_val_acc:.2f}%")
    print(f"  平均训练速度: {sum(train_speeds)/len(train_speeds):.1f} samples/sec")
    print(f"  平均验证速度: {sum(val_speeds)/len(val_speeds):.1f} samples/sec")
    print(f"  总参数量: {total_params:,}")
    print("\n生成的文件:")
    print("  1. best_model_mlp.pth    - 最佳模型")
    print("  2. latest_model_mlp.pth  - 最终模型")
    print("  3. best_model_mlp.onnx   - ONNX格式")
    print("  4. training_metrics_mlp.png - 训练曲线")
    print("  5. runs/mlp_experiment/   - TensorBoard日志")
    print("\n查看TensorBoard:")
    print("  运行命令: tensorboard --logdir=runs")
    print("  浏览器访问: http://localhost:6006")
    print("=" * 70)

if __name__ == '__main__':
    main()
