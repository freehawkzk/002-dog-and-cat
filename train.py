"""
猫狗分类统一训练脚本
支持三种模型：MLP、CNN、ResNet18
使用方法：
    python train.py --model mlp
    python train.py --model cnn
    python train.py --model resnet18
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from model import get_model, get_model_info


# ==================== 配置管理 ====================
class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, model_name, args=None):
        self.model_name = model_name.lower()
        model_info = get_model_info(self.model_name)
        
        # 基础配置
        self.data_dir = args.data_dir if args and args.data_dir else './data'
        self.use_flat_structure = False
        
        # 模型相关配置
        self.input_size = model_info['input_size']
        self.learning_rate = args.learning_rate if args and args.learning_rate else model_info['learning_rate']
        
        # 训练配置 - 默认值
        default_batch_size = 64 if self.model_name == 'resnet18' else 128
        default_epochs = 30 if self.model_name == 'resnet18' else 50
        
        self.batch_size = args.batch_size if args and args.batch_size is not None else default_batch_size
        self.num_epochs = args.epochs if args and args.epochs is not None else default_epochs
        self.train_ratio = 0.8
        
        # 数据加载配置 - 默认值
        self.num_workers = args.num_workers if args and args.num_workers is not None else 8
        self.pin_memory = True
        self.prefetch_factor = args.prefetch_factor if args and args.prefetch_factor is not None else 4
        self.persistent_workers = True
        
        # 早停配置
        self.early_stop_patience = args.patience if args and args.patience is not None else 10
        
        # ResNet18特殊配置
        if self.model_name == 'resnet18':
            self.freeze_epochs = args.freeze_epochs if args and args.freeze_epochs is not None else 10
        
        # 设备配置
        device_str = args.device if args and args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA不可用，将使用CPU训练")
            device_str = 'cpu'
        self.device = torch.device(device_str)
        
        # 输出目录 - 保存到 checkpoints 文件夹
        self.output_dir = os.path.join('checkpoints', self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard日志目录
        self.log_dir = os.path.join('runs', f'{self.model_name}_experiment')


# ==================== 数据预处理 ====================
def get_transforms(config):
    """获取数据预处理transforms"""
    
    input_size = config.input_size
    
    # 训练集：数据增强
    train_transform = transforms.Compose([
        transforms.Resize((input_size + 20, input_size + 20)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集：基本预处理
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# ==================== 数据加载 ====================
def load_datasets(config):
    """加载数据集"""
    
    train_transform, val_transform = get_transforms(config)
    
    if config.use_flat_structure:
        print("使用 data/cat 和 data/dog 结构...")
        full_dataset = datasets.ImageFolder(root=config.data_dir)
        
        train_size = int(config.train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, list(range(train_size)))
        val_dataset = Subset(full_dataset, list(range(train_size, len(full_dataset))))
        
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
    else:
        print("使用 data/training_set 和 data/test_set 结构...")
        train_dataset = datasets.ImageFolder(
            root=os.path.join(config.data_dir, 'training_set'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(config.data_dir, 'test_set'),
            transform=val_transform
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True  # 丢弃不完整的batch，提高训练效率
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")
    
    return train_loader, val_loader, val_dataset


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, epoch=0, writer=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    
    # 预取第一个batch
    data_iter = iter(train_loader)
    try:
        images, labels = next(data_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    except StopIteration:
        return 0, 0, 0, 0
    
    for batch_idx in range(len(train_loader)):
        # 预取下一个batch（异步传输）
        try:
            next_images, next_labels = next(data_iter)
            next_images = next_images.to(device, non_blocking=True)
            next_labels = next_labels.to(device, non_blocking=True)
        except StopIteration:
            next_images, next_labels = None, None
        
        # 混合精度训练
        if scaler is not None:
            with torch.amp.autocast('cuda'):
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
        
        # 更新到下一个batch
        if next_images is not None:
            images, labels = next_images, next_labels
        
        # TensorBoard记录（每100个batch）
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    avg_speed = total / (time.time() - epoch_start)
    current_lr = optimizer.param_groups[0]['lr']
    
    return epoch_loss, epoch_acc, avg_speed, current_lr


# ==================== 验证函数 ====================
def validate(model, val_loader, criterion, device, use_amp=True):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    val_start = time.time()
    
    with torch.no_grad():
        for images, labels in val_loader:
            # 使用 non_blocking=True 异步传输
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
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
    avg_speed = total / (time.time() - val_start)
    
    return epoch_loss, epoch_acc, avg_speed


# ==================== 绘图函数 ====================
def plot_metrics(train_losses, train_accs, val_losses, val_accs, train_speeds, val_speeds, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 子图1: 损失曲线
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 精度曲线
    axes[1].plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'm-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: 速度曲线
    axes[2].plot(epochs, train_speeds, 'c-', label='Train Speed', linewidth=2, marker='o', markersize=3)
    axes[2].plot(epochs, val_speeds, 'orange', label='Val Speed', linewidth=2, marker='s', markersize=3)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Speed (samples/sec)', fontsize=12)
    axes[2].set_title('Speed', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")
    plt.close()


# ==================== 混淆矩阵 ====================
def plot_confusion_matrix(model, val_loader, device, class_names, save_path):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算各项指标
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Confusion Matrix\nAcc: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%', 
              fontsize=14)
    
    # 在格子中添加百分比
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存: {save_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


# ==================== 预测样本可视化 ====================
def plot_prediction_samples(model, dataset, device, class_names, save_path, num_samples=5):
    """
    绘制预测样本可视化
    - 每类随机选择 num_samples 张预测正确的样本
    - 每类随机选择 num_samples 张预测错误的样本
    """
    import random
    from torchvision.transforms.functional import to_pil_image
    
    model.eval()
    
    # 收集预测结果
    correct_samples = {i: [] for i in range(len(class_names))}  # 每类正确的样本
    wrong_samples = {i: [] for i in range(len(class_names))}    # 每类错误的样本
    
    # 反标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, label = dataset[idx]
            image_tensor = image_tensor.to(device).unsqueeze(0)
            
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            
            # 反标准化图像用于显示
            img_display = image_tensor.squeeze(0).cpu() * std + mean
            img_display = torch.clamp(img_display, 0, 1)
            img_pil = to_pil_image(img_display)
            
            sample_info = {
                'image': img_pil,
                'true_label': label,
                'pred_label': predicted,
                'true_name': class_names[label],
                'pred_name': class_names[predicted]
            }
            
            if predicted == label:
                correct_samples[label].append(sample_info)
            else:
                wrong_samples[label].append(sample_info)
    
    # 随机选择样本
    selected_correct = {}
    selected_wrong = {}
    
    for class_idx in range(len(class_names)):
        # 随机选择正确样本
        if len(correct_samples[class_idx]) >= num_samples:
            selected_correct[class_idx] = random.sample(correct_samples[class_idx], num_samples)
        else:
            selected_correct[class_idx] = correct_samples[class_idx]
        
        # 随机选择错误样本
        if len(wrong_samples[class_idx]) >= num_samples:
            selected_wrong[class_idx] = random.sample(wrong_samples[class_idx], num_samples)
        else:
            selected_wrong[class_idx] = wrong_samples[class_idx]
    
    # 绘制图像
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes * 2, num_samples, figsize=(num_samples * 3, num_classes * 6))
    
    if num_samples == 1:
        axes = axes.reshape(num_classes * 2, 1)
    
    for class_idx in range(num_classes):
        # 绘制正确样本（上排）
        for j in range(num_samples):
            ax = axes[class_idx * 2, j]
            if j < len(selected_correct[class_idx]):
                sample = selected_correct[class_idx][j]
                ax.imshow(sample['image'])
                ax.set_title(f"Correct: {sample['true_name']}", fontsize=10, color='green')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # 绘制错误样本（下排）
        for j in range(num_samples):
            ax = axes[class_idx * 2 + 1, j]
            if j < len(selected_wrong[class_idx]):
                sample = selected_wrong[class_idx][j]
                ax.imshow(sample['image'])
                ax.set_title(f"True: {sample['true_name']}\nPred: {sample['pred_name']}", 
                           fontsize=10, color='red')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # 添加行标签
    for class_idx in range(num_classes):
        axes[class_idx * 2, 0].text(-0.1, 0.5, f'{class_names[class_idx]}\nCorrect', 
                                    transform=axes[class_idx * 2, 0].transAxes,
                                    ha='center', va='center', fontsize=12, rotation=90, color='green')
        axes[class_idx * 2 + 1, 0].text(-0.1, 0.5, f'{class_names[class_idx]}\nWrong', 
                                        transform=axes[class_idx * 2 + 1, 0].transAxes,
                                        ha='center', va='center', fontsize=12, rotation=90, color='red')
    
    plt.suptitle('Prediction Samples: Correct (Green) vs Wrong (Red)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"预测样本可视化已保存: {save_path}")
    plt.close()
    
    # 打印统计信息
    print(f"\n预测样本统计:")
    for class_idx in range(num_classes):
        correct_count = len(correct_samples[class_idx])
        wrong_count = len(wrong_samples[class_idx])
        total = correct_count + wrong_count
        print(f"  {class_names[class_idx]}: 正确 {correct_count}/{total}, 错误 {wrong_count}/{total}")


# ==================== 主训练流程 ====================
def train(model_name, args=None):
    """主训练函数"""
    
    # 初始化配置
    config = TrainingConfig(model_name, args)
    
    print("=" * 70)
    print(f"猫狗分类训练 - {model_name.upper()}模型")
    print("=" * 70)
    
    # GPU优化设置
    if torch.cuda.is_available():
        # 启用cuDNN自动优化
        torch.backends.cudnn.benchmark = True
        # 启用TF32加速（Ampere架构GPU）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 设置CUDA设备
        torch.cuda.set_device(0)
        # 清空缓存
        torch.cuda.empty_cache()
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Batch Size: {config.batch_size}")
        print(f"Num Workers: {config.num_workers}")
        print("已启用GPU优化: cuDNN benchmark, TF32, 异步数据传输")
    
    # 加载数据
    print(f"\n[1/8] 加载数据集（输入尺寸: {config.input_size}x{config.input_size}）...")
    train_loader, val_loader, val_dataset = load_datasets(config)
    
    # 创建模型
    print(f"\n[2/8] 创建模型...")
    if config.model_name == 'mlp':
        model = get_model('mlp', input_dim=config.input_size * config.input_size * 3)
    elif config.model_name == 'cnn':
        model = get_model('cnn', img_size=config.input_size)
    else:
        model = get_model('resnet18', freeze_features=True)
    
    model = model.to(config.device)
    
    # 检查是否存在已保存的最佳模型
    best_model_path = os.path.join(config.output_dir, 'best_model.pth')
    checkpoint_path = os.path.join(config.output_dir, 'checkpoint.pth')
    resume_training = False
    start_epoch = 1
    checkpoint_data = None
    
    if os.path.exists(checkpoint_path):
        print(f"\n发现训练检查点: {checkpoint_path}")
        print("加载检查点继续训练...")
        checkpoint_data = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        resume_training = True
        start_epoch = checkpoint_data.get('epoch', 1) + 1
        print(f"从第 {start_epoch} 个epoch继续训练")
    elif os.path.exists(best_model_path):
        print(f"\n发现已保存的模型: {best_model_path}")
        print("加载已保存模型继续训练...")
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        resume_training = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=config.learning_rate, weight_decay=1e-4)
    
    # 混合精度
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    if scaler:
        print("已启用混合精度训练")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 如果存在checkpoint，加载优化器和调度器状态
    if checkpoint_data is not None:
        if 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint_data:
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
    
    # TensorBoard
    writer = SummaryWriter(config.log_dir)
    print(f"TensorBoard日志: {config.log_dir}")
    
    # 训练
    print(f"\n[3/8] 开始训练...")
    print("-" * 70)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_speeds, val_speeds = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    # 如果存在checkpoint，加载训练状态
    if checkpoint_data is not None:
        best_val_acc = checkpoint_data.get('best_val_acc', 0.0)
        patience_counter = checkpoint_data.get('patience_counter', 0)
        train_losses = checkpoint_data.get('train_losses', [])
        train_accs = checkpoint_data.get('train_accs', [])
        val_losses = checkpoint_data.get('val_losses', [])
        val_accs = checkpoint_data.get('val_accs', [])
        train_speeds = checkpoint_data.get('train_speeds', [])
        val_speeds = checkpoint_data.get('val_speeds', [])
        print(f"恢复训练状态: 最佳精度 {best_val_acc:.2f}%")
    
    # ResNet18两阶段训练
    if config.model_name == 'resnet18':
        # 判断当前应该处于哪个阶段
        if start_epoch <= config.freeze_epochs:
            print("阶段1: 冻结特征层训练...")
        else:
            print("阶段2: 解冻所有层微调...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate / 10, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    for epoch in range(start_epoch, config.num_epochs + 1):
        # ResNet18第二阶段：解冻微调
        if config.model_name == 'resnet18' and epoch == config.freeze_epochs + 1:
            print("\n阶段2: 解冻所有层微调...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate / 10, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 训练
        train_loss, train_acc, train_speed, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, config.device, scaler, epoch, writer
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_speeds.append(train_speed)
        
        # 验证
        val_loss, val_acc, val_speed = validate(
            model, val_loader, criterion, config.device, use_amp=True
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
        
        # 打印进度
        print(f"Epoch [{epoch}/{config.num_epochs}] "
              f"- Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"- Acc: {train_acc:.2f}%/{val_acc:.2f}% "
              f"- Speed: {train_speed:.1f}/{val_speed:.1f} "
              f"- LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))
            print(f"  >> 新的最佳验证精度: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 保存检查点（每个epoch都保存）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'patience_counter': patience_counter,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'train_speeds': train_speeds,
            'val_speeds': val_speeds,
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, os.path.join(config.output_dir, 'checkpoint.pth'))
        
        # 早停
        if patience_counter >= config.early_stop_patience:
            print(f"\n早停触发！连续{config.early_stop_patience}个epoch无提升")
            break
    
    writer.close()
    print("-" * 70)
    print(f"训练完成！最佳验证精度: {best_val_acc:.2f}%")
    
    # 保存最终模型
    print(f"\n[4/8] 保存最终模型...")
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'latest_model.pth'))
    
    # 绘制曲线
    print(f"\n[5/8] 绘制训练曲线...")
    plot_metrics(train_losses, train_accs, val_losses, val_accs, 
                train_speeds, val_speeds, 
                os.path.join(config.output_dir, 'training_metrics.png'))
    
    # 绘制混淆矩阵（使用最佳模型）
    print(f"\n[6/8] 绘制混淆矩阵...")
    # 加载最佳模型
    if config.model_name == 'mlp':
        best_model = get_model('mlp', input_dim=config.input_size * config.input_size * 3)
    elif config.model_name == 'cnn':
        best_model = get_model('cnn', img_size=config.input_size)
    else:
        best_model = get_model('resnet18', freeze_features=False)
    
    best_model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pth')))
    best_model = best_model.to(config.device)
    
    # 获取类别名称
    class_names = ['cat', 'dog']
    
    cm_metrics = plot_confusion_matrix(
        best_model, val_loader, config.device, class_names,
        os.path.join(config.output_dir, 'confusion_matrix.png')
    )
    
    # 绘制预测样本可视化
    print(f"\n[7/8] 绘制预测样本可视化...")
    plot_prediction_samples(
        best_model, val_dataset, config.device, class_names,
        os.path.join(config.output_dir, 'prediction_samples.png'),
        num_samples=5
    )
    
    # 转换ONNX
    print(f"\n[8/8] 转换ONNX格式...")
    model.eval()
    dummy_input = torch.randn(1, 3, config.input_size, config.input_size).to(config.device)
    torch.onnx.export(
        model, dummy_input, os.path.join(config.output_dir, 'best_model.onnx'),
        export_params=True, opset_version=11,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 总结
    print("\n" + "=" * 70)
    print("训练总结:")
    print(f"  最佳验证精度: {best_val_acc:.2f}%")
    print(f"  混淆矩阵指标:")
    print(f"    - Precision: {cm_metrics['precision']:.2f}%")
    print(f"    - Recall: {cm_metrics['recall']:.2f}%")
    print(f"    - F1 Score: {cm_metrics['f1']:.2f}")
    print(f"  平均训练速度: {sum(train_speeds)/len(train_speeds):.1f} samples/sec")
    print(f"\n输出目录: {config.output_dir}")
    print("  - best_model.pth")
    print("  - latest_model.pth")
    print("  - best_model.onnx")
    print("  - training_metrics.png")
    print("  - confusion_matrix.png")
    print("  - prediction_samples.png")
    print(f"\n查看TensorBoard: tensorboard --logdir=runs")
    print("=" * 70)


# ==================== 命令行入口 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='猫狗分类训练脚本')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mlp', 'cnn', 'resnet18'],
                       help='选择模型: mlp, cnn, resnet18')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认: MLP/CNN=50, ResNet18=30)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小 (默认: MLP/CNN=128, ResNet18=64)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='学习率 (默认: MLP=0.001, CNN=0.001, ResNet18=0.0001)')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值 (默认: 10)')
    
    # 数据加载参数
    parser.add_argument('--num-workers', type=int, default=8,
                       help='数据加载线程数 (默认: 8)')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                       help='数据预取因子 (默认: 4)')
    
    # ResNet18专用参数
    parser.add_argument('--freeze-epochs', type=int, default=10,
                       help='ResNet18冻结特征层的训练轮数 (默认: 10)')
    
    # 其他参数
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='数据集目录 (默认: ./data)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='训练设备 (默认: 自动检测，优先使用cuda)')
    
    args = parser.parse_args()
    
    # 打印模型信息
    model_info = get_model_info(args.model)
    print(f"\n模型信息:")
    print(f"  名称: {args.model.upper()}")
    print(f"  描述: {model_info['description']}")
    print(f"  输入尺寸: {model_info['input_size']}x{model_info['input_size']}")
    print(f"  预期精度: {model_info['expected_accuracy']}")
    print(f"  参数量: {model_info['parameters']}")
    
    # 打印训练参数
    default_epochs = 30 if args.model == 'resnet18' else 50
    default_batch_size = 64 if args.model == 'resnet18' else 128
    print(f"\n训练参数:")
    print(f"  Epochs: {args.epochs if args.epochs else default_epochs}")
    print(f"  Batch Size: {args.batch_size if args.batch_size else default_batch_size}")
    print(f"  Learning Rate: {args.learning_rate if args.learning_rate else model_info['learning_rate']}")
    print(f"  Early Stop Patience: {args.patience}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Prefetch Factor: {args.prefetch_factor}")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  Device: {args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')}")
    if args.model == 'resnet18':
        print(f"  Freeze Epochs: {args.freeze_epochs}")
    print()
    
    train(args.model, args)
