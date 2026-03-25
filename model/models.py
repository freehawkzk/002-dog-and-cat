"""
猫狗分类模型定义
包含三种模型架构：
1. CatDogMLP - 多层感知机
2. CatDogCNN - 卷积神经网络
3. ResNet18Classifier - 预训练ResNet18
"""

import torch
import torch.nn as nn
from torchvision import models


# ==================== MLP模型 ====================
class CatDogMLP(nn.Module):
    """
    多层感知机模型
    结构：
    - 输入层：150528 (224x224x3)
    - 隐藏层1：2048神经元 + BatchNorm + Dropout
    - 隐藏层2：1024神经元 + BatchNorm + Dropout
    - 隐藏层3：512神经元 + BatchNorm + Dropout
    - 输出层：2（猫、狗）
    """
    def __init__(self, input_dim=150528, num_classes=2):
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
        # 展平图像: (batch, 3, 224, 224) -> (batch, 150528)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        
        return x


# ==================== CNN模型 ====================
class CatDogCNN(nn.Module):
    """
    改进的4层CNN卷积神经网络
    结构：Conv -> BN -> ReLU -> Pool (x4) -> FC -> FC
    输入：224x224x3
    """
    def __init__(self, num_classes=2, img_size=224):
        super(CatDogCNN, self).__init__()
        
        self.img_size = img_size
        
        # 第一层卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三层卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 第四层卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 计算全连接层输入维度
        # 输入: img_size x img_size -> 4次池化 -> img_size/16 x img_size/16
        fc_input_dim = 256 * (img_size // 16) * (img_size // 16)
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
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


# ==================== ResNet18模型 ====================
class ResNet18Classifier(nn.Module):
    """
    基于预训练ResNet18的猫狗分类器
    迁移学习策略：冻结特征提取层，只训练分类头
    输入：224x224x3 (ResNet标准输入)
    """
    def __init__(self, num_classes=2, freeze_features=True):
        super(ResNet18Classifier, self).__init__()
        
        # 加载预训练ResNet18 (使用新的weights参数)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
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


# ==================== 模型工厂函数 ====================
def get_model(model_name, num_classes=2, **kwargs):
    """
    模型工厂函数，根据名称创建模型
    
    参数:
        model_name: 模型名称 ('mlp', 'cnn', 'resnet18')
        num_classes: 分类数量
        **kwargs: 模型特定参数
        
    返回:
        model: 对应的模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'mlp':
        input_dim = kwargs.get('input_dim', 150528)  # 224x224x3
        model = CatDogMLP(input_dim=input_dim, num_classes=num_classes)
        
    elif model_name == 'cnn':
        img_size = kwargs.get('img_size', 224)
        model = CatDogCNN(num_classes=num_classes, img_size=img_size)
        
    elif model_name == 'resnet18':
        freeze_features = kwargs.get('freeze_features', True)
        model = ResNet18Classifier(num_classes=num_classes, freeze_features=freeze_features)
        
    else:
        raise ValueError(f"未知的模型名称: {model_name}. 可选: 'mlp', 'cnn', 'resnet18'")
    
    return model


# ==================== 获取模型信息 ====================
def get_model_info(model_name):
    """
    获取模型的配置信息
    
    返回:
        dict: 包含模型相关配置（输入尺寸、学习率建议等）
    """
    model_configs = {
        'mlp': {
            'input_size': 224,
            'learning_rate': 0.0001,
            'description': '3层MLP，适合快速实验',
            'expected_accuracy': '65-75%',
            'parameters': '~330M'
        },
        'cnn': {
            'input_size': 224,
            'learning_rate': 0.001,
            'description': '4层CNN，平衡性能与速度',
            'expected_accuracy': '82-88%',
            'parameters': '~2M'
        },
        'resnet18': {
            'input_size': 224,
            'learning_rate': 0.001,
            'description': '预训练ResNet18，最高性能',
            'expected_accuracy': '92-96%',
            'parameters': '~11M'
        }
    }
    
    model_name = model_name.lower()
    if model_name not in model_configs:
        raise ValueError(f"未知的模型名称: {model_name}")
    
    return model_configs[model_name]


if __name__ == '__main__':
    # 测试模型创建
    print("测试模型创建...")
    
    # 测试MLP
    mlp = get_model('mlp')
    x = torch.randn(2, 3, 224, 224)
    out = mlp(x)
    print(f"MLP输出形状: {out.shape}")
    
    # 测试CNN
    cnn = get_model('cnn')
    x = torch.randn(2, 3, 224, 224)
    out = cnn(x)
    print(f"CNN输出形状: {out.shape}")
    
    # 测试ResNet18
    resnet = get_model('resnet18')
    x = torch.randn(2, 3, 224, 224)
    out = resnet(x)
    print(f"ResNet18输出形状: {out.shape}")
    
    print("\n所有模型测试通过！")
