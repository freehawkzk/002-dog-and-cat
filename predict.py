"""
猫狗分类推理脚本
支持加载训练好的模型进行图片分类预测
使用方法：
    python predict.py --model resnet18 --image path/to/image.jpg
    python predict.py --model cnn --image path/to/image.jpg --checkpoint checkpoints/cnn/best_model.pth
"""

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import get_model, get_model_info


# ==================== 配置 ====================
CLASS_NAMES = ['cat', 'dog']


# ==================== 图像预处理 ====================
def get_transform(input_size=224):
    """获取推理时的预处理transforms"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ==================== 加载模型 ====================
def load_model(model_name, checkpoint_path, device):
    """加载训练好的模型"""
    model_info = get_model_info(model_name)
    input_size = model_info['input_size']
    
    # 创建模型
    if model_name == 'mlp':
        model = get_model('mlp', input_dim=input_size * input_size * 3)
    elif model_name == 'cnn':
        model = get_model('cnn', img_size=input_size)
    else:
        model = get_model('resnet18', freeze_features=False)
    
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, input_size


# ==================== 单张图片预测 ====================
def predict_image(model, image_path, input_size, device):
    """对单张图片进行预测"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    transform = get_transform(input_size)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return CLASS_NAMES[predicted_class], confidence, image


# ==================== 批量预测 ====================
def predict_batch(model, image_paths, input_size, device):
    """批量预测多张图片"""
    transform = get_transform(input_size)
    results = []
    
    for image_path in image_paths:
        try:
            class_name, confidence, _ = predict_image(model, image_path, input_size, device)
            results.append({
                'path': image_path,
                'class': class_name,
                'confidence': confidence
            })
        except Exception as e:
            results.append({
                'path': image_path,
                'error': str(e)
            })
    
    return results


# ==================== 可视化结果 ====================
def visualize_prediction(image, predicted_class, confidence, save_path=None):
    """可视化预测结果"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    
    # 设置标题颜色
    color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red'
    
    plt.title(f'预测: {predicted_class}\n置信度: {confidence:.2%}', 
              fontsize=14, color=color)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存: {save_path}")
    
    plt.show()


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='猫狗分类推理脚本')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp', 'cnn', 'resnet18'],
                       help='模型类型: mlp, cnn, resnet18')
    parser.add_argument('--image', type=str, required=True,
                       help='要预测的图片路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重路径（默认使用 checkpoints/{model}/best_model.pth）')
    parser.add_argument('--no-visualize', action='store_true',
                       help='不显示可视化结果')
    parser.add_argument('--save-result', type=str, default=None,
                       help='保存预测结果图的路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查图片是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图片不存在 - {args.image}")
        return
    
    # 确定模型路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join('checkpoints', args.model, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 模型文件不存在 - {checkpoint_path}")
        print(f"请先运行: python train.py --model {args.model}")
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    print(f"权重文件: {checkpoint_path}")
    model, input_size = load_model(args.model, checkpoint_path, device)
    
    # 预测
    print(f"\n预测图片: {args.image}")
    predicted_class, confidence, image = predict_image(model, args.image, input_size, device)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("预测结果:")
    print(f"  类别: {predicted_class}")
    print(f"  置信度: {confidence:.2%}")
    print("=" * 50)
    
    # 可视化
    if not args.no_visualize:
        visualize_prediction(image, predicted_class, confidence, args.save_result)


# ==================== 交互模式 ====================
def interactive_mode():
    """交互式预测模式"""
    print("=" * 50)
    print("猫狗分类器 - 交互模式")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 选择模型
    print("\n可用模型:")
    print("  1. mlp      - MLP模型")
    print("  2. cnn      - CNN模型")
    print("  3. resnet18 - ResNet18模型 (推荐)")
    
    model_name = input("\n请选择模型 (mlp/cnn/resnet18): ").strip().lower()
    if model_name not in ['mlp', 'cnn', 'resnet18']:
        print("无效选择，使用默认模型: resnet18")
        model_name = 'resnet18'
    
    # 加载模型
    checkpoint_path = os.path.join('checkpoints', model_name, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误: 模型文件不存在 - {checkpoint_path}")
        return
    
    print(f"\n加载模型...")
    model, input_size = load_model(model_name, checkpoint_path, device)
    print("模型加载完成！")
    
    # 预测循环
    while True:
        print("\n" + "-" * 50)
        image_path = input("请输入图片路径 (或输入 'q' 退出): ").strip()
        
        if image_path.lower() == 'q':
            print("退出程序")
            break
        
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 - {image_path}")
            continue
        
        try:
            predicted_class, confidence, image = predict_image(model, image_path, input_size, device)
            print(f"\n预测结果: {predicted_class} (置信度: {confidence:.2%})")
            
            show_vis = input("显示可视化结果? (y/n): ").strip().lower()
            if show_vis == 'y':
                visualize_prediction(image, predicted_class, confidence)
        except Exception as e:
            print(f"预测失败: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1:
        # 无参数时进入交互模式
        interactive_mode()
    else:
        main()
