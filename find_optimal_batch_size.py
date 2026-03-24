"""
自动寻找最优Batch Size工具
用于测试不同batch size下的GPU显存占用和训练速度
"""

import torch
import torch.nn as nn
import time
from cat_dog_classifier_improved import CatDogCNN_Improved, IMG_SIZE, DEVICE

def test_batch_size(batch_size, img_size=150, num_iterations=10):
    """
    测试指定batch size的显存占用和训练速度
    
    参数:
        batch_size: 批次大小
        img_size: 图像尺寸
        num_iterations: 测试迭代次数
    
    返回:
        (显存占用MB, 平均迭代时间ms, 是否成功)
    """
    if not torch.cuda.is_available():
        print("未检测到GPU，无法测试")
        return None, None, False
    
    try:
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建模型和数据
        model = CatDogCNN_Improved().to(DEVICE)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()
        
        # 创建虚拟数据
        images = torch.randn(batch_size, 3, img_size, img_size).to(DEVICE)
        labels = torch.randint(0, 2, (batch_size,)).to(DEVICE)
        
        # 预热
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 正式测试
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 计算指标
        memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        
        # 清理
        del model, images, labels, optimizer, criterion, scaler
        torch.cuda.empty_cache()
        
        return memory_allocated, avg_time, True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, None, False
        else:
            raise e

def find_optimal_batch_size(img_size=150, target_memory_usage=0.90):
    """
    自动寻找最优batch size
    
    参数:
        img_size: 图像尺寸
        target_memory_usage: 目标显存使用率（0-1之间）
    """
    print("=" * 60)
    print("寻找最优Batch Size")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("未检测到GPU！")
        return
    
    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
    print(f"\nGPU型号: {gpu_name}")
    print(f"总显存: {total_memory:.0f} MB")
    print(f"目标显存使用率: {target_memory_usage*100:.0f}%")
    print(f"图像尺寸: {img_size}x{img_size}")
    print("-" * 60)
    
    # 测试不同batch size
    test_sizes = [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    results = []
    
    print("\n测试进度:")
    print(f"{'Batch Size':<12} {'显存(MB)':<12} {'时间(ms)':<12} {'显存使用率':<12} {'状态'}")
    print("-" * 60)
    
    for bs in test_sizes:
        memory, time_ms, success = test_batch_size(bs, img_size)
        
        if success:
            memory_usage = memory / total_memory * 100
            status = "✓" if memory_usage < target_memory_usage * 100 else "⚠ 接近上限"
            results.append((bs, memory, time_ms, memory_usage))
            print(f"{bs:<12} {memory:<12.0f} {time_ms:<12.1f} {memory_usage:<11.1f}% {status}")
        else:
            print(f"{bs:<12} {'OOM':<12} {'OOM':<12} {'OOM':<12} ✗ 显存不足")
            break  # 继续增大也会OOM
    
    if not results:
        print("\n所有batch size都显存不足！请减小图像尺寸。")
        return
    
    print("-" * 60)
    
    # 找到最优batch size
    # 选择显存使用率接近目标且最大的batch size
    optimal_bs = None
    for bs, memory, time_ms, usage in results:
        if usage < target_memory_usage * 100:
            optimal_bs = bs
        else:
            break
    
    if optimal_bs is None:
        optimal_bs = results[0][0]  # 使用最小的batch size
    
    print(f"\n推荐配置:")
    print(f"  最优Batch Size: {optimal_bs}")
    print(f"  预计显存占用: {results[[r[0] for r in results].index(optimal_bs)][1]:.0f} MB")
    print(f"  预计每次迭代时间: {results[[r[0] for r in results].index(optimal_bs)][2]:.1f} ms")
    
    # 计算吞吐量
    best_throughput = 0
    best_bs = results[0][0]
    for bs, memory, time_ms, usage in results:
        if usage < target_memory_usage * 100:
            throughput = bs / time_ms * 1000  # images per second
            if throughput > best_throughput:
                best_throughput = throughput
                best_bs = bs
    
    print(f"\n最高吞吐量配置:")
    print(f"  Batch Size: {best_bs}")
    print(f"  吞吐量: {best_throughput:.1f} images/sec")
    
    print("\n" + "=" * 60)
    print("建议：")
    print(f"1. 将 cat_dog_classifier_improved.py 中的 BATCH_SIZE 改为 {optimal_bs}")
    print(f"2. 如果追求最高吞吐量，可使用 BATCH_SIZE = {best_bs}")
    print("=" * 60)

if __name__ == '__main__':
    find_optimal_batch_size(img_size=150, target_memory_usage=0.85)
