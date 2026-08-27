






import os

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,5,6,7'

import time
import torch
import threading


def burn_gpu(device_id):
    """单卡占用线程函数"""
    device = torch.device(f"cuda:{device_id}")
    print(f"进程启动：正在占用逻辑设备 {device} ({torch.cuda.get_device_name(device_id)})")
    
    # 初始化大矩阵（10000x10000 约占用 400MB 显存，可调大）
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    
    while True:
        # 重复矩阵乘法以保持利用率
        c = torch.matmul(a, b)
        time.sleep(0.01) # 稍作停顿，防止主板供电或温度瞬间过高

if __name__ == "__main__":
    # 获取当前可见的 GPU 数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("未检测到可用的 GPU，请检查 CUDA 驱动或 CUDA_VISIBLE_DEVICES 设置！")
        exit()

    print(f"检测到环境中共有 {gpu_count} 块可见显卡，准备同时占用...")
    
    # 为每块可见显卡启动一个线程
    threads = []
    for i in range(gpu_count):
        t = threading.Thread(target=burn_gpu, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    
    print("所有显卡已开始重复运算占位，按 Ctrl+C 退出...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n已收到停止信号，正在退出...")
