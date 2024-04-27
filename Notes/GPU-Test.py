import torch
import pynvml

pynvml.nvmlInit()  # 初始化 NVML

print("CUDA Available:", torch.cuda.is_available())     # 检查 CUDA 是否可用

if torch.cuda.is_available():
    number_of_devices = torch.cuda.device_count()  # 获取设备数量
    print("Number of CUDA devices available:", number_of_devices)

    for i in range(number_of_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取设备句柄
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 获取内存信息
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # 获取设备利用率信息

        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Total Memory: {info.total / 1024 ** 2} MB")
        print(f"    Used Memory: {info.used / 1024 ** 2} MB")
        print(f"    Free Memory: {info.free / 1024 ** 2} MB")
        print(f"    GPU Utilization: {util.gpu}%")
        print(f"    Memory Utilization: {util.memory}%")

pynvml.nvmlShutdown()  # 关闭 NVML
