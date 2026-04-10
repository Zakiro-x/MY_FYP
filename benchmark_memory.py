import os
import psutil
import time
from predict import get_model_and_labels

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    start_load = time.time()
    model, labels = get_model_and_labels()
    load_time = time.time() - start_load

    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"模型加载时间: {load_time:.2f} 秒")
    print(f"加载前内存: {mem_before:.2f} MB")
    print(f"加载后内存: {mem_after:.2f} MB")
    print(f"内存增加: {mem_after - mem_before:.2f} MB")