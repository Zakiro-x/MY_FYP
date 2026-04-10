import time
import numpy as np
import tensorflow as tf
from predict import get_model_and_labels, load_image_preserve_aspect
from config import IMG_SIZE

# 加载模型（首次加载会稍慢，但不计入推理时间）
model, labels = get_model_and_labels()

# 准备一张测试图像（使用你已有的任意一张MRI切片，如测试集中的一张）
test_image_path = "D:/Code/FYP/splits/test/NonDemented/OAS1_0001_MR1_mpr-2_142.jpg"  # 替换为实际路径

# 预处理图像（与predict_one中逻辑一致）
img = load_image_preserve_aspect(test_image_path, IMG_SIZE)
img = tf.expand_dims(img, axis=0)
img = tf.keras.applications.efficientnet_v2.preprocess_input(img)

# 预热：执行一次推理，确保GPU初始化完成
_ = model.predict(img, verbose=0)

# 正式测量：运行100次取平均
n_iter = 100
times = []
for _ in range(n_iter):
    start = time.perf_counter()
    _ = model.predict(img, verbose=0)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # 转换为毫秒

avg_ms = np.mean(times)
std_ms = np.std(times)
print(f"单次推理平均耗时: {avg_ms:.2f} ± {std_ms:.2f} ms")
print(f"最小值: {np.min(times):.2f} ms, 最大值: {np.max(times):.2f} ms")