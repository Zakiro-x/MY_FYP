"""
predict.py
----------
预测模块，供 app.py 网页调用
从 outputs 目录加载模型进行预测
"""

import os
import json
import numpy as np
import tensorflow as tf
import config

# 全局缓存，避免重复加载模型
_MODEL = None
_LABELS = None


def load_image_preserve_aspect(image_path, target_size=300):
    """
    加载图片并保持宽高比填充到正方形
    与训练时保持一致
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    orig_h = tf.shape(img)[0]
    orig_w = tf.shape(img)[1]
    
    scale = tf.cast(target_size, tf.float32) / tf.cast(tf.maximum(orig_h, orig_w), tf.float32)
    new_h = tf.cast(tf.cast(orig_h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(orig_w, tf.float32) * scale, tf.int32)
    
    resized = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.LANCZOS3)
    
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    padded = tf.pad(resized, 
                    [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    constant_values=0)
    
    return padded


def load_artifacts(artifacts_dir=None):
    """
    从 outputs 目录加载模型和标签映射
    """
    if artifacts_dir is None:
        # 默认使用 config 中的路径
        model_path = os.path.join(config.MODELS_DIR, "model.keras")
        labels_path = os.path.join(config.CONFIGS_DIR, "labels.json")
    else:
        model_path = os.path.join(artifacts_dir, "models", "model.keras")
        labels_path = os.path.join(artifacts_dir, "configs", "labels.json")
    
    if not os.path.exists(model_path):
        # 尝试从 best_model 复制
        best_model_path = os.path.join(config.MODELS_DIR, "best_model.keras")
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            raise FileNotFoundError(f"模型不存在: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    # 确保键是整数
    labels = {int(k): v for k, v in labels.items()}
    
    return model, labels


def get_model_and_labels(artifacts_dir=None):
    """
    懒加载模型，供 Flask 调用
    第一次调用时加载，后续直接返回缓存
    """
    global _MODEL, _LABELS
    if _MODEL is None:
        print("正在加载模型...")
        _MODEL, _LABELS = load_artifacts(artifacts_dir)
        print("模型加载完成！")
    return _MODEL, _LABELS


def predict_one(model, labels, image_path, img_size=None):
    """
    预测单张图片
    返回: (预测类别, 置信度, 所有类别概率字典)
    """
    if img_size is None:
        img_size = model.input_shape[1]
    
    # 加载并预处理图片
    img = load_image_preserve_aspect(image_path, img_size)
    img = tf.expand_dims(img, axis=0)  # 添加批次维度
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    # 预测
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    
    return (
        labels[idx],
        float(probs[idx]),
        {labels[i]: float(probs[i]) for i in range(len(probs))}
    )