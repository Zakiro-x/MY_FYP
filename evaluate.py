"""
evaluate.py
-----------
模型评估脚本，生成混淆矩阵、分类报告等
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import config
from train import create_dataset, preprocess_only

# 设置matplotlib使用英文，避免Linux下中文显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_test_dataset():
    """
    创建测试数据集，保持与训练时相同的预处理
    """
    test_dir = os.path.join(config.SPLITS_DIR, "test")
    test_ds, class_names = create_dataset(test_dir, shuffle=False)
    
    test_ds = test_ds.map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
    
    return test_ds, class_names


def evaluate_model(model, test_ds, class_names, save_dir):
    """
    完整评估模型，生成混淆矩阵和分类报告
    """
    print("正在收集预测结果...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        true = np.argmax(labels.numpy(), axis=1)
        
        y_true.extend(true)
        y_pred.extend(preds)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 分类报告
    print("\n分类报告:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    # 混淆矩阵 - 使用英文标签
    print("\n正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"混淆矩阵已保存: {os.path.join(save_dir, 'confusion_matrix.png')}")
    
    # 各类别准确率
    print("\n各类别准确率:")
    class_acc = {}
    for i, name in enumerate(class_names):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == i) / np.sum(mask)
            class_acc[name] = acc
            print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    overall_acc = np.mean(y_pred == y_true)
    print(f"\n总体准确率: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # 保存结果
    results = {
        "overall_accuracy": float(overall_acc),
        "class_accuracy": class_acc,
        "num_samples": len(y_true),
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载模型
    model_path = os.path.join(config.MODELS_DIR, "best_model.keras")
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 - {model_path}")
        return
    
    print("正在加载模型...")
    model = tf.keras.models.load_model(model_path)
    
    # 加载类别信息
    labels_path = os.path.join(config.CONFIGS_DIR, "labels.json")
    with open(labels_path, "r") as f:
        labels = json.load(f)
    class_names = [labels[str(i)] for i in range(len(labels))]
    
    # 创建测试数据集
    print("正在加载测试数据...")
    test_ds, _ = create_test_dataset()
    
    # 评估
    evaluate_model(model, test_ds, class_names, config.REPORTS_DIR)
    
    print(f"\n评估完成！结果保存在: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()