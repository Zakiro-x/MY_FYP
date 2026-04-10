"""
plot_curves.py
--------------
绘制训练曲线（准确率和损失）
阶段2的 epoch 接在阶段1后面
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import config


def plot_training_curves(log_path, save_dir):
    """
    绘制训练过程中的准确率和损失曲线
    阶段2的 epoch 会接在阶段1后面
    """
    if not os.path.exists(log_path):
        print(f"错误: 训练日志不存在 - {log_path}")
        return
    
    # 读取日志
    log = pd.read_csv(log_path)
    
    print(f"读取到 {len(log)} 条记录")
    
    # 重新计算 epoch 编号（阶段2接在阶段1后面）
    # 找出阶段1和阶段2的分界点（epoch 重置为0的地方）
    epochs = []
    current_base = 0
    for i, epoch_val in enumerate(log['epoch']):
        if i > 0 and epoch_val == 0:
            # 遇到 epoch 重置为0，说明进入阶段2
            current_base = log['epoch'].iloc[i-1] + 1
        epochs.append(current_base + epoch_val)
    
    log['epoch_continuous'] = epochs
    
    print(f"连续 epoch: {log['epoch_continuous'].iloc[0]} -> {log['epoch_continuous'].iloc[-1]}")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== 左图：准确率曲线 ==========
    ax1.plot(log['epoch_continuous'], log['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(log['epoch_continuous'], log['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=current_base, color='gray', linestyle='--', alpha=0.5, label='Stage 1 → Stage 2')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training and Validation Accuracy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== 右图：损失曲线 ==========
    ax2.plot(log['epoch_continuous'], log['loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(log['epoch_continuous'], log['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.axvline(x=current_base, color='gray', linestyle='--', alpha=0.5, label='Stage 1 → Stage 2')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Validation Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {save_path}")
    
    # 打印最佳结果
    best_val_acc = log['val_accuracy'].max()
    best_idx = log.loc[log['val_accuracy'].idxmax(), 'epoch_continuous']
    print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {int(best_idx)})")


def main():
    print("正在绘制训练曲线...")
    
    log_path = os.path.join(config.LOGS_DIR, "training_log.csv")
    plot_training_curves(log_path, config.FIGURES_DIR)


if __name__ == "__main__":
    main()