"""
config.py
---------
项目配置文件 - 优化后的参数
"""

import os

# 获取当前文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 路径配置 ====================
RAW_DATA_DIR = os.path.join(CURRENT_DIR, "data_raw")
SPLITS_DIR = os.path.join(CURRENT_DIR, "data_splits")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")

MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
CONFIGS_DIR = os.path.join(OUTPUT_DIR, "configs")

# ==================== 数据配置 ====================
IMG_SIZE = 300
BATCH_SIZE = 64
NUM_CLASSES = 4
CLASS_NAMES = ["non_demented", "very_mild", "mild", "moderate"]

# ==================== 训练配置 ====================
SKIP_STAGE1 = False
SEED = 42
EPOCHS_FROZEN = 20  # 从12增加到20
EPOCHS_FINETUNE = 40  # 从18增加到40

LR_FROZEN = 5e-4  # 从1e-3降低，更稳定
LR_FINETUNE = 5e-5  # 从1e-4降低

DROPOUT_RATE = 0.2  # 从0.4降低，减少信息丢失
L2_REG = 0.001  # 从0.005降低
FT_LAYERS = 100  # 从200降低，解冻更少层防止过拟合

# ==================== 数据增强配置（大幅减弱）====================
AUGMENTATION = {
    'rotation': 0.02,  # 从0.05降低
    'zoom': 0.05,      # 从0.1降低
    'translation': 0.02,  # 从0.05降低
    'contrast': 0.05,  # 从0.1降低
    'flip_horizontal': False,
}

# ==================== 早停配置（放宽）====================
EARLY_STOPPING = {
    'monitor': 'val_accuracy',  # 改为监控准确率
    'patience': 8,  # 从5增加到8
    'restore_best_weights': True,
    'min_delta': 0.002
}

# ==================== 学习率衰减配置 ====================
REDUCE_LR = {
    'monitor': 'val_accuracy',  # 改为监控准确率
    'factor': 0.5,
    'patience': 3,  # 从2增加到3
    'min_lr': 1e-7
}


def create_dirs():
    """创建所有需要的文件夹"""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)