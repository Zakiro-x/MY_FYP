# plot_model_structure.py
# Matplotlib Professional - Model Structure Diagram
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import config

# 屏蔽 TF 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --------------------------
# 1. 构建你的模型（和训练完全一致）
# --------------------------
def build_model():
    base_model = tf.keras.applications.EfficientNetV2B2(
        include_top=False,
        weights=None,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification Head
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)

    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)

    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# --------------------------
# 2. 全新绘图设计（英文 + 极简学术风）
# --------------------------
def plot_model_beautiful(model):
    # 模型层名称（英文，无乱码）
    layer_labels = [
        f"Input\n({config.IMG_SIZE}×{config.IMG_SIZE}×3)",
        "Preprocess",
        "Backbone\nEfficientNetV2-B2",
        "Global Avg Pooling",
        "Dense 512\nReLU",
        "Batch Norm",
        f"Dropout\n({config.DROPOUT_RATE})",
        "Dense 256\nReLU",
        "Batch Norm",
        f"Dropout\n({config.DROPOUT_RATE})",
        f"Output\n{config.NUM_CLASSES} Classes\nSoftmax"
    ]

    # 学术配色（高级、不刺眼、论文友好）
    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
        "#6A994E", "#577590", "#F3722C", "#F9C74F",
        "#90A959", "#43AA8B", "#277DA1"
    ]

    # 画布设置
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(10, 12))
    plt.title("AD MRI Classification Model Structure", fontsize=16, weight='bold', pad=25)
    plt.axis('off')

    total_layers = len(layer_labels)
    box_width = 3.0
    box_height = 0.9
    x_center = 0

    # 循环绘制每一层
    for i, (label, color) in enumerate(zip(layer_labels, colors)):
        y = total_layers - i - 1  # 从上到下排列

        # 绘制圆角矩形风格层
        rect = plt.Rectangle(
            (x_center - box_width/2, y - box_height/2),
            box_width, box_height,
            facecolor=color, edgecolor='white', linewidth=2.5,
            zorder=2
        )
        plt.gca().add_patch(rect)

        # 文字
        plt.text(
            x_center, y, label,
            ha='center', va='center', fontsize=11,
            color='white', weight='bold'
        )

        # 绘制箭头
        if i < total_layers - 1:
            plt.arrow(
                x_center, y - box_height/2 - 0.05,
                0, -0.4,
                head_width=0.15, head_length=0.1,
                fc='black', ec='black', zorder=1
            )

    plt.xlim(-2, 2)
    plt.ylim(-0.5, total_layers + 0.5)
    plt.tight_layout()

    # 保存高清图
    save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_structure_final.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("✅ SUCCESS: Model structure saved!")
    print(f"📂 Path: {save_path}")
    print("🎨 Design: Clean English + Academic Style")
    print("✅ No Chinese, No Garbled, No Errors!")
    print("=" * 60)

# --------------------------
# 运行
# --------------------------
if __name__ == "__main__":
    model = build_model()
    plot_model_beautiful(model)
    model.summary()