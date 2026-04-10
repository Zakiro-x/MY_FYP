"""
train.py
--------
训练 EfficientNetV2B2 - 优化版本
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import config

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

config.create_dirs()


def load_image_preserve_aspect(image_path, target_size=300):
    """加载图片并保持宽高比填充到正方形"""
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


def create_dataset(directory, shuffle=True):
    """从文件夹创建 TensorFlow 数据集"""
    class_names = sorted([d for d in os.listdir(directory) 
                         if os.path.isdir(os.path.join(directory, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_to_idx[class_name])
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    if shuffle:
        np.random.seed(config.SEED)
        indices = np.random.permutation(len(image_paths))
        image_paths = image_paths[indices]
        labels = labels[indices]
    
    def load_and_process(path, label):
        img = load_image_preserve_aspect(path, config.IMG_SIZE)
        label_onehot = tf.one_hot(label, depth=len(class_names))
        return img, label_onehot
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names


def create_augmentation():
    """创建数据增强层"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(config.AUGMENTATION['rotation']),
        tf.keras.layers.RandomZoom(config.AUGMENTATION['zoom']),
        tf.keras.layers.RandomTranslation(
            config.AUGMENTATION['translation'], 
            config.AUGMENTATION['translation']
        ),
        tf.keras.layers.RandomContrast(config.AUGMENTATION['contrast']),
    ])


def preprocess_only(x, y):
    """仅预处理，不增强"""
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    return x, y


def augment_and_preprocess(x, y, augmentation):
    """数据增强 + 预处理"""
    x = augmentation(x, training=True)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    return x, y


def build_model(num_classes):
    """构建优化后的模型 - 更大的分类头 + BatchNormalization"""
    base_model = tf.keras.applications.EfficientNetV2B2(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # 更大的分类头
    x = tf.keras.layers.Dense(512, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_REG))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compute_class_weights(train_dir, class_names):
    """计算类别权重"""
    counts = []
    for c in class_names:
        class_path = os.path.join(train_dir, c)
        n = len([f for f in os.listdir(class_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        counts.append(n)
    
    y = []
    for idx, n in enumerate(counts):
        y.extend([idx] * n)
    y = np.array(y, dtype=np.int32)
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    
    return {int(i): float(w) for i, w in enumerate(class_weights)}, counts


def save_configs(class_names):
    """保存类别映射和实验配置"""
    class_indices = {name: i for i, name in enumerate(class_names)}
    labels = {i: name for name, i in class_indices.items()}
    
    with open(os.path.join(config.CONFIGS_DIR, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)
    
    with open(os.path.join(config.CONFIGS_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    
    exp_config = {
        "img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "epochs_frozen": config.EPOCHS_FROZEN,
        "epochs_finetune": config.EPOCHS_FINETUNE,
        "dropout_rate": config.DROPOUT_RATE,
        "l2_reg": config.L2_REG,
        "augmentation": config.AUGMENTATION,
    }
    with open(os.path.join(config.CONFIGS_DIR, "exp_config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)


def main():
    print("=" * 60)
    print("优化版训练 - EfficientNetV2B2")
    print("=" * 60)
    
    train_dir = os.path.join(config.SPLITS_DIR, "train")
    val_dir = os.path.join(config.SPLITS_DIR, "val")
    test_dir = os.path.join(config.SPLITS_DIR, "test")
    
    print("\n正在加载数据集...")
    train_ds, class_names = create_dataset(train_dir, shuffle=True)
    val_ds, _ = create_dataset(val_dir, shuffle=False)
    test_ds, _ = create_dataset(test_dir, shuffle=False)
    
    print(f"类别: {class_names}")
    
    save_configs(class_names)
    
    class_weight, counts = compute_class_weights(train_dir, class_names)
    print("\n类别分布:")
    for name, count in zip(class_names, counts):
        print(f"  {name}: {count} 张")
    print(f"类别权重: {class_weight}")
    
    print("\n正在应用数据增强...")
    augmentation = create_augmentation()
    
    train_ds = train_ds.map(
        lambda x, y: augment_and_preprocess(x, y, augmentation), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
    
    print("\n正在构建优化模型...")
    model, base_model = build_model(len(class_names))
    model.summary()
    
    # ==================== 阶段1：冻结训练 ====================
    print(f"\n【阶段1】冻结基础模型训练 ({config.EPOCHS_FROZEN} 轮)")
    
    callbacks_frozen = [
        CSVLogger(os.path.join(config.LOGS_DIR, "training_log.csv"), append=False),
        ReduceLROnPlateau(
            monitor=config.REDUCE_LR['monitor'],
            factor=config.REDUCE_LR['factor'],
            patience=config.REDUCE_LR['patience'],
            min_lr=config.REDUCE_LR['min_lr']
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, "best_model_frozen.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode='max'
        )
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=config.LR_FROZEN),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=callbacks_frozen,
        class_weight=class_weight,
        verbose=1
    )
    
    # ==================== 阶段2：微调 ====================
    print(f"\n【阶段2】微调最后 {config.FT_LAYERS} 层 (最多 {config.EPOCHS_FINETUNE} 轮)")
    
    for layer in base_model.layers[-config.FT_LAYERS:]:
        layer.trainable = True
    
    callbacks_finetune = [
        CSVLogger(os.path.join(config.LOGS_DIR, "training_log.csv"), append=True),
        EarlyStopping(
            monitor=config.EARLY_STOPPING['monitor'],
            patience=config.EARLY_STOPPING['patience'],
            restore_best_weights=config.EARLY_STOPPING['restore_best_weights'],
            min_delta=config.EARLY_STOPPING['min_delta'],
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor=config.REDUCE_LR['monitor'],
            factor=config.REDUCE_LR['factor'],
            patience=config.REDUCE_LR['patience'],
            min_lr=config.REDUCE_LR['min_lr'],
            mode='max'
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode='max'
        )
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=config.LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=callbacks_finetune,
        class_weight=class_weight,
        verbose=1
    )
    
    # ==================== 最终评估 ====================
    print("\n最终评估:")
    best_model = tf.keras.models.load_model(os.path.join(config.MODELS_DIR, "best_model.keras"))
    
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    
    print(f"验证集 - 准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"测试集 - 准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    results = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "class_counts": {name: count for name, count in zip(class_names, counts)},
        "class_weights": class_weight
    }
    
    with open(os.path.join(config.REPORTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n训练完成！")
    print(f"模型保存在: {os.path.join(config.MODELS_DIR, 'best_model.keras')}")


if __name__ == "__main__":
    main()