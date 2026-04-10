import tensorflow as tf
from train import build_model
import config

# 构建模型
model, base_model = build_model(config.NUM_CLASSES)

# 1. 总参数量
total_params = model.count_params()
print(f"总参数量: {total_params / 1e6:.2f} M")

# 2. 可训练参数量（版本二中只训练最后100层 + 分类头）
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

# 3. 非可训练参数量（冻结的主干网络部分）
non_trainable_params = total_params - trainable_params
print(f"非可训练参数量: {non_trainable_params / 1e6:.2f} M")

# 4. 分类头的参数量（可以手动计算或通过层名筛选）
# 假设分类头是模型最后几层，可以根据名称包含 'dense', 'batch_normalization' 等来统计
classifier_layers = ['dense_1', 'dense_2', 'batch_normalization', 'batch_normalization_1', 'dropout', 'dropout_1']
classifier_params = 0
for layer in model.layers:
    if any([cl in layer.name for cl in classifier_layers]):
        classifier_params += layer.count_params()
print(f"分类头参数量: {classifier_params / 1e6:.2f} M")