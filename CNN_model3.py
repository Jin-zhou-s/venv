import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from Data_process_class import Process_data, Image_preprocessing
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# 加载和准备数据
process_data = Process_data()
image_process = Image_preprocessing()

process_data.download_and_unzipped_file("GTSRB_Final_Training_Images.zip")
date_df = process_data.get_processed_data('image_datasets/image/GTSRB/Final_Training/Images')

crop_df = image_process.crop_image(date_df)

# 这样你会得到60%训练，20%验证和20%测试
train_df, test_df = train_test_split(crop_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

print(train_df.head())
print(val_df.head())

train_generator = image_process.train_data_process(train_df)

val_generator = image_process.test_data_process(val_df)

test_generator = image_process.test_data_process(test_df)

print(train_generator)
print(val_generator)

# 使用已经存在的ResNet50模型进行迁移学习
# 加载预训练的ResNet50模型，不包括顶层
base_model = ResNet50(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加最终的softmax层，假设有43个类别
predictions = Dense(43, activation='softmax')(x)

# 定义整个模型
model = Model(inputs=base_model.input, outputs=predictions)

# # 冻结所有卷积层的权重
# for layer in base_model.layers:
#     layer.trainable = False

# 假设我们只解冻最后的N个卷积层
N = 3  # 可以更改这个值为你想要解冻的卷积层的数量
count = 0
for layer in base_model.layers[::-1]:  # 从后向前遍历模型的层
    if isinstance(layer, Conv2D):
        layer.trainable = True
        count += 1
        if count >= N:
            break

# 由于我们进行了模型的微调，通常需要使用更小的学习率
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# # 编译模型
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型概述
model.summary()

# 训练模型
for epoch in range(10):
    print(f"Epoch: {epoch+1}/10")
    model.fit(
        train_generator,
        epochs=1,
        steps_per_epoch=len(train_df) // 32,
        validation_data=val_generator,
        validation_steps=len(val_df) // 32,
    )
    image_process.reset_train_generator()
    image_process.reset_test_generator()

# 模型评估
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_df) // 32)
print('Test accuracy:', test_acc, 'Test loss:', test_loss)
