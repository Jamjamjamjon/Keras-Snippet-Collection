import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# 过滤掉JPG中不含有JFIF头的图像
# ------------------------
dir_list = ["cats/Abyssinian", 
            "cats/Bengal",
            "cats/Birman",
            "cats/Bombay",
            "cats/British_Shorthair",
            "cats/Egyptian_Mau",
            "cats/Maine_Coon",
            "cats/Persian",
            "cats/Ragdoll",
            "cats/Russian_Blue",
            "cats/Siamese",
            "cats/Sphynx"
            ]

def filter_JPG_without_JFIF_head(dir_list):
    num_skipped = 0
    for folder_name in (dir_list):
        for fname in os.listdir(folder_name):
            fpath = os.path.join(folder_name, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

# filter_JPG_without_JFIF_head(dir_list)


# ------------------------
# 数据集
# ------------------------
image_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "cats",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"

)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "cats",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)


# ------------------------
# 可视化
# ------------------------
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(tf.argmax(labels[i]).numpy())
#         plt.axis("off")
# plt.show()


# ------------------------
# resize to 150 * 150 
# ------------------------
# train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
# validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, image_size), y))


# ------------------------
# 防止io阻塞
# ------------------------
# train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)
train_ds = train_ds.cache().prefetch(buffer_size=10)
validation_ds = val_ds.cache().prefetch(buffer_size=10)



# ------------------------
# data augmentation
# ------------------------
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
    ]
)


# ------------------------
#  model
# ------------------------
base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=image_size+(3,),
    include_top=False,
)  

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=image_size+(3,))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2

# Scale inputs to [-1, +1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(12)(x)
model = keras.Model(inputs, outputs)

model.summary()


# ------------------------
#  compile then train
# ------------------------

checkpoint_filepath = "ckpt/{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max')


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# model.fit(train_ds, epochs=10, validation_data=validation_ds,
          # callbacks=[model_checkpoint_callback])


# ------------------------
#  finetune
# ------------------------ 

base_model.trainable = True
model.summary()


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="ckpt/{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.h5",
    save_weights_only=True,
    monitor='accuracy',
    mode='max')


model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# model.fit(train_ds, epochs=30, validation_data=validation_ds,
#           callbacks=[model_checkpoint_callback])


# ------------------------
#  predict
# ------------------------ 
img = keras.preprocessing.image.load_img(
    "cats/Sphynx/Sphynx_24.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(tf.argmax(score))






