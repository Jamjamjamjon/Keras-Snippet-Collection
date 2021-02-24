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
# load datatset & hyperParameters
# ------------------------
def create_train_and_validation_dataset(
                                    addr="cats",
                                    image_size=(150,150),
                                    batch_size=64,
                                    validation_split=0.1,
                                    label_mode="categorical",
                                    seed=1117
                                    ):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        addr,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode

    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        addr,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=label_mode
    )

    return train_ds, val_ds

train_ds, val_ds = create_train_and_validation_dataset()

#--------------
# 数据集shape查看
#---------------

for images, labels in train_ds.take(1):
  print(images.shape)
  print(labels.shape)
  # print(labels)
  print(tf.argmax(labels[0]))

# ------------------------
# 可视化
# ------------------------
def image_show(figsize=15, num_row=3, num_column=3):
    plt.figure(figsize=(figsize, figsize))
    for images, labels in train_ds.take(1):
        for i in range(num_row*num_column):
            ax = plt.subplot(num_row, num_column, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(tf.argmax(labels[i]).numpy())
            plt.axis("off")
    plt.show()


# ------------------------
# 防止io阻塞
# ------------------------
# train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_ds = val_ds.cache().batch(batch_size).prefetch(buffer_size=10)
train_ds = train_ds.cache().prefetch(buffer_size=10)
validation_ds = val_ds.cache().prefetch(buffer_size=10)




