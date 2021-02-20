import tensorflow as tf
from tensorflow import keras
import numpy as np

#----------------------------------
# soft split in T2T-ViT
# corresponds to torch.nn.unfold()
#----------------------------------
# - inputs_shape = (N, H, W, C)
# - outputs_shape = (N, L, C*kernel_size**2)
#---------------------------------
class SoftSplit(tf.keras.layers.Layer):
    def __init__(self, kernel_size, stride, padding_num=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_num = padding_num


    def call(self, inputs):
        N, W, H, C = inputs.shape

        # padding
        x = tf.keras.layers.ZeroPadding2D(padding=self.padding_num)(inputs)

        # soft split
        x = tf.image.extract_patches(
            x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
            rates=[1, 1, 1, 1]
            )

        # reshape
        x = tf.reshape(x, (N, -1, C*self.kernel_size**2))

        return x




#----------------------------------
# test code
#----------------------------------
inputs = np.random.randint(0, 256, (1, 224, 224, 3))
outputs = SoftSplit(kernel_size=7,
                    stride=4,
                    padding_num=2)(inputs) # (1, 3136, 147)




