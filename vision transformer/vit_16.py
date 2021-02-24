
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        if len(inputs.shape) == 3:
            N = 1
            W, H, C = inputs.shape
        else:
            N, W, H, C = inputs.shape
            
#         batch_size = tf.shape(inputs)[0]

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
#         patch_dims = x.shape[-1]
#         x = tf.reshape(patches, [batch_size, -1, patch_dims])
        x = tf.reshape(x, (N, -1, C*self.kernel_size**2))

        return x



# ----------------------------------
# Image(WHC) -> patches(N*C*k**2)
# ----------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



# ----------------------------------
# class + pos + patches = tokens
# ----------------------------------
class Patches2Tokens(layers.Layer):
    def __init__(self, num_patches, hidden_dims, dropout_rate=0.1, name="Patches2Tokens"):
        super().__init__()
        
        self.num_patches = num_patches
        self.hidden_dims = hidden_dims
        
        self.linear_projection = layers.Dense(units=self.hidden_dims)

        # class embedding
        self.class_embedding = self.add_weight("class_embedding", shape=(1, 1, self.hidden_dims), trainable=True)
        
        # position embedding
        self.position_embedding = self.add_weight("position_embedding", shape=(1, self.num_patches+1, self.hidden_dims),
                                                  trainable=True)
        
        self.dropout = keras.layers.Dropout(dropout_rate)
        
        
        
    def call(self, inputs):

        batch_size = inputs.shape[0]

        if batch_size is None:
            batch_size = -1


        x = self.linear_projection(inputs)
        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.hidden_dims])
        x = tf.concat([class_embedding, x], axis=1)
        # postion_embedding + Concat(Linear, class) -> (1, 577, 768)
        x += self.position_embedding
        x = self.dropout(x)

        return x


# ----------------------------------
# Multi-head-self-attention
# ----------------------------------
class MSA(keras.layers.Layer):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def build(self, input_shape):
        embeded_dims = input_shape[-1]
        num_heads = self.num_heads



        if embeded_dims % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embeded_dims} should be divisible by number of heads = {num_heads}"
            )

        self.embeded_dims = embeded_dims

        # head_size
        self.projection_dim = embeded_dims // num_heads

        # q k v 
        self.query_dense = tf.keras.layers.Dense(embeded_dims, name="query")
        self.key_dense = tf.keras.layers.Dense(embeded_dims, name="key")
        self.value_dense = tf.keras.layers.Dense(embeded_dims, name="value")
        
        # W0合并所有的head
        self.combine_heads = tf.keras.layers.Dense(embeded_dims, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights


    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)


        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention, (batch_size, -1, self.embeded_dims))
        
        output = self.combine_heads(concat_attention)
        return output, weights



# ----------------------------------
# transformer encoder 
# ----------------------------------
class TransformerEncoderBlock(keras.layers.Layer):
	def __init__(self, unique_name, hidden_dims, num_heads, dropout_rate=0.1):
		super().__init__(name=unique_name)

		self.hidden_dims = hidden_dims
		self.num_heads = num_heads

		self.head_size = (int)(self.hidden_dims / self.num_heads)

		self.layer_norm_1 = keras.layers.LayerNormalization(name="LN_1")
		self.multi_head_atttention = MSA(num_heads=self.num_heads)
        
# 		self.multi_head_atttention = keras.layers.MultiHeadAttention(
#             num_heads=self.num_heads, key_dim=embeded_dim
#         )

		self.dropout = keras.layers.Dropout(dropout_rate)
		self.layer_norm_2 = keras.layers.LayerNormalization(name="LN_2")
		self.mlp_layer = MLPLayer(self.hidden_dims)


	def call(self, inputs):
		input_x = inputs
		x = self.layer_norm_1(inputs)
# 		x = self.multi_head_atttention(x, x) 
		x, _ = self.multi_head_atttention(x)
		x = self.dropout(x)
		x = x + input_x

		y = self.layer_norm_2(x)
		y = self.mlp_layer(y)

		return x + y



# ----------------------------------
# MLP
# ----------------------------------
class MLPLayer(keras.layers.Layer):

    def __init__(self, hidden_dims):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        
        # 4 * embeded_dim = 4 * 768 = 3072 ,参数 4 默认值，base模型 MLP size = 3072
        self.layer1 = keras.layers.Dense(4 * self.hidden_dims, activation=tf.nn.gelu, name='Dense_0')
        self.dropout1 = keras.layers.Dropout(0.1)
        
        # self.layer2 = keras.layers.Dense(self.hidden_dims, name='Dense_1')
        self.layer2 = keras.layers.Dense(self.hidden_dims, name='Dense_1')
        self.dropout2 = keras.layers.Dropout(0.1)

    def call(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return self.dropout2(x)


# ----------------------------------
# vit_16_model
# ----------------------------------
class vit_16_builder():
    def __init__(self, 
                patch_size = 16,
                img_size = (384, 384),
                num_layers = 12,
                num_heads = 12,
                num_classes = 1000,
                ):
        
        self.num_patches = (img_size[0] // patch_size) ** 2
        self.hidden_dims = 768
        self.input_shape = img_size+(3,)
        self.batch_size = 64
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes

        
    def build(self):

        inputs = layers.Input(shape=self.input_shape, batch_size=self.batch_size, name="Input_layer") 
        
        #. 1 patches
        # patches = Patches(self.patch_size)(inputs) # (1, 576, 768)
        patches = SoftSplit(kernel_size=self.patch_size, stride=self.patch_size, padding_num=0)(inputs)

        # 2 class + pos + patches = tokens
        x = Patches2Tokens(num_patches=self.num_patches, hidden_dims=self.hidden_dims)(patches) # (1, 576, 768)
        # print(encoded_patches.shape)

        # 3、Transforer Encoder (* num_layers)
        for i in range(self.num_layers):
            x = TransformerEncoderBlock(unique_name=f"tranformer_encoder_{i}", 
                                        hidden_dims=self.hidden_dims, 
                                        num_heads=self.num_heads)(x)

        # 4、LN
        x = keras.layers.LayerNormalization(name='encoder_LN')(x)

        # 5、head
        outputs = keras.layers.Dense(self.num_classes, name='head')(x[:, 0])

        # 6、build model
        vit_model = keras.Model(inputs, outputs, name="vit_16_base")
        return vit_model


vit_16 = vit_16_builder().build()
vit_16.summary()