#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#----------------------------------
# soft split in T2T-ViT
# corresponds to torch.nn.unfold()
# Done! 
#----------------------------------
class SoftSplit(keras.layers.Layer):
    """
    #----------------------------------------------
    # - inputs_shape = (N, H, W, C)
    # - outputs_shape = (N, L, C*kernel_size**2)
    #----------------------------------------------
    """
    def __init__(self, kernel_size, strides, padding_num=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_num = padding_num


    def call(self, inputs):
        if len(inputs.shape) == 3:
            N = 1
            W, H, C = inputs.shape
        else:
            N, W, H, C = inputs.shape
            

        # zero padding
        x = tf.keras.layers.ZeroPadding2D(padding=self.padding_num)(inputs)

        # soft split
        x = tf.image.extract_patches(
            x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.strides, self.strides, 1],
            padding="VALID",
            rates=[1, 1, 1, 1]
            )

        # reshape
        x = tf.reshape(x, (N, -1, C*self.kernel_size**2))

        return x



# ----------------------------------
# MSA
# ----------------------------------
class MSA(keras.layers.Layer):
    def __init__(self, num_heads, embeded_dims):
        super().__init__()
        self.num_heads = num_heads
        self.embeded_dims = embeded_dims

    def build(self, input_shape):
        # embeded_dims = input_shape[-1]
        embeded_dims = self.embeded_dims
        num_heads = self.num_heads



        if embeded_dims % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {self.embeded_dims} should be divisible by number of heads = {num_heads}"
            )

        # head_size
        self.projection_dim = self.embeded_dims // num_heads

        # q k v 
        self.query_dense = tf.keras.layers.Dense(self.embeded_dims, name="query")
        self.key_dense = tf.keras.layers.Dense(self.embeded_dims, name="key")
        self.value_dense = tf.keras.layers.Dense(self.embeded_dims, name="value")
        
        # W0合并所有的head
        self.combine_heads = tf.keras.layers.Dense(self.embeded_dims, name="out")

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
# MLP 
# ----------------------------------
class MLP(keras.layers.Layer):

    def __init__(self, hidden_dims, mlp_ratio=1.0):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.mlp_ratio = mlp_ratio
        
        self.layer1 = keras.layers.Dense(self.mlp_ratio * self.hidden_dims, 
                                         activation=tf.nn.gelu, 
                                         name='mlp_Dense_0')
        self.dropout1 = keras.layers.Dropout(0.1)
        
        self.layer2 = keras.layers.Dense(self.hidden_dims, name='mlp_Dense_1')
        self.dropout2 = keras.layers.Dropout(0.1)

    def call(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return self.dropout2(x)


# ----------------------------------
# T2T-transformer 
# depth = 2
# num_heads = 1, 
# mlp_ratio = 1.0
# MLP_units = 64 
# Linear embed_dim=384, (t2t-vit-14)
# MSA后无residual
# ----------------------------------
class T2T_Transformer(keras.layers.Layer):
	def __init__(self, 
                 unique_name, 
                 hidden_dims=64, 
                 num_heads=1, 
                 dropout_rate=0.1, 
                 mlp_ratio=1.0
                 ):
		super().__init__(name=unique_name)

		self.hidden_dims = hidden_dims
		self.num_heads = num_heads

		self.layer_norm_1 = keras.layers.LayerNormalization(name="LN_1")
		self.multi_head_atttention = MSA(num_heads=self.num_heads, embeded_dims=self.hidden_dims)

		self.dropout = keras.layers.Dropout(dropout_rate)
		self.layer_norm_2 = keras.layers.LayerNormalization(name="LN_2")
		self.mlp_layer = MLP(hidden_dims=self.hidden_dims, mlp_ratio=mlp_ratio)


	def call(self, inputs):
		input_x = inputs
		x = self.layer_norm_1(inputs)
		x, _ = self.multi_head_atttention(x)
		x = self.dropout(x)
 		# x = x + input_x

		y = self.layer_norm_2(x)
		y = self.mlp_layer(y)

		return x + y



# ----------------------------------
# T2T-module 
# ----------------------------------
class T2T_Module(keras.layers.Layer):
    def __init__(self, 
                 hidden_dims=64,
                 num_heads=1,
                 mlp_ratio=1.0
                 ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio


        self.soft_split1 = SoftSplit(kernel_size=7, strides=4, padding_num=2)
        self.soft_split2 = SoftSplit(kernel_size=3, strides=2, padding_num=1)
        self.soft_split3 = SoftSplit(kernel_size=3, strides=2, padding_num=1)

        self.linear_projection = keras.layers.Dense(384, name="linear_projection")
        

    def call(self, inputs):

        # soft_split1:  (32, 3136, 147)
        x = self.soft_split1(inputs) 

        # T2T_transformer_1:  (32, 3136, 147)
        x = T2T_Transformer(unique_name="T2T_process_encoder1", 
                            hidden_dims=self.hidden_dims, 
                            num_heads=self.num_heads, 
                            mlp_ratio=self.mlp_ratio
                            )(x)


        # reshape_1:  (32, 56, 56, 64)
        N, new_HW, C = x.shape
        x = tf.reshape(x, [N, (int)(np.sqrt(new_HW)), (int)(np.sqrt(new_HW)), C]) 


        # soft_split2:  (32, 784, 576)
        x = self.soft_split2(x)


        # T2T_transformer_2:  (32, 784, 64)
        x = T2T_Transformer(unique_name="T2T_process_encoder2", 
                            hidden_dims=self.hidden_dims, 
                            num_heads=self.num_heads, 
                            mlp_ratio=self.mlp_ratio
                            )(x)

        # reshape_2:  (32, 28, 28, 64)
        N, new_HW, C = x.shape
        x = tf.reshape(x, [N, (int)(np.sqrt(new_HW)), (int)(np.sqrt(new_HW)), C])  

        # soft_split3:  (32, 196, 576)
        x = self.soft_split3(x)

        # linear_projection: (32, 196, 768)
        x = self.linear_projection(x)

        return x


# ----------------------------------------------------------------------
# class_embedding + pos_embedding + patches_embedding -> tokens
# pos_embedding 待修改
# ----------------------------------------------------------------------
class FixedTokens(layers.Layer):
    def __init__(self, num_patches, hidden_dims=384, dropout_rate=0.1):
        super().__init__()
        
        self.num_patches = num_patches
        self.hidden_dims = hidden_dims
        
        self.linear_projection = layers.Dense(units=self.hidden_dims)

        # class embedding
        self.class_embedding = self.add_weight("class_embedding", shape=(1, 1, self.hidden_dims), trainable=True)
        
        # position embedding
        self.position_embedding = self.add_weight("position_embedding", 
                                                  shape=(1, self.num_patches+1, self.hidden_dims),
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
# transformer encoder  标准encoder
# residual在MLP后有，MSA后也有。
# ----------------------------------
class TransformerLayer(keras.layers.Layer):
    def __init__(self, 
                 unique_name, 
                 hidden_dims=384, 
                 num_heads=6, 
                 dropout_rate=0.1, 
                 mlp_ratio=3.0
                 ):
        super().__init__(name=unique_name)

        self.hidden_dims = hidden_dims
        self.num_heads = num_heads

        self.layer_norm_1 = keras.layers.LayerNormalization(name="LN_1")
        self.multi_head_atttention = MSA(num_heads=self.num_heads, embeded_dims=self.hidden_dims)

        self.dropout = keras.layers.Dropout(dropout_rate)
        self.layer_norm_2 = keras.layers.LayerNormalization(name="LN_2")
        self.mlp_layer = MLP(hidden_dims=self.hidden_dims, mlp_ratio=mlp_ratio)


    def call(self, inputs):
        input_x = inputs
        x = self.layer_norm_1(inputs)
        x, _ = self.multi_head_atttention(x)
        x = self.dropout(x)
        x = x + input_x

        y = self.layer_norm_2(x)
        y = self.mlp_layer(y)

        return x + y



# ----------------------------------
# t2t-vit-14_model
# embed_dim=384, depth=14, num_heads=6, mlp_ratio=3

# 14 / 19/ 24
# [depth | num_heads | hidden_dims | MLP_Size]
# [14, 6, 384, 384*3]
# [19, 7, 448, 448*3]
# [24, 8, 512, 512*3] 

# ----------------------------------



class T2T_ViT():
    def __init__(self, 
                img_size = (224, 224),
                batch_size = 64,
                num_classes = 1000,
                hidden_dims_backbone=384,
                num_layers_backbone = 14,
                num_heads_bakcbone = 6
                ):
        
        # fixed parameters
        self.img_size = img_size
        self.batch_size = batch_size
        self.input_shape = img_size+(3,)
        self.num_classes = num_classes
        self.num_patches = (224 // (4 * 2 * 2)) * (224 // (4 * 2 * 2)), # 196, there are 3 sfot split, stride are 4,2,2 seperately


        # t2t_module参数
        self.mlp_ratio_t2t_module = 1.0
        self.num_heads_t2t_module = 1
        self.hidden_dims_t2t_module = 64


        # backbone参数
        self.hidden_dims_backbone = hidden_dims_backbone
        self.num_layers_backbone = num_layers_backbone
        self.num_heads_bakcbone = num_heads_bakcbone
        self.mlp_ratio_backbone = 3.0


        

    def build(self):
        inputs = keras.Input(shape=self.img_size+(3,), batch_size=self.batch_size, name="Input_Layer")
        
        # T2T_Module
        x = T2T_Module(hidden_dims=self.hidden_dims_t2t_module,
                       num_heads=self.num_heads_t2t_module,
                       mlp_ratio=self.mlp_ratio_t2t_module
                       )(inputs)
        
        
        x = FixedTokens(num_patches=196, hidden_dims=self.hidden_dims_backbone)(x)

        # Transforer Encoder (* num_layers)
        for i in range(self.num_layers_backbone):
            x = TransformerLayer(unique_name=f"tranformer_layer_{i}", 
                                hidden_dims=self.hidden_dims_backbone,
                                num_heads=self.num_heads_bakcbone,
                                mlp_ratio=self.mlp_ratio_backbone,
                                )(x)  

        # LN
        x = keras.layers.LayerNormalization(name='encoder_LN')(x)
        # head
        outputs = keras.layers.Dense(self.num_classes, name='head')(x[:, 0])
        # build model
        model = keras.Model(inputs, outputs)
    
        return model



def t2t_vit_14():
    return T2T_ViT(hidden_dims_backbone=384, num_layers_backbone=14, num_heads_bakcbone=6).build()

def t2t_vit_19():
    return T2T_ViT(hidden_dims_backbone=448, num_layers_backbone=19, num_heads_bakcbone=7).build()

def t2t_vit_24():
    return T2T_ViT(hidden_dims_backbone=512, num_layers_backbone=24, num_heads_bakcbone=8).build()
    

model = t2t_vit_14()
model.summary()




