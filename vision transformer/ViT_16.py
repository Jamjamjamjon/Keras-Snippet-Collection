import tensorflow as tf
from tensorflow import keras
import typing
import numpy as np


# ----------------------------------
# TransformerInputConv2DLayer 
# ----------------------------------
class TransformerInputConv2DLayer(tf.keras.layers.Layer):
    """

	# module 1: TransformerInputConv2DLayer
	# vision transformer中最直接的分patch处理image代码
	# 1、将 W*H*C 维度的输入x，压缩成 N*(C*p*p),然后经过linear projection，得到（1,24,24,768），
	# 2、再reshape成(1,576,768),再跟class_embedding(1,1,768)进行concat，再和position_embedding（1,577, 768）进行相加，
	# 3、输出(1, 577, 768),即(batch_size, 有效序列长度, embeded_dim)

	## test code:
	image_size = (384, 384, 3)
	batch_size = 1
	patch_size = 16

	input_layer = keras.layers.Input(shape=image_size, batch_size=batch_size)
	print("input_layer:", input_layer.shape)

	outputs = TransformerInputConv2DLayer(image_size, patch_size)(input_layer)
	print("outputs:", outputs.shape)
	
    """

    def __init__(self, image_size=None, patch_size=None, dropout_rate = 0.1):
        super(TransformerInputConv2DLayer, self).__init__(name="input_preprocessing")
        self.patch_size = patch_size
        
        self.h = image_size[0]
        self.w = image_size[1]
        self.c = image_size[2]
        
        # 有效长度 N， sequence length， n个patch
        self.n = (int)(self.h * self.w / self.patch_size ** 2)
        
        # d_model、patch_dim、embedded_dim
        self.embeded_dim = self.c * self.patch_size ** 2

        # class embedding
        self.class_embedding = self.add_weight("class_embedding", 
        										shape=(1, 1, self.embeded_dim), 
        										trainable=True)
        
        # position embedding
        self.position_embedding = self.add_weight("position_embedding", 
        										  shape=(1, self.n + 1, self.embeded_dim),
                                                  trainable=True)
        
        # 线性映射 linear 本质上相当于 stride等于原图wh的conv过程
        self.linear_projection = keras.layers.Conv2D(self.embeded_dim, self.patch_size, 
        								strides=(self.patch_size, self.patch_size), 
        								padding="valid",
                                        name="linear_projection")


        # drop out
        self.dropout = keras.layers.Dropout(dropout_rate)


    def call(self, inputs):
        # inputs的shape(?, 284, 284, 3)
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = -1
            
        # 1、输入x的shape(?, 384, 384, 3) -> 线性映射后shape (?, 24, 24, 768)
        x = self.linear_projection(inputs)
        
        # 2、得到n、h、w、c, reshape : (1, 24, 24, 768) ->  (1, 576, 768) 
        n, h, w, c = x.shape
        x = tf.reshape(x, [n, h * w, c])

        # 3、class_embedding shape: (1, 1, 768)
        # tf.broadcast_to() 作用：利用广播将原始矩阵成倍增加，广播是使数组具有兼容形状以进行算术运算的过程。
        class_embedding = tf.broadcast_to(self.class_embedding, [batch_size, 1, self.embeded_dim])
        
        # 5、class_embedding(1, 1, 768) [CONCAT] Linear_projection (1, 576, 768) -> (1, 577, 768)
        x = tf.concat([class_embedding, x], axis=1)
        
        # 6、postion_embedding + Concat(Linear, class) -> (1, 577, 768)
        x += self.position_embedding
        
        # 7、drop out
        outputs = self.dropout(x)

        return outputs


# ----------------------------------
# MLPBlock in transformer 
# ----------------------------------
class MLPBlock(keras.layers.Layer):
	def __init__(self, embeded_dim, dropout_rate=0.1):
		super().__init__()
		self.embeded_dim = embeded_dim

		# 4 * embeded_dim = 4 * 768 = 3072 ,参数 4 默认值，base模型 MLP size = 3072
		self.linear_1 = keras.layers.Dense(4 * self.embeded_dim, activation=tf.nn.gelu, name="MLP_linear_1")
		self.dropout = keras.layers.Dropout(dropout_rate)

		self.linear_2 = keras.layers.Dense(embeded_dim, name="MLP_linear_2")



	def call(self, inputs):
		x = self.linear_1(inputs)
		x = self.dropout(x)
		x = self.linear_2(x)
		outputs = self.dropout(x)

		return outputs




# ----------------------------------
# Multi-head-attention
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


# --------------------------------
# MSA 写法 2: Multi-head-attention
# --------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self._droput_rate = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[num_query_features, self.num_heads, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.query_bias = self.add_weight(
                name="query_bias",
                shape=[self.num_heads, self.head_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[num_key_features, self.num_heads, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.key_bias = self.add_weight(
                name="key_bias",
                shape=[self.num_heads, self.head_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[num_value_features, self.num_heads, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.value_bias = self.add_weight(
                name="value_bias",
                shape=[self.num_heads, self.head_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )


        self.projection_kernel = self.add_weight(
            name="out_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="out_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = tf.einsum("...NI , IHO -> ...NHO", query, self.query_kernel) + self.query_bias
        key = tf.einsum("...MI , IHO -> ...MHO", key, self.key_kernel) + self.key_bias
        value = tf.einsum("...MI , IHO -> ...MHO", value, self.value_kernel) + self.value_bias

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
        )

        return config



# ----------------------------------
# transformer encoder 
# ----------------------------------
class TransformerEncoderBlock(keras.layers.Layer):
	def __init__(self, unique_name, embeded_dim, num_heads, dropout_rate=0.1):
		super().__init__(name=unique_name)

		self.embeded_dim = embeded_dim
		self.num_heads = num_heads

		self.head_size = (int)(self.embeded_dim / self.num_heads)

		self.layer_norm_1 = keras.layers.LayerNormalization(name="LN_1")
		# self.multi_head_atttention = MultiHeadAttention(self.head_size, self.num_heads)
		self.multi_head_atttention = MSA(num_heads=self.num_heads)

		self.dropout = keras.layers.Dropout(dropout_rate)
		self.layer_norm_2 = keras.layers.LayerNormalization(name="LN_2")

		self.mlp_layer = MLPBlock(self.embeded_dim)


	def call(self, inputs):
		input_x = inputs
		x = self.layer_norm_1(inputs)
		# x = self.multi_head_atttention([x, x])
		x, _ = self.multi_head_atttention(x)
		x = self.dropout(x)
		x = x + input_x

		y = self.layer_norm_2(x)
		y = self.mlp_layer(y)

		return x + y


# ----------------------------------
# 创建 ViT-16  
# ----------------------------------
class VitBuilder():
    """
     # num_layers=12: base模型。24是larger， 32是huge
     # patch_size=16
    """
    
    def __init__(self, image_size=(384, 384, 3),
                 patch_size=16, 
                 num_heads=12,        # 12个head
                 num_layers=12,       # base
                 
                 num_classes=1000,    # 1000个输出
                 batch_size=1):       # batch size


        self.image_size = image_size
        self.patch_size = patch_size

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embeded_dim = patch_size ** 2 * image_size[2]
        self.batch_size = batch_size

    def build(self):
        
        
        # 1、输入层
        input_layer = keras.layers.Input(shape=self.image_size, 
        								 batch_size=self.batch_size,
        								 name="inputs")
        
        # 2、Linear Projection of Flatten Patches
        x = TransformerInputConv2DLayer(self.image_size, self.patch_size)(input_layer)
        
        # 3、Transforer Encoder (* num_layers)
        for i in range(self.num_layers):
            x = TransformerEncoderBlock(unique_name=f"tranformer_encoder_{i}", 
            							embeded_dim=self.embeded_dim, 
            							num_heads=self.num_heads)(x)
        	
        # 4、LN
        x = keras.layers.LayerNormalization(name='encoder_LN')(x)
        
        # 5、head
        output = keras.layers.Dense(self.num_classes, name='head')(x[:, 0])

        # 6、build model
        vit_model = keras.Model(input_layer, output)

        return vit_model





# ----------------------------------
# build model
# ----------------------------------
model = VitBuilder().build()
model.summary()



