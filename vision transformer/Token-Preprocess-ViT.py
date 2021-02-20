
# ----------------------------------
# TransformerInputConv2DLayer 
# ----------------------------------
class TransformerInputConv2DLayer(tf.keras.layers.Layer):
    """
	# vision transformer中最直接的分patch处理image代码
	# 1、将 W*H*C 维度的输入x，压缩成 N*(C*p*p),然后经过linear projection，得到（1,24,24,768），
	# 2、再reshape成(1,576,768),再跟class_embedding(1,1,768)进行concat，再和position_embedding（1,577, 768）进行相加，
	# 3、输出(1, 577, 768),即(batch_size, 有效序列长度, embeded_dim)
	
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
