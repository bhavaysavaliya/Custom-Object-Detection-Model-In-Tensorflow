import tensorflow as tf
from keras.layers import Conv2D,BatchNormalization,Lambda,ZeroPadding2D,MaxPooling2D,Layer
from keras.activations import swish

class YoloSPPFLayer(Layer):
    def __init__(self, c0,conv1_config=None,conv_final_config=None,**kwargs):
        super(YoloSPPFLayer, self).__init__(**kwargs)
        if conv1_config is None:
            self.conv1 = YoloConvLayer(k=1, s=1, p=0, c=c0)
        else:
            self.conv1 = YoloConvLayer.from_config(conv1_config)
        
        self.c0 = c0
        self.maxpool1 = MaxPooling2D()
        self.maxpool2 = MaxPooling2D()
        # self.maxpool3 = MaxPooling2D()
        if conv_final_config is None:
            self.conv_final = YoloConvLayer(k=1, s=1, p=0, c=c0)
        else:
            self.conv_final = YoloConvLayer.from_config(conv_final_config)


    def call(self, prev):
        op = self.conv1(prev)
        x1 = self.maxpool1(op)
        x2 = self.maxpool2(x1)
        # x3 = self.maxpool3(x2)
        
        _, bp, cp, dp = op.shape
        _, b1, c1, d1 = x1.shape
        _, b2, c2, d2 = x2.shape

        # _, b3, c3, _ = x3.shape
        b3,c3=b2,c2
        prev_reshape = tf.reshape(prev, shape=[-1, b3, c3, int((bp * cp * dp) / (b3 * c3))])
        x1_reshape = tf.reshape(x1, shape=[-1, b3, c3, int((b1 * c1 * d1) / (b3 * c3))])
        x2_reshape = tf.reshape(x2, shape=[-1, b3, c3, int((b2 * c2 * d2) / (b3 * c3))])
        
        concatenated_tensor = tf.concat([prev_reshape, x1_reshape, x2_reshape
                                         #,x3
                                         ], axis=3)
        final_op = self.conv_final(concatenated_tensor)
        
        return final_op
    
    def get_config(self):
        base_config = super(YoloSPPFLayer, self).get_config()
        conv1_config = None
        conv_final_config = None
        if self.conv1 is not None:
            conv1_config = self.conv1.get_config()
        if self.conv_final is not None:
            conv_final_config = self.conv_final.get_config()
        return dict(list(base_config.items()) + [
            ('c0', self.c0), ('conv1_config', conv1_config), ('conv_final_config', conv_final_config)
        ])

    @classmethod
    def from_config(cls, config):
        c0 = config.pop('c0')
        conv1_config = config.pop('conv1_config', None)
        conv_final_config = config.pop('conv_final_config', None)
        return cls(c0=c0, conv1_config=conv1_config, conv_final_config=conv_final_config, **config)
    
    def build(self, input_shape):
        super(YoloSPPFLayer, self).build(input_shape)







class YoloConvLayer(Layer):
    def __init__(self, k, s, p, c, **kwargs):
        super(YoloConvLayer, self).__init__(**kwargs)
        self.k = k
        self.s = s
        self.p = p
        self.c = c

        self.padding_layer = ZeroPadding2D(padding=(self.p, self.p))
        self.conv_layer = Conv2D(kernel_size=self.k, filters=self.c, strides=self.s, padding="valid",
                                 activation=tf.nn.leaky_relu)
        self.batch_norm = BatchNormalization()
        self.silu_activation = Lambda(lambda x: swish(x))

    def call(self, inputs):
        x_padded = self.padding_layer(inputs)
        x_conv = self.conv_layer(x_padded)
        x_norm = self.batch_norm(x_conv)
        x_out = self.silu_activation(x_norm)
        return x_out

    def get_config(self):
        config = super(YoloConvLayer, self).get_config()
        config.update({
            'k': self.k,
            's': self.s,
            'p': self.p,
            'c': self.c
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super(YoloConvLayer, self).build(input_shape)









class YoloBottleneckLayer(Layer):
    def __init__(self, d, shortcut=True, conv1_config=None, conv2_config=None, **kwargs):
        super(YoloBottleneckLayer, self).__init__(**kwargs)
        self.d = d
        self.shortcut = shortcut
        if conv1_config is None:
            self.conv1 = YoloConvLayer(k=1, s=1, p=0, c=d)
        else:
            self.conv1 = YoloConvLayer.from_config(conv1_config)
        if conv2_config is None:
            self.conv2 = YoloConvLayer(k=1, s=1, p=0, c=d)
        else:
            self.conv2 = YoloConvLayer.from_config(conv2_config)

    def call(self, prev):
        x1 = self.conv1(prev)
        x2 = self.conv2(x1)
        
        if self.shortcut:
            op = prev + x2
        else:
            op = x2
        return op
    
    def get_config(self):
        base_config = super(YoloBottleneckLayer, self).get_config()
        conv1_config = None
        conv2_config = None
        if self.conv1 is not None:
            conv1_config = self.conv1.get_config()
        if self.conv2 is not None:
            conv2_config = self.conv2.get_config()
        return dict(list(base_config.items()) + [
            ('d', self.d), ('shortcut', self.shortcut), ('conv1_config', conv1_config), ('conv2_config', conv2_config)
        ])

    @classmethod
    def from_config(cls, config):
        d = config.pop('d')
        shortcut = config.pop('shortcut')
        conv1_config = config.pop('conv1_config', None)
        conv2_config = config.pop('conv2_config', None)
        return cls(d=d, shortcut=shortcut, conv1_config=conv1_config, conv2_config=conv2_config, **config)
    
    def build(self, input_shape):
        super(YoloBottleneckLayer, self).build(input_shape)






class YoloC2FLayer(Layer):
    def __init__(self, n, c, boolean_lambda, conv1_config=None, conv_final_config=None, bottleneck_configs=None, **kwargs):
        super(YoloC2FLayer, self).__init__(**kwargs)
        self.boolean_lambda = boolean_lambda
        self.n = n
        self.c = c
        if conv1_config is None:
            self.conv1 = YoloConvLayer(k=1, s=1, p=0, c=c)
        else:
            self.conv1 = YoloConvLayer.from_config(conv1_config)
        if conv_final_config is None:
            self.conv_final = YoloConvLayer(k=1, s=1, p=0, c=c)
        else:
            self.conv_final = YoloConvLayer.from_config(conv_final_config)
        if bottleneck_configs is None:
            self.bottleneck_layers = [
                YoloBottleneckLayer(d=int(c / 2), shortcut=self.boolean_lambda) for i in range(n)
            ]
        else:
            self.bottleneck_layers = [
                YoloBottleneckLayer.from_config(config) for config in bottleneck_configs
            ]

    def call(self, prev):
        x1 = self.conv1(prev)
        new_cout = int(self.c / 2)
        output_1 = x1[:, :, :, :new_cout]
        output_2 = x1[:, :, :, new_cout:]
        concat_layer = tf.concat([output_1, output_2], axis=3)
        loop = output_2

        for layer in self.bottleneck_layers:
            loop = layer(loop)
            concat_layer = tf.concat([concat_layer, loop], axis=3)

        op = self.conv_final(concat_layer)
        return op
    
    def get_config(self):
        base_config = super(YoloC2FLayer, self).get_config()
        conv1_config = None
        conv_final_config = None
        bottleneck_configs = None
        if self.conv1 is not None:
            conv1_config = self.conv1.get_config()
        if self.conv_final is not None:
            conv_final_config = self.conv_final.get_config()
        if self.bottleneck_layers is not None:
            bottleneck_configs = [layer.get_config() for layer in self.bottleneck_layers]
        return dict(list(base_config.items()) + [
            ('n', self.n), ('c', self.c), ('boolean_lambda', self.boolean_lambda),
            ('conv1_config', conv1_config), ('conv_final_config', conv_final_config),
            ('bottleneck_configs', bottleneck_configs)
        ])
    
    @classmethod
    def from_config(cls, config):
        n = config.pop('n')
        c = config.pop('c')
        boolean_lambda = config.pop('boolean_lambda')
        conv1_config = config.pop('conv1_config', None)
        conv_final_config = config.pop('conv_final_config', None)
        bottleneck_configs = config.pop('bottleneck_configs', None)
        return cls(n=n, c=c, boolean_lambda=boolean_lambda, conv1_config=conv1_config,
                   conv_final_config=conv_final_config, bottleneck_configs=bottleneck_configs, **config)
    
    
    def build(self, input_shape):
        super(YoloC2FLayer, self).build(input_shape)