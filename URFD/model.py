from keras import Model
from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Activation, ReLU, GlobalAveragePooling3D
from keras.layers import Dense, GlobalMaxPooling3D
from DepthwiseConv3D import DepthwiseConv3D

def _conv_block(inputs, filters, alpha, kernel=(3,3,3), strides=(1,1,1)):
    filters = int(filters * alpha)
    x = Conv3D(filters, kernel,
                      padding='same',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(inputs)
    x = BatchNormalization()(x)
    return ReLU(6., name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1,1,1)):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = DepthwiseConv3D(kernel_size=(3,3,3), strides=strides, padding='same', depth_multiplier=depth_multiplier,
                        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = Conv3D(pointwise_conv_filters, kernel_size=(1,1,1), padding='same', strides=(1,1,1) ,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x

def MobileNet3D(input_shape=(16,224,224,3), nb_classes=1, pooling='avg', alpha=0.25, depth_multiplier=1, activation='sigmoid'):
    img_input = Input(shape=input_shape)
    x = _conv_block(img_input, 32, alpha, strides=(1, 2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2,2,2))
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,strides=(2,2,2))
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2,2,2))
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,strides=(2,2,2))
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier)

    if pooling == 'avg':
        x = GlobalAveragePooling3D()(x)
    else:
        x = GlobalMaxPooling3D()(x)
    out = Dense(nb_classes, activation=activation)(x)
    return Model(inputs=img_input, outputs=out)

# model = MobileNet3D()
# model.summary(line_length=150)
