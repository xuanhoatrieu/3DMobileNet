from keras import Model
from keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization, Activation, Lambda
from keras.layers import GlobalAveragePooling3D, Concatenate, Dense
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2

def Conv_BN_relu(inputs, num_filter, kernel_size=(3,3,3), strides=(1,1,1), padding='same', use_bias=False):
    x = Conv3D(num_filter, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(input_shape_1, input_shape_2, nb_class):
    inp1 = Input(shape=input_shape_1)
    inp2 = Input(shape=input_shape_2)

    x1 = Conv_BN_relu(inp1, 32, kernel_size=(3,7,7), strides=(1,2,2))
    x1 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x1)
    x1 = Conv_BN_relu(x1, 64)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)
    x1 = Conv_BN_relu(x1, 128)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)
    x1 = Conv_BN_relu(x1, 128)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)

    x2 = Conv_BN_relu(inp2, 32, kernel_size=(3,7,7), strides=(1,2,2))
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x2)
    x2 = Conv_BN_relu(x2, 64)
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x2)
    x2 = Conv_BN_relu(x2, 128)
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x2)
    x2 = Conv_BN_relu(x2, 128)
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x2)

    x = Concatenate()([x1, x2])
    x = Conv_BN_relu(x, 256, kernel_size=(1,3,3))
    x = GlobalAveragePooling3D()(x)
    out = Dense(nb_class, activation='sigmoid')(x)
    return Model(inputs=[inp1, inp2], outputs=out)

# model = build_model((8,224,224,3), (8,224,224,1), 1)
# model.summary(line_length=150)

