from keras import Model
from keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization, Activation, Lambda
from keras.layers import GlobalAveragePooling3D, Concatenate, Dense

def Conv_BN_relu(inputs, num_filter, kernel_size=(3,3,3), strides=(1,1,1), padding='same', use_bias=False):
    x = Conv3D(num_filter, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_model(input_shape, nb_class):
    inp = Input(shape=input_shape)


    x1 = Conv_BN_relu(inp, 32, kernel_size=(3,7,7), strides=(1,2,2))
    x1 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x1)
    x1 = Conv_BN_relu(x1, 64)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)
    x1 = Conv_BN_relu(x1, 128)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)
    x1 = Conv_BN_relu(x1, 128)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x1)


    x = Conv_BN_relu(x1, 256, kernel_size=(1,3,3))
    x = GlobalAveragePooling3D()(x)
    out = Dense(nb_class, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out)

# model = build_model((16,224,224,3), 1)
# model.summary(line_length=150)

