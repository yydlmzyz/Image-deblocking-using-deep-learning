from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy
import math


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def residual_block(input,filters,kernel_size):
    conv_1 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(input)
    norm_1 = BatchNormalization(axis=-1)(conv_1)
    relu_1 = LeakyReLU(alpha=0.25)(norm_1)
    conv_2 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(relu_1)
    norm_2 = BatchNormalization(axis=-1)(conv_2)
    return add([input, norm_2])

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    x_1 = Conv2D(64, (9, 9), padding='same', activation='linear', kernel_initializer='glorot_uniform')(ip)
    x_1 = BatchNormalization(axis= -1)(x_1)
    x_1 = LeakyReLU(alpha=0.25)(x_1)
    x=x_1
    for i in range(5):
        x = residual_block(x, 64,3)
    x = Conv2D(64, (3, 3), padding='same',kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = add([x_1,x])
    x = Conv2D(16,(1, 1),padding='same',activation='relu')(x)
    op = Conv2D(img_channel, (5, 5),padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)

    deblocking =Model(inputs=ip,outputs= op)
    deblocking.compile(optimizer='adadelta',loss='mean_squared_error', metrics=[psnr])
    return deblocking


#plot_model(deblocking, to_file='model.png', show_shapes=True, show_layer_names=True)





