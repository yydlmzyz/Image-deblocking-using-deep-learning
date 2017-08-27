from keras.models import Model
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import optimizers
import keras.backend as K
import numpy
import math

def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255
def ssim(y_true, y_pred):#may be wrong
    K1 = 0.04
    K2 = 0.06
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5
    L =  255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim



def conv_factory(x, nb_filter,kernel_size):
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(x)

def denseblock(x, nb_layers, nb_filter, growth_rate):
    for i in range(nb_layers):
        if i<=2:
            kernel_size=3
        elif i==3 or i==5:
            kernel_size=1
        else:
            kernel_size=5
        merge_tensor = conv_factory(x, growth_rate,kernel_size)
        #x = merge([merge_tensor, x], mode='concat', concat_axis=-1)
        x=concatenate([merge_tensor, x],axis=-1)
    return x

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    x = Conv2D(16, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
    x= denseblock(x, 6, 16, 16)

    op=Conv2D(img_channel, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    deblocking =Model(inputs=ip,outputs= op)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim])
    return deblocking
