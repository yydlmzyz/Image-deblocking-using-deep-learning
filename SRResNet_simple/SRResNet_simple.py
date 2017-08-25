from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers
import numpy
import math

def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255
def ssim(y_true, y_pred):#may be wrong
    """structural similarity measurement system."""
    K1 = 0.04
    K2 = 0.06
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5
    ## L, number of pixels, C1, C2, two constants
    L =  255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim

def residual_block(input,filters,kernel_size):
    conv_1 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(input)
    norm_1 = BatchNormalization(axis=-1)(conv_1)
    relu_1 = LeakyReLU(alpha=0.25)(norm_1)
    conv_2 = Conv2D(filters, (kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(relu_1)
    norm_2 = BatchNormalization(axis=-1)(conv_2)
    return add([input, norm_2])


def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    L_1 = Conv2D(64, (9, 9), padding='same', activation='linear', kernel_initializer='glorot_uniform')(ip)
    L_1 = LeakyReLU(alpha=0.25)(L_1)
    L_2=L_1
    for i in range(3):
        L_2 = residual_block(L_2, 64,3)

    L_3 = Conv2D(64, (3, 3), padding='same',kernel_initializer='glorot_uniform')(L_2)
    L_3 = BatchNormalization(axis=-1)(L_3)
    L_3 = add([L_1,L_3])
    L_4= Conv2D(128, (1, 1), padding='same',kernel_initializer='glorot_uniform')(L_3)
    op = Conv2D(img_channel, (9, 9),padding='same', activation='tanh', kernel_initializer='glorot_uniform')(L_4)

    deblocking =Model(inputs=ip,outputs= op)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim])
    return deblocking

