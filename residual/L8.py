from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras import optimizers
import numpy
import math

def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(255. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255
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

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    L1 = Conv2D(32, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
    L2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L1)
    L3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L2)
    L4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L3)
    L4=concatenate([L4,L1],axis=-1)
    L5 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L4)
    L6 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L5)
    L6=concatenate([L6,L1],axis=-1)
    L7 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L6)
    L8 = Conv2D(img_channel, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L7)
    deblocking =Model(inputs=ip,outputs= L8)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim])
    return deblocking

'''
from keras.utils import plot_model
plot_model(deblocking, to_file='model1.png', show_shapes=True, show_layer_names=True)
'''




