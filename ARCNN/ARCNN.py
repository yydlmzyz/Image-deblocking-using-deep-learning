from keras.models import Model
from keras.layers import Input, Conv2D
from keras import optimizers
from keras import backend as K
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
    
def create_model(img_height,img_width,img_channel):
    input_img = Input(shape=(img_height,img_width,img_channel))
    feature1 = Conv2D(64,(9,9),padding='same',activation='relu')(input_img)
    enhance_feature = Conv2D(32,(7,7),padding='same',activation='relu')(feature1)
    conv3 = Conv2D(16,(1,1),padding='same',activation='relu')(enhance_feature)
    output = Conv2D(img_channel,(5,5),padding='same',activation='relu')(conv3)
    deblocking =Model(input_img,output)

    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim])
    return deblocking




