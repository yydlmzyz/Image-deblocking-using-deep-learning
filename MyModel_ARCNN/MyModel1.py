from keras.models import Model
from keras.layers import Input, Conv2D
from keras import backend as K
import numpy
import math


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))
    mse=K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def create_model(img_height,img_width,img_channel):
    input_img = Input(shape=(img_height,img_width,img_channel))
    feature1 = Conv2D(64,(9,9),padding='same',activation='relu')(input_img)
    enhance_feature = Conv2D(32,(7,7),padding='same',activation='relu')(feature1)
    conv3 = Conv2D(16,(1,1),padding='same',activation='relu')(enhance_feature)
    output = Conv2D(img_channel,(5,5),padding='same',activation='relu')(conv3)
    deblocking =Model(input_img,output)

    deblocking.compile(optimizer='adadelta',loss='mean_squared_error', metrics=[psnr])
    return deblocking




