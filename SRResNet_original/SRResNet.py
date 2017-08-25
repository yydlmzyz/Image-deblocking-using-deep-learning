from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from Metrics import psnr,ssim
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


def upscale(input,filters,kernel_size):#128 64 64
    conv_1=Conv2D(filters,(kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(x)
    upscale_1=UpSampling2D(size=(2, 2))(conv_1)
    relu_1 = LeakyReLU(alpha=0.25)(upscale_1)
    conv_2=Conv2D(filters,(kernel_size, kernel_size), padding='same',kernel_initializer='glorot_uniform')(relu_1)
    upscale_2=UpSampling2D(size=(2, 2))(conv_2)
    relu_2 = LeakyReLU(alpha=0.25)(upscale_2)
    return relu_2


def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    x_1 = Conv2D(64, (9, 9), padding='same', activation='linear', kernel_initializer='glorot_uniform')(ip)
    x_1 = LeakyReLU(alpha=0.25)(x_1)
    x=x_1
    for i in range(5):#or 15
        x = residual_block(x, 64,3)

    x = Conv2D(64, (3, 3), padding='same',kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = add([x_1,x])

    x=upscale(x)
    op = Conv2D(img_channel, (9, 9),padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)

    deblocking =Model(inputs=ip,outputs= op)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim])
    return deblocking

from keras.utils import plot_model
plot_model(deblocking, to_file='model.png', show_shapes=True, show_layer_names=True)


'''
from scipy.ndimage import gaussian_filter
def ssim(y_true, y_pred):
    img1=y_true, img2=y_pred
    sd=1.5
    C1=0.01**2
    C2=0.03**2
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2
    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)

'''