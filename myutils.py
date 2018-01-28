from keras import backend as K
import numpy


def psnr(y_true, y_pred):
    mse = K.mean(K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2)))
    return 10 * K.log(1. / mse) / numpy.log(10)

#SSIM is wrong!
def ssim(y_true, y_pred):
    """structural similarity measurement system."""
    K1 = 0.01
    K2 = 0.03
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5
    L =  1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim
