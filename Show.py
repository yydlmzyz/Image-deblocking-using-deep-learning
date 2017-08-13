import MyModel
from pathlib import Path
import numpy as np
import h5py
import Image
import math
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
#from keras.utils import plot_model

#1st step:setting path and file
save_dir = Path('.')
img_ori_file=save_dir/'img_ori.jpg'
img_file = save_dir / 'img_com10.jpg'
weights_file = save_dir / 'weights_10.h5'

#2nd step:input image
img_com = Image.open(str(img_file))
img = np.asarray(img_com,dtype = np.float32)
print 'img_shape:', img.shape#check
img=img[:,:,0:3]
img_ori = Image.open(str(img_ori_file))
img_ori = np.asarray(img_ori,dtype = np.float32)
img_ori=img_ori[:,:,0:3]
[img_height,img_width,img_channel]=img.shape  
print 'img_shape:', img.shape#check

img_batch = np.zeros((1,img_height,img_width,img_channel), dtype = np.float32)
img_batch[0,:,:,:] = img/255.0
img_ori_batch = np.zeros((1,img_height,img_width,img_channel), dtype = np.float32)
img_ori_batch[0,:,:,:] = img_ori/255.0
print 'img_batch_shape:', img_batch.shape


#3rd step:build model 
deblocking=MyModel.create_model(img_height,img_width,img_channel)
deblocking.load_weights(str(weights_file))
#plot_model(deblocking, to_file=str(save_dir/'model.png'), show_shapes=True, show_layer_names=True)
#generate imgs
pre_img = deblocking.predict(img_batch)
print 'pre_img.shape:', pre_img.shape


#4th step:show comparsion
def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

p_ori=psnr(img, img_ori)
p_pred=psnr(pre_img[0,:,:,:]*255.0, img_ori)
s_ori =ssim(img,img_ori,multichannel=True)
s_pred=ssim(pre_img[0,:,:,:]*255.0,img_ori,multichannel=True)

# show the figure
fig = plt.figure()
plt.suptitle("PSNR_ori: %.2f, SSIM_ori: %.2f,PSNR_pred: %.2f,/ SSIM_pred: %.2f" % (p_ori, s_ori,p_pred,s_pred))

ax = fig.add_subplot(1, 3, 1)
plt.imshow(img_ori_batch[0,:,:,:])
plt.title('origin')
plt.axis("off")

ax = fig.add_subplot(1, 3, 2)
plt.imshow(pre_img[0,:,:,:])
plt.title('output')
plt.axis("off")

ax = fig.add_subplot(1, 3, 3)
plt.imshow(img_batch[0,:,:,:])
plt.title('input')
plt.axis("off")
# show the images
plt.show()








