import L8
from pathlib import Path
import numpy as np
import h5py
import Image
import math
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from keras import backend as K
import matplotlib.cm as cm

def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255. / rmse)

#############################################
#1st step:setting path and file
save_dir = Path('.')
img_label=save_dir/'label.jpg'
image_input = save_dir / 'Data10.jpg'
weights_file = save_dir / 'weights1.h5'

#2nd step:input image
image_input = Image.open(str(image_input))#RGB Image
img_input=image_input.convert('YCbCr') #YCbCr Image
img_input = np.asarray(img_input,dtype = np.float32)#array[height,width,channel]


[img_height,img_width,img_channel]=img_input.shape  

input_batch=np.expand_dims(img_input,axis=0)/255.0#you should expand the dimensionality to 4

#3rd step:build model 
deblocking=L8.create_model(img_height,img_width,img_channel)
deblocking.load_weights(str(weights_file))
#plot_model(deblocking, to_file=str(save_dir/'model.png'), show_shapes=True, show_layer_names=True)

############################################
#visualize Layers
'''
get_2nd_layer_output=K.function([deblocking.layers[0].input,K.learning_phase()],[deblocking.layers[36].output])
# output in test mode = 0 train mode=1
second_layer=get_2nd_layer_output([input_batch,0])[0]
print '2nd layer:',second_layer.shape

fig = plt.figure()
plt.suptitle('2nd layer')
for i in range(second_layer.shape[3]):
    plt.subplot(4, second_layer.shape[3]/4, i+1)
    plt.imshow(second_layer[0,:,:,i]*255.0,cmap = cm.Greys_r)
    plt.axis('off')
#show
plt.show()
'''
################################################
#4th step:show comparsion
pre_img = deblocking.predict(input_batch)
print 'predict finished'

pre_img=Image.fromarray((pre_img[0,:,:,:]*255.0).astype('uint8'), mode='YCbCr')
pre_img=pre_img.convert('RGB')
pre_img = np.asarray(pre_img,dtype = np.float32)


img_label = Image.open(str(img_label))
img_label = np.asarray(img_label,dtype = np.float32)

img_input_test = np.asarray(image_input,dtype = np.float32)

p_ori=psnr(img_input_test, img_label)
p_pred=psnr(pre_img,img_label)
s_ori =ssim(img_input_test, img_label,multichannel=True)
s_pred=ssim(pre_img,img_label,multichannel=True)

# show the figure
fig = plt.figure()
plt.suptitle("PSNR_ori: %.2f, SSIM_ori: %.2f,PSNR_pred: %.2f,/ SSIM_pred: %.2f" % (p_ori, s_ori,p_pred,s_pred))
ax = fig.add_subplot(1, 3, 1)
plt.imshow(img_label/255.0)
plt.title('origin')
plt.axis("off")

ax = fig.add_subplot(1, 3, 2)
plt.imshow(pre_img/255.0)
plt.title('output')
plt.axis("off")

ax = fig.add_subplot(1, 3, 3)
plt.imshow(img_input_test/255.0)
plt.title('input')
plt.axis("off")
# show the images
plt.show()








