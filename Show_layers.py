import L8
from pathlib import Path
import numpy as np
import h5py
import Image
import math
import matplotlib.pyplot as plt
from keras import backend as K
import matplotlib.cm as cm

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

#4th step:Show
for i in range(7):#7 means only the previous 7 layers will be show 
    get_layer_output=K.function([deblocking.layers[0].input,K.learning_phase()],[deblocking.layers[i+1].output])
    
    layer=get_layer_output([input_batch,0])[0]# output in test mode = 0 train mode=1
    print 'layer:',i,layer.shape

    #plot all channels output,
    fig = plt.figure()
    plt.suptitle('%d layer'%(i+1))
    for j in range(layer.shape[3]):
        plt.subplot(4, layer.shape[3]/4, j+1)
        plt.imshow(layer[0,:,:,j]*255.0,cmap = cm.Greys_r)
        plt.axis('off')
    plt.show()

    #only plot the channel 0 2 4 6 output
    fig = plt.figure()
    plt.suptitle('%d layer'%(i+1))
    for k in range(4):
        plt.subplot(1,4,k+1)
        plt.imshow(layer[0,:,:,k*2]*255.0,cmap = cm.Greys_r)
        plt.axis('off')
    plt.show()










