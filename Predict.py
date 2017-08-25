'''
you put images in /ImageIuput/, then run it, you will get the processed images in /ImageOutput/
'''
import MyModel_v1
from pathlib import Path
import numpy
import h5py
import Image
import numpy
import os

#1st step:setting path and file
save_dir = Path('.')
weights_file = save_dir / 'WeightsHigh.h5'
Image_dir = '/home/lhj/wjq/deblock_keras/Bubble/high/'
Image_output=save_dir / 'OutputBubbleHigh'
path=str(Image_dir)

if Image_output.exists():
    print 'folder esxited'
else:
    Image_output.mkdir()


names = os.listdir(path)
names = sorted(names)
nums = names.__len__()

data = []
for i in range(nums):
    name=path + names[i]
    img = Image.open(name)
    img =img.convert('YCbCr') 
    img=numpy.asarray(img)#dtype = numpy.int
    #img=img[:,:,0:3]
    print img.shape#check
    data.append(img)

data = numpy.array(data)
data=data/255.0
#check:
print 'data.shape:', data.shape
[count,img_height,img_width,img_channel]=data.shape

#3rd step:predict images
deblocking_model=MyModel_v1.create_model(img_height,img_width,img_channel)
deblocking_model.load_weights(str(weights_file))

input = numpy.zeros((1,img_height,img_width,img_channel), dtype = numpy.float32)

for i in range(count):
    input[0,:,:,:] = data[i,:,:,:]
    pre_img = deblocking_model.predict(input)
    output=pre_img[0,:,:,:]*255.0
    output=Image.fromarray(output.astype('uint8'), mode='YCbCr')
    output=output.convert('RGB')
    output.save(str(Image_output/'%d.jpg') %i)
