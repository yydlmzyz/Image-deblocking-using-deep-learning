'''
this version can only process images with same shape!
you put images with same shape in /ImageIuput/, then run it, you will get the processed images in /ImageOutput/
'''
import MyModel
from pathlib import Path
import numpy
import h5py
import Image
import numpy


#1st step:setting path and file
save_dir = Path('.')
weights_file = save_dir / 'weights_HQ.hdf5'
Image_dir = save_dir / 'InputHQ/'
Image_output=save_dir / 'HQOutput'

if Image_output.exists():
    print 'folder esxited'
else:
    Image_output.mkdir()


#2nd step: transfer images to array
data=[]
for image in (Image_dir).glob('*'):
    img = Image.open(image)
    img=numpy.asarray(img)#dtype = numpy.int
    img=img[:,:,0:3]
    print img.shape#check
    data.append(img)
    
data = numpy.array(data)
data=data/255.0
#check:
print 'data.shape:', data.shape
[count,img_height,img_width,img_channel]=data.shape


#3rd step:predict images and save
deblocking_model=MyModel.create_model(img_height,img_width,img_channel)
deblocking_model.load_weights(str(weights_file))

input = numpy.zeros((1,img_height,img_width,img_channel), dtype = numpy.float32)

for i in range(count):
    input[0,:,:,:] = data[i,:,:,:]
    output = deblocking_model.predict(input)
    output=output[0,:,:,:]*255.0
    output=Image.fromarray(output.astype('uint8'), mode='RGB')
    output.save(str(Image_output/'%d.jpg') %i)

