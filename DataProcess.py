import h5py
import numpy
import math
import Image
import os

'''
this code can't compress images,
you must cmmpress image ahead.And put compressed images in a folder and labels in another floder
'''
DATA_PATH = '/home/lhj/wjq/deblock_keras/Data/'
LABEL_PATH = '/home/lhj/wjq/deblock_keras/Label/'
patch_size=40
stride=10

def prepare_data(path):
    names = os.listdir(path)
    names = sorted(names)#Attention
    nums = names.__len__()

    data = []#list
    for i in range(nums):#0 1 2
        name=path + names[i]
        img = Image.open(name)#jpg or png.Attention
        img=numpy.asarray(img)#dtype = numpy.int
        img=img[:,:,0:3]#for jpg [R,G,B],in case for png [R,G,B,A]
        shape=img.shape
        print 'img.shape:',shape
        row_num = ((shape[0] - patch_size) / stride)+1
        col_num =((shape[1] - patch_size) / stride)+1
        row_shift = (shape[0] - ((row_num - 1) * stride + patch_size)) / 2
        col_shift = (shape[1] - ((col_num - 1) * stride + patch_size)) / 2
        
        for x in range(row_num):
            x_start = row_shift + (x) * stride
            x_end = row_shift + (x) * stride + patch_size
            for y in range(col_num):
                y_start = col_shift + (y) * stride
                y_end = col_shift + (y) * stride + patch_size
                sub_img = img[x_start:x_end, y_start:y_end, :]
                data.append(sub_img)

    data = numpy.array(data)  #list has no shape          
    return data

def write_hdf5(data, label,output_filename):
    """
    This function is used to save image data or its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """
    x = data.astype(numpy.int)
    y = label.astype(numpy.int)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


if __name__ == "__main__":
    data= prepare_data(DATA_PATH)
    label = prepare_data(LABEL_PATH)
    write_hdf5(data, label, '/home/lhj/wjq/deblock_keras/DATA.h5')

