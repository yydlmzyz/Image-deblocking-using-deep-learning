import h5py
import numpy
import math
import Image
import os

'''
this code can't compress images,
you must cmmpress image by matlab or other thing and put compressed images in a folder and labels in a floder
'''
DATA_PATH = '/home/lhj/wjq/deblock_keras/TrainDataBSD/'
LABEL_PATH = '/home/lhj/wjq/deblock_keras/TrainLabelBSD/'
patch_size=42
stride=16
def mse(A,B):
	return numpy.mean((A - B) ** 2.)

def prepare(Datapath,Labelpath):
    namesdata = os.listdir(Datapath)
    nameslabel = os.listdir(Labelpath)
    namesdata = sorted(namesdata)#Attention
    nameslabel=sorted(nameslabel)
    nums = namesdata.__len__()

    data = []
    label = []
    for i in range(nums):
        namedata=Datapath + namesdata[i]
        namelabel=Labelpath+nameslabel[i]
        imgdata = Image.open(namedata)
        imglabel=Image.open(namelabel)
        imgdata =imgdata.convert('YCbCr')
        imglabel=imglabel.convert('YCbCr')
        imgdata=numpy.asarray(imgdata,dtype = numpy.float32)
        imglabel=numpy.asarray(imglabel,dtype = numpy.float32)
        #imglabel=imglabel-imgdata#Attention
       
        shape=imgdata.shape
        print 'img.shape:',shape
        row_num = ((shape[0] - patch_size) / stride)+1
        col_num =((shape[1] - patch_size) / stride)+1
        row_shift = (shape[0] - ((row_num - 1) * stride + patch_size)) / 2
        col_shift = (shape[1] - ((col_num - 1) * stride + patch_size)) / 2
        print 'row_num:',row_num,'col_num:',col_num
        
        for x in range(row_num):
            x_start = row_shift + (x) * stride
            x_end = row_shift + (x) * stride + patch_size
            for y in range(col_num):
                y_start = col_shift + (y) * stride
                y_end = col_shift + (y) * stride + patch_size
                sub_img_data = imgdata[x_start:x_end, y_start:y_end, :]
                sub_img_label=imglabel[x_start:x_end, y_start:y_end, :]
                MSE=mse(sub_img_label,sub_img_data)
                if MSE>=50: 
                    sub_img_label=sub_img_label-sub_img_data
                    data.append(sub_img_data)
                    label.append(sub_img_label)

    data = numpy.array(data)  #list has no shape
    label=numpy.array(label)
    print 'data.shape:',data.shape          
    return data,label

def write_hdf5(data, label,output_filename):
    """
    This function is used to save image data or its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """
    x = data.astype(numpy.float32)
    y = label.astype(numpy.float32)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


if __name__ == "__main__":
    data,label= prepare(DATA_PATH,LABEL_PATH)
    write_hdf5(data, label, '/home/lhj/wjq/deblock_keras/TrainDataBSDr.h5')

