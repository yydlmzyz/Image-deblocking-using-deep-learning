import os
#from PIL import Image
import cv2
import numpy
from skimage.measure import compare_ssim,compare_psnr
import mymodels

class ImageDataset:
    def __init__(self, root_dir):

        self.input_dir = os.path.join(root_dir,'input')
        self.label_dir = os.path.join(root_dir,'label')

    def len(self):
        return os.listdir(self.input_dir).__len__()

    def getitem(self, idx):
        input_names = sorted(os.listdir(self.input_dir))
        label_names = sorted(os.listdir(self.label_dir))

        input_name = os.path.join(self.input_dir,input_names[idx])
        #input_image = io.imread(input_name)#skimage io get numpy
        #input_image =Image.open(input_name)#Image get jpg
        input_image=cv2.imread(input_name)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


        label_name = os.path.join(self.label_dir,label_names[idx])
        #label_image = io.imread(label_name)
        #label_image =Image.open(label_name)
        label_image=cv2.imread(label_name)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

        sample = {'input_image': input_image, 'label_image': label_image, 'name': input_names[idx]}

        return sample


def checkpoint(name,psnr1,psnr2,ssim1,ssim2):
    print('{},psnr:{:.4f}->{:.4f},ssim:{:.4f}->{:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2))
    #write to text
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write(('{} {:.4f}->{:.4f} {:.4f}->{:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2))+'\r\n')
    output.close()


def test():

    avg_psnr1, avg_ssim1, avg_psnr2, avg_ssim2 = 0,0,0,0

    for i in range(mydataset.len()):
        sample=mydataset.getitem(i)
        input_image,label_image,name=sample['input_image'],sample['label_image'],sample['name']
 
        #if use PIL
        #inputs = numpy.asarray(input_image)
        #label = numpy.asarray(label_image)

        #if use cv2
        inputs = input_image
        label = label_image

        [img_height,img_width,img_channels]=inputs.shape

        inputs_batch=numpy.expand_dims(inputs,axis=0)/255.0

        #build model and load weights

        model=mymodels.DenseNet_shallow(img_height,img_width,img_channels)
        model=model.create_model()
        model.load_weights(weights_file)

        #predict
        output = model.predict(inputs_batch)
        #restrict
        output[output[:] > 1] = 1
        output[output[:] < 0] = 0

        output = (output[0,:,:,:]*255.0).astype('uint8')
        #output_image=Image.fromarray(output, mode='RGB')
        output_image=output

        #calculate psnr&ssim
        psnr1 =compare_psnr(inputs, label)
        psnr2 =compare_psnr(output, label)
        ssim1=compare_ssim(inputs, label, multichannel=True)
        ssim2=compare_ssim(output, label, multichannel=True)

        avg_ssim1 += ssim1
        avg_psnr1 += psnr1
        avg_ssim2 += ssim2
        avg_psnr2 += psnr2

        #save output and record
        checkpoint(name,psnr1,psnr2,ssim1,ssim2)
        #output_image.save(os.path.join(Image_folder,'output','{}.jpg'.format(name[:-4])))
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(Image_folder,'output','{}.jpg'.format(name[:-4])),output_image)

    #print and save
    avg_psnr1 = avg_psnr1/mydataset.len()
    avg_ssim1 = avg_ssim1/mydataset.len()
    avg_psnr2 = avg_psnr2/mydataset.len()
    avg_ssim2 = avg_ssim2/mydataset.len()
    print('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2))
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2)+'\r\n')
    output.close()


#------------------------------------------------------------------
#set path
root_dir=os.getcwd()
Image_folder=os.path.join(root_dir,'test')
weights_file=os.path.join(root_dir,'DenseNet_shallow.h5')
if not os.path.exists(os.path.join(Image_folder, 'output')):
    os.mkdir(os.path.join(Image_folder, 'output'))


#set dataset
mydataset = ImageDataset(root_dir=Image_folder)


if __name__=='__main__':
    test()
