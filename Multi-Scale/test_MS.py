import os
#from PIL import Image
import cv2
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import MS_model
import myutils
import msssim


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.input_dir = os.path.join(root_dir,'output_qp_42_no_sao','320x480')
        self.label_dir = os.path.join(root_dir,'images_png','320x480')
        self.transform = transform

    def __len__(self):
        return os.listdir(self.input_dir).__len__()

    def __getitem__(self, idx):
        input_names = sorted(os.listdir(self.input_dir))
        label_names = sorted(os.listdir(self.label_dir))

        input_name = os.path.join(self.input_dir,input_names[idx])
        #input_image =Image.open(input_name)#Image get jpg
        input_image=cv2.imread(input_name)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        label_name = os.path.join(self.label_dir,label_names[idx])
        #label_image =Image.open(label_name)
        label_image=cv2.imread(label_name)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

        sample = {'input_image': input_image, 'label_image': label_image, 'name': input_names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def edge_clip(image):
    if image.shape[0]%2==1:
        image=image[:-1,:,:]
    if image.shape[1]%2==1:
        image=image[:,:-1,:]
    return image


class mytransform(object):
    def __call__(self, sample):
        input_image, label_image,name= sample['input_image'], sample['label_image'],sample['name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #input_image = numpy.asarray(input_image).transpose(2, 0, 1)/255.0
        #label_image = numpy.asarray(label_image).transpose(2, 0, 1)/255.0
        input_image=edge_clip(input_image)
        label_image=edge_clip(label_image)

        input_image = input_image.transpose(2, 0, 1)/255.0
        label_image = label_image.transpose(2, 0, 1)/255.0
        

        return {'input_image': torch.from_numpy(input_image).float(),
                'label_image': torch.from_numpy(label_image).float(),
                'name':name}
            
def wrap_variable(input_batch, label_batch, use_gpu,flag):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda(),volatile=flag), Variable(label_batch.cuda(),volatile=flag))

        else:
            input_batch, label_batch = (Variable(input_batch,volatile=flag),Variable(label_batch,volatile=flag))
        return input_batch, label_batch


def checkpoint(name,psnr1,psnr2,ssim1,ssim2,msssim1,msssim2):
    print('{},psnr:{:.4f}->{:.4f},ssim:{:.4f}->{:.4f},msssim:{:.4f}-{:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2,msssim1,msssim2))
    #write to text
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write(('{} {:.4f}->{:.4f} {:.4f}->{:.4f},{:.4f}->{:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2,msssim1,msssim2))+'\r\n')
    output.close()
def save(output_image,name):
    output_data=output_image.data[0]
    if use_gpu:
        img=255.0*output_data.clone().cpu().numpy()
    else:
        img=255.0*output_data.clone().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    #img = Image.fromarray(img)
    #img.save(os.path.join(Image_folder,'output','{}.jpg'.format(name[:-4])))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(Image_folder,'output','{}.png'.format(name[:-4])),img)

downsampling=nn.AvgPool2d(2)

def test():
    model.eval()
    avg_psnr1 = 0
    avg_ssim1 = 0
    avg_msssim1 = 0
    #output and label
    avg_psnr2 = 0
    avg_ssim2 = 0
    avg_msssim2 = 0


    for i, sample in enumerate(dataloader):
        input_image,label_image,name=sample['input_image'],sample['label_image'],sample['name'][0]#tuple to str
 
        #Wrap with torch Variable
        input_image,label_image=wrap_variable(input_image,label_image, use_gpu,True)
        inputs_1=downsampling(input_image)
        label_1=downsampling(label_image)
        inputs_2=downsampling(inputs_1)
        label_2=downsampling(label_1)

        output_image, outputs_1, outputs_2 = model(input_image,inputs_1,inputs_2)


        #predict
        #output_image = model(input_image)
        #clamp in[0,1]
        output_image=output_image.clamp(0.0, 1.0)


        #calculate psnr
        psnr1 =myutils.psnr(input_image, label_image)
        psnr2 =myutils.psnr(output_image, label_image)
        # ssim is calculated with the normalize (range [0, 1]) image
        ssim1 = torch.sum((myutils.ssim(input_image, label_image, size_average=False)).data)/1.0#batch_size
        ssim2 = torch.sum((myutils.ssim(output_image, label_image, size_average=False)).data)/1.0

    #msssim
        msssim1 = numpy.sum((msssim.MultiScaleSSIM(numpy.expand_dims(input_image.data[0].clone().cpu().numpy(),axis=0), numpy.expand_dims(label_image.data[0].clone().cpu().numpy(),axis=0), max_val=1.0)))/1.0#batch_size
        msssim2 = numpy.sum((msssim.MultiScaleSSIM(numpy.expand_dims(output_image.data[0].clone().cpu().numpy(),axis=0), numpy.expand_dims(label_image.data[0].clone().cpu().numpy(),axis=0), max_val=1.0)))/1.0

        avg_ssim1 += ssim1
        avg_psnr1 += psnr1
        avg_ssim2 += ssim2
        avg_psnr2 += psnr2
        avg_msssim1 += msssim1
        avg_msssim2 += msssim2

        #save output and record
        checkpoint(name,psnr1,psnr2,ssim1,ssim2,msssim1,msssim2)
        save(output_image,name)

    #print and save
    avg_psnr1 = avg_psnr1/len(dataloader)
    avg_ssim1 = avg_ssim1/len(dataloader)
    avg_psnr2 = avg_psnr2/len(dataloader)
    avg_ssim2 = avg_ssim2/len(dataloader)
    avg_msssim1 = avg_msssim1/len(dataloader)
    avg_msssim2 = avg_msssim2/len(dataloader)

    print('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f} Avg. MSSSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2,avg_msssim1,avg_msssim2))
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f}Avg. MSSSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2,avg_msssim1,avg_msssim2)+'\r\n')
    output.close()


#------------------------------------------------------------------
#cuda
use_gpu=torch.cuda.is_available()

#set path
root_dir=os.getcwd()
Image_folder=os.path.join(root_dir)
model_weights_file=os.path.join(root_dir,'parameters_nosao','MSqp42-124-0.001293-29.0237-0.8455param.pth')


#set model  
model=MS_model.IntraDeblocking()
#vgg=mymodel_new.Vgg16(requires_grad=False)
if use_gpu:
    model = model.cuda()
    #vgg = vgg.cuda()

model.load_state_dict(torch.load(model_weights_file))
#model=torch.load(model_weights_file)

#set dataset and dataloader
mydataset = ImageDataset(root_dir=Image_folder, transform=mytransform())
dataloader = DataLoader(mydataset, batch_size=1,shuffle=False, num_workers=0)



def main():
    test()

if __name__=='__main__':
    main()
