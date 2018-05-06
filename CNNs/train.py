import argparse
import h5py
import numpy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

import mymodel
import myutils

#prepare data
class MyDataset(Dataset):
    def __init__(self,data_file):
        self.file=h5py.File(str(data_file),'r')
        self.inputs=self.file['data'][:].astype(numpy.float32)/255.0#simple normalization in[0,1]
        self.label=self.file['label'][:].astype(numpy.float32)/255.0

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self,idx):
        inputs=self.inputs[idx,:,:,:].transpose(2,0,1)
        label=self.label[idx,:,:,:].transpose(2,0,1)
        inputs=torch.Tensor(inputs)
        label=torch.Tensor(label)
        sample={'inputs':inputs,'label':label}
        return sample
   

def checkpoint(epoch,loss,psnr,ssim,mse):
    model.eval()
    model_path1 = str(checkpoint_dir/'{}-{:.6f}-{:.4f}-{:.4f}.pth'.format(epoch,loss,psnr,ssim))
    torch.save(model,model_path1)

    if use_gpu:
        model.cpu()#you should save weights on cpu not on gpu

    #save weights
    model_path = str(checkpoint_dir/'{}-{:.6f}-{:.4f}-{:.4f}param.pth'.format(epoch,loss,psnr,ssim))

    torch.save(model.state_dict(),model_path)   

    #print and save record
    print('Epoch {} : Avg.loss:{:.6f}'.format(epoch,loss))
    print("Test Avg. PSNR: {:.4f} Avg. SSIM: {:.4f} Avg.MSE{:.6f} ".format(psnr,ssim,mse))
    print("Checkpoint saved to {}".format(model_path))

    output = open(str(checkpoint_dir/'train_result.txt'),'a+')
    output.write(('{} {:.4f} {:.4f} {:.4f}'.format(epoch,loss,psnr,ssim))+'\r\n')
    output.close()

    if use_gpu:
        model.cuda()#don't forget return to gpu
    #model.train()


def wrap_variable(input_batch, label_batch, use_gpu,flag):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda(),volatile=flag), Variable(label_batch.cuda(),volatile=flag))

        else:
            input_batch, label_batch = (Variable(input_batch,volatile=flag),Variable(label_batch,volatile=flag))
        return input_batch, label_batch



def train(epoch):
    model.train()
    sum_loss=0.0
    
    for iteration, sample in enumerate(dataloader):#difference between (dataloader) &(dataloader,1)
        inputs,label=sample['inputs'],sample['label']

        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu, False)

        #clear the optimizer
        optimizer.zero_grad()

        # forward propagation
        outputs = model(inputs)

        #get the loss for backward
        loss =criterion(outputs, label)

        #backward propagation and optimize
        loss.backward()
        optimizer.step()

        if iteration%100==0:
            print("===> Epoch[{}]({}/{}):loss: {:.6f}".format(epoch, iteration, len(dataloader), loss.data[0]))   
        #if iteration==101:
        #    break

        #caculate the average loss
        sum_loss += loss.data[0]

    return sum_loss/len(dataloader)


def test():
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    avg_mse  = 0
    for iteration, sample in enumerate(test_dataloader):
        inputs,label=sample['inputs'],sample['label']
        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu, True)

        outputs = model(inputs)
        mse  = criterion(outputs,label).data[0]
        psnr = myutils.psnr(outputs, label)
        ssim = torch.sum((myutils.ssim(outputs, label, size_average=False)).data)/args.testbatchsize
        avg_ssim += ssim
        avg_psnr += psnr
        avg_mse  += mse
    return (avg_psnr / len(test_dataloader)),(avg_ssim / len(test_dataloader)),(avg_mse/len(test_dataloader))



def main():
    #train & test & record
    for epoch in range(args.epochs):
        loss=train(epoch)
        psnr,ssim,mse = test()
        checkpoint(epoch,loss,psnr,ssim,mse)



#---------------------------------------------------------------------------------------------------
# Training settings
parser = argparse.ArgumentParser(description='ARCNN')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--testbatchsize', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
args = parser.parse_args()

print(args)

#----------------------------------------------------------------------------------------------------
#set other parameters
#1.set cuda
use_gpu=torch.cuda.is_available()


#2.set path and file
save_dir = Path('.')
checkpoint_dir = Path('.') / 'Checkpoints_ARCNN'#save model parameters and train record
if checkpoint_dir.exists():
    print 'folder esxited'
else:
    checkpoint_dir.mkdir()

model_weights_file=checkpoint_dir/'134-0.000398-34.0115-0.9435param.pth'


#3.set dataset and dataloader
dataset=MyDataset(data_file=save_dir/'TrainData32.h5')
test_dataset=MyDataset(data_file=save_dir/'TestData32.h5')

dataloader=DataLoader(dataset,batch_size=args.batchsize,shuffle=True,num_workers=0)
test_dataloader=DataLoader(test_dataset,batch_size=args.testbatchsize,shuffle=False,num_workers=0)


#4.set model& criterion& optimizer
model=mymodel.ARCNN()

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)

if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()

#load parameters
if not use_gpu:
    model.load_state_dict(torch.load(str(model_weights_file), map_location=lambda storage, loc: storage))
    #model=torch.load(str(model_weights_file), map_location=lambda storage, loc: storage)
else:
    model.load_state_dict(torch.load(str(model_weights_file)))  
    #model=torch.load(str(model_weights_file))


#show mdoel&parameters&dataset
print('Model Structure:',model)
print('# parameters:', sum(param.numel() for param in model.parameters()))
params = list(model.parameters())
for i in range(len(params)):
    print('layer:',i+1,params[i].size())

print('length of dataset:',len(dataset))


if __name__=='__main__':
    main()
