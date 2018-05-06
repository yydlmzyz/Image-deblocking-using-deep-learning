import argparse
import h5py
import numpy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from loss import GeneratorLoss
from math import log10
import GAN_model
import myutils

#prepare data
class MyDataset(Dataset):
    def __init__(self,data_file):
        self.file=h5py.File(str(data_file),'r')
        self.inputs=self.file['data'][:].astype(numpy.float32)/255.0#simple normalization in[0,1] 
        self.label=self.file['label'][:].astype(numpy.float32)/255.0
        #BUG!


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self,idx):
        inputs=self.inputs[idx,:,:,:].transpose(2,0,1)
        label=self.label[idx,:,:,:].transpose(2,0,1)
        inputs=torch.Tensor(inputs)
        label=torch.Tensor(label)
        sample={'inputs':inputs,'label':label}
        return sample



def checkpoint(epoch,d_loss,g_loss,d_score,g_score,psnr,ssim,mse):
    netG.eval()
    netD.eval()
    model_pathG = str(checkpoint_dir/'netG{}-{:.6f}-{:.4f}-{:.4f}.pth'.format(epoch,g_loss,psnr,ssim))
    torch.save(netG,model_pathG)

    model_pathD = str(checkpoint_dir/'netD{}-{:.6f}-{:.6f}-{:.6f}-{:.6f}.pth'.format(epoch,d_loss,g_loss,d_score,g_score))
    torch.save(netD,model_pathD)

    if use_gpu:
        netG.cpu()#you should save weights on cpu not on gpu
        netD.cpu()

    #save weights
    model_path_G_params = str(checkpoint_dir/'netG{}-{:.6f}-{:.4f}-{:.4f}params.pth'.format(epoch,g_loss,psnr,ssim))
    torch.save(netG.state_dict(),model_path_G_params)

    model_path_D_params = str(checkpoint_dir/'netD{}-{:.6f}-{:.6f}-{:.6f}-{:.6f}params.pth'.format(epoch,d_loss,g_loss,d_score,g_score))
    torch.save(netD.state_dict(),model_path_D_params)

    #print and save record
    print('Epoch {} : Avg.d_loss:{:.6f},g_loss:{:.6f},d_score:{:.6f},g_score:{:.6f},'.format(epoch,d_loss,g_loss,d_score,g_score))
    print("Test Avg. PSNR: {:.4f} Avg. SSIM: {:.4f} Avg.MSE{:.6f} ".format(psnr,ssim,mse))
    print("Checkpoint saved to {}".format(model_path_G_params))

    output = open(str(checkpoint_dir/'train_result.txt'),'a+')
    output.write(('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.4f} {:.4f} {:.6f}'.format(epoch,d_loss,g_loss,d_score,g_score,psnr,ssim,mse))+'\r\n')
    output.close()

    if use_gpu:
        netG.cuda()#don't forget return to gpu
        netD.cuda()



def wrap_variable(input_batch, label_batch, use_gpu,flag):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda(),volatile=flag), Variable(label_batch.cuda(),volatile=flag))

        else:
            input_batch, label_batch = (Variable(input_batch,volatile=flag),Variable(label_batch,volatile=flag))
        return input_batch, label_batch



def train(epoch):
    netG.train()
    netD.train()
    
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    for iteration, sample in enumerate(dataloader):#difference between (dataloader) &(dataloader,1)
        inputs,label=sample['inputs'],sample['label']
        batch_size = inputs.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu, False)

        # forward propagation in netG to get the fake_img
        fake_img=netG(inputs)

        #clear the optimizer of netD first!
        netD.zero_grad()
        #optimizerD.zero_grad()

        #forward propagation in netD
        real_out = netD(label).mean()
        fake_out = netD(fake_img).mean()

        #get the loss of Discriminator for backward
        d_loss = 1 - real_out + fake_out

        #another methond to calculate the d_loss,which use cross entropy
        '''
        #generator a target first:real is high,fake is low
        target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)#target of real should be high
        target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)#target of fake should be low

        #use BCE loss(cross entroy:loss(o,t)=-\frac{1}{n}\sum_i(t[i] log(o[i])+(1-t[i]) log(1-o[i])))
        adversarial_criterion = nn.BCELoss()
        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        '''


        #backward propagation and optimize the netD
        d_loss.backward(retain_graph=True)
        optimizerD.step()#Attention! use optimizerD means only optimize netD

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################

        #clear the optimizer of netG first!
        netG.zero_grad()

        #get the loss from fake_out & fake_img,which is generate from netG,so netD will get the gradient
        g_loss = generator_criterion(fake_out, fake_img, label)

        #backward propagation and optimize the netG
        g_loss.backward()
        optimizerG.step()

        ############################
        # (3) record and show the loss&score
        ###########################
        #calculate the loss
        fake_img = netG(inputs)
        fake_out = netD(fake_img).mean()

        g_loss = generator_criterion(fake_out, fake_img, label)#the lower, the better
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.data[0] * batch_size#the lower,the better,which means discriminator is right
        running_results['d_score'] += real_out.data[0] * batch_size#the higher,means discriminator the better
        running_results['g_score'] += fake_out.data[0] * batch_size#the higher,means generator the better(no more than 0.5)


        #which is used for monitor
        if iteration%100==0:
            print("===> Epoch[{}]({}/{}):loss_d: {:.6f} Loss_G: {:.6f} D(label)/d_score: {:.6f} D(G(inputs))/g_score: {:.6f}".format(
                epoch, iteration, len(dataloader),running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))   
        #if iteration==30:
        #    break

    return running_results['d_loss'] / running_results['batch_sizes'], running_results['g_loss'] / running_results['batch_sizes'], running_results['d_score'] / running_results['batch_sizes'],running_results['g_score'] / running_results['batch_sizes']



def test():
    netG.eval()
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

    for iteration, sample in enumerate(test_dataloader):
        inputs,label=sample['inputs'],sample['label']
        batch_size = inputs.size(0)
        valing_results['batch_sizes'] += batch_size

        #Wrap with torch Variable
        inputs,label=wrap_variable(inputs, label, use_gpu, True)
        #get the output of netG
        outputs = netG(inputs)
        #calcualte metrics
        batch_mse = ((outputs - label) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = myutils.ssim(outputs, label).data[0]
        valing_results['ssims'] += batch_ssim * batch_size#sum each patch,you must divide batch sizes
    
    valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

    return valing_results['psnr'], valing_results['ssim'], valing_results['mse'] / valing_results['batch_sizes']



def main():
    #train & test & record
    for epoch in range(args.epochs):
        d_loss,g_loss,d_score,g_score=train(epoch)
        psnr,ssim,mse = test()
        checkpoint(epoch,d_loss,g_loss,d_score,g_score,psnr,ssim,mse)



#---------------------------------------------------------------------------------------------------
# Training settings
parser = argparse.ArgumentParser(description='GAN')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--testbatchsize', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--d_lr', type=float, default=0.001, help='Learning Rate. Default=0.0001')
args = parser.parse_args()

print(args)

#----------------------------------------------------------------------------------------------------
#set other parameters
#1.set cuda
use_gpu=torch.cuda.is_available()


#2.set path and file
save_dir = Path('.')
checkpoint_dir = Path('.') / 'Checkpoints_GAN'#save model parameters and train record
if checkpoint_dir.exists():
    print 'folder esxited'
else:
    checkpoint_dir.mkdir()

generator_weights_file=checkpoint_dir/'18-0.001067-30.8295-0.9167param.pth'
discriminator_weights_file=checkpoint_dir/'netD10-0.925502-0.002677-0.543894-0.469396params.pth'


#4.set model& criterion& optimizer
netG=GAN_model.Generator()
netD=GAN_model.Discriminator()


generator_criterion = GeneratorLoss()

optimizerG = torch.optim.Adam(netG.parameters(), lr=args.g_lr)
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.d_lr)


'''
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
'''
if use_gpu:
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

#load parameters
if not use_gpu:
    netG.load_state_dict(torch.load(str(generator_weights_file), map_location=lambda storage, loc: storage))
    netD.load_state_dict(torch.load(str(discriminator_weights_file), map_location=lambda storage, loc: storage))
    #model=torch.load(str(model_weights_file), map_location=lambda storage, loc: storage)
else:
    netG.load_state_dict(torch.load(str(generator_weights_file)))  
    netD.load_state_dict(torch.load(str(discriminator_weights_file)))  
    #model=torch.load(str(model_weights_file))




#3.set dataset and dataloader
dataset=MyDataset(data_file=save_dir/'TrainData32.h5')
test_dataset=MyDataset(data_file=save_dir/'TestData32.h5')

dataloader=DataLoader(dataset,batch_size=args.batchsize,shuffle=True,num_workers=0)
test_dataloader=DataLoader(test_dataset,batch_size=args.testbatchsize,shuffle=False,num_workers=0)




#show mdoel&parameters&dataset
print('NetG Structure:',netG)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
params = list(netG.parameters())
for i in range(len(params)):
    print('layer:',i+1,params[i].size())

print('NetD Structure:',netD)
print('# generator parameters:', sum(param.numel() for param in netD.parameters()))
params = list(netD.parameters())
for i in range(len(params)):
    print('layer:',i+1,params[i].size())

print('length of dataset:',len(dataset))


if __name__=='__main__':
    main()


netG = Generator()