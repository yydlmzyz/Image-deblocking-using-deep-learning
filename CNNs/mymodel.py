import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models
import math
from torch.autograd import Variable



#model_1 baseline
class ARCNN(nn.Module):

    def __init__(self):
        super(ARCNN, self).__init__()
        #feature extract
        self.conv1=nn.Conv2d(3,64,9,1,(9-1)/2)
        #nolinear mapping
        self.conv2=nn.Conv2d(64,32,7,1,(7-1)/2)
        self.conv3=nn.Conv2d(32,16,1,1,0)
        #reconstruct
        self.conv4=nn.Conv2d(16,3,5,1,(5-1)/2)
        #self.relu=nn.ReLU(inplace=False)#not sure

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):#
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        return x

#########################################################################################################
#model_2 
class L8(nn.Module):

    def __init__(self):
        super(L8, self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=11,
            stride=1,
            padding=5# if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1  
            )
        self.conv2=nn.Conv2d(32,64,3,1,1)
        self.conv3=nn.Conv2d(64,64,3,1,1)
        self.conv4=nn.Conv2d(64,64,3,1,1)
        self.conv5=nn.Conv2d(32+64,64,1,1,0)
        self.conv6=nn.Conv2d(64,64,5,1,2)
        self.conv7=nn.Conv2d(64+32,128,1,1,0)
        self.conv8=nn.Conv2d(128,3,5,1,2)
        #self.relu=nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):#not sure what is x?
        x1=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x1))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=torch.cat([x, x1], 1)#the dimensionality of Variable is [number,channel,height,width]
        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=torch.cat([x, x1], 1)
        x=F.relu(self.conv7(x))
        x=F.relu(self.conv8(x))
        return x

#############################################################################################
#the following model refer to https://blog.csdn.net/abluemouse/article/details/78710553


#model_3:refer to vdsr:https://arxiv.org/abs/1511.04587
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class vdar(nn.Module):
    def __init__(self):
        super(vdar, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        #weights initialization by normal(Gaussinn) distribution:normal_(mean=0, std=1 , gengerator=None*)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)#global residual
        return out


#########################################################################################################

#model_5 refer to edsr:https://arxiv.org/abs/1707.02921
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)#attrntion:relu not prelu

    def forward(self, x):
        residual = self.conv(x)
        residual = self.relu(residual)
        residual = self.conv(residual)#the original model multipy a value 0.1 here,which is to be analyzed in the feature
        out      = torch.add(x,residual)

        return out

class edar(nn.Module):
    def __init__(self):
        super(edar, self).__init__()
        self.head= nn.Conv2d(3,64,3,1,1)
        self.body = nn.ModuleList([ResidualBlock(64) for i in range(8)])
        self.tail=nn.Conv2d(64,64,3,1,1)
        self.reconstruct=nn.Conv2d(64,3,3,1,1)
        self.relu= nn.ReLU(inplace=True)

        #weights initialization by normal(Gaussinn) distribution:normal_(mean=0, std=1 , gengerator=None*)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  

    def forward(self, x):
        x = self.head(x)
        residual=x#i'm afraid this may cause residual is x
        for layer in self.body:
            residual = layer(residual)
        residual = self.tail(residual)#
        out  = torch.add(x,residual)#global residual
        #out=self.tail(out)
        out  = self.reconstruct(out)
        return out

#########################################################################################################
#change add to cat

class edar2(nn.Module):
    def __init__(self):
        super(edar2, self).__init__()
        self.head= nn.Conv2d(3,64,3,1,1)
        self.body = nn.ModuleList([ResidualBlock(64) for i in range(8)])
        self.tail=nn.Conv2d(64,64,3,1,1)
        self.reconstruct=nn.Conv2d(128,3,3,1,1)
        self.relu= nn.ReLU(inplace=True)

        #weights initialization by normal(Gaussinn) distribution:normal_(mean=0, std=1 , gengerator=None*)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  

    def forward(self, x):
        x = self.head(x)
        residual=x#i'm afraid this may cause residual is x
        for layer in self.body:
            residual = layer(residual)
        residual = self.relu(self.tail(residual))#
        out  = torch.cat([x,residual],1)#global residual
        #out=self.tail(out)
        out  = self.reconstruct(out)
        return out



#########################################################################################################


#model_6 refer to SRDenseNet:http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf
class _Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(_Dense_Block, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=112, out_channels=16, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))

        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))

        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))

        return cout8_dense

class ARDenseNet(nn.Module):
    def __init__(self):
        super(ARDenseNet, self).__init__()
        
        self.relu = nn.PReLU()
        self.lowlevel = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(in_channels=640, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(_Dense_Block, 128)
        self.denseblock2 = self.make_layer(_Dense_Block, 256)
        self.denseblock3 = self.make_layer(_Dense_Block, 384)
        self.denseblock4 = self.make_layer(_Dense_Block, 512)
        '''
        self.denseblock5 = self.make_layer(_Dense_Block, 640)
        self.denseblock6 = self.make_layer(_Dense_Block, 768)
        self.denseblock7 = self.make_layer(_Dense_Block, 896)
        self.denseblock8 = self.make_layer(_Dense_Block, 1024)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):    
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual,out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)
        '''
        out = self.denseblock5(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock6(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock7(concat)
        concat = torch.cat([concat,out], 1)
        
        out = self.denseblock8(concat)
        out = torch.cat([concat,out], 1)
        '''
        out = self.bottleneck(concat)

        out = self.reconstruction(out)
       
        return out



#########################################################################################################

#model_7 refer to RDN:https://arxiv.org/abs/1802.08797
class Residual_Dense_Block(nn.Module):
    def __init__(self):
        super(Residual_Dense_Block,self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64+32*2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64+32*3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64+32*4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64+32*5, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fusion= nn.Conv2d(in_channels=64+32*6, out_channels=64, kernel_size=1, stride=1, padding=0)#local feature fusion

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        cout1_dense = torch.cat([conv1,x], 1)

        conv2 = self.relu(self.conv2(cout1_dense))
        cout2_dense = torch.cat([conv1,conv2,x], 1)

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = torch.cat([conv1,conv2,conv3,x], 1)

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = torch.cat([conv1,conv2,conv3,conv4,x], 1)

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = torch.cat([conv1,conv2,conv3,conv4,conv5,x], 1)

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,x], 1)

        conv7 = self.fusion(cout6_dense)
        out=torch.add(conv7,x)#local residual learning
        return out




class Residual_Dense_Network(nn.Module):
    def __init__(self):
        super(Residual_Dense_Network,self).__init__()
        self.shallow_feature_extraction_1=nn.Conv2d(3,64,3,1,1)
        self.shallow_feature_extraction_2=nn.Conv2d(64,64,3,1,1)

        self.RDB=Residual_Dense_Block()

        self.global_feature_fusion1=nn.Conv2d(64*8,64,3,1,1)#adaptively fuse a range of features with different levels
        self.global_feature_fusion2=nn.Conv2d(64,64,3,1,1)# extract features for global residual learning

        self.reconstruct=nn.Conv2d(64,3,3,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        #SFENET
        shallow_feature_1=self.shallow_feature_extraction_1(x)#for global residual learning

        shallow_feature_2=self.shallow_feature_extraction_2(shallow_feature_1)

        #RDB 16
        RDB_1=self.RDB(shallow_feature_2)
        RDB_2=self.RDB(RDB_1)
        RDB_3=self.RDB(RDB_2)
        RDB_4=self.RDB(RDB_3)
        RDB_5=self.RDB(RDB_4)
        RDB_6=self.RDB(RDB_5)
        RDB_7=self.RDB(RDB_6)
        RDB_8=self.RDB(RDB_7)
        '''
        RDB_9=self.RDB(RDB_8)
        RDB_10=self.RDB(RDB_9)
        RDB_11=self.RDB(RDB_10)
        RDB_12=self.RDB(RDB_11)
        RDB_13=self.RDB(RDB_12)
        RDB_14=self.RDB(RDB_13)
        RDB_15=self.RDB(RDB_14)
        RDB_16=self.RDB(RDB_15)
        '''

        #concat=torch.cat([RDB_1,RDB_2,RDB_3,RDB_4,RDB_5,RDB_6,RDB_7,RDB_8,RDB_9,RDB_10,RDB_11,RDB_12,RDB_13,RDB_14,RDB_15,RDB_16], 1)
        concat=torch.cat([RDB_1,RDB_2,RDB_3,RDB_4,RDB_5,RDB_6,RDB_7,RDB_8], 1)

        #DFF
        global_feature1=self.global_feature_fusion1(concat)
        global_feature2=self.global_feature_fusion2(global_feature1)

        #global residual learning
        out=torch.add(shallow_feature_1,global_feature2)
        #reconstruct
        out=self.reconstruct(out)

        return out



if __name__== '__main__':
        #show mdoel&parameters&dataset
    model=Residual_Dense_Network().cuda()
    print('Model Structure:',model)
    print('parameters:', sum(param.numel() for param in model.parameters()))
    params = list(model.parameters())
    for i in range(len(params)):
        print('layer:',i+1,params[i].size())

    x = Variable(torch.rand(8,3,4,4)).cuda()
    y1,y2,y3 = model(x)
    print y1,y2,y3


