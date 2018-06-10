import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models
import math
from torch.autograd import Variable



#model_1
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


#model_3:refer to vdsr
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



class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)#attrntion:relu not prelu
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)#the original model multipy a value 0.1 here,which is to be analyzed in the feature
        out      = torch.add(x,residual)

        return out


#model_4_1 refer to edsr
class edar(nn.Module):
    def __init__(self):
        super(edar, self).__init__()
        self.head= nn.Conv2d(3,64,3,1,1)
        #self.body = nn.ModuleList([ResidualBlock(64) for i in range(8)])
        self.body = self.make_layer(ResidualBlock, 8)
        self.tail=nn.Conv2d(64,64,3,1,1)
        self.reconstruct=nn.Conv2d(64,3,3,1,1)
        self.relu= nn.ReLU(inplace=True)

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
        x = self.head(x)
        residual=x#i'm afraid this may cause residual is x
        for layer in self.body:
            residual = layer(residual)
        residual = self.tail(residual)#
        out  = torch.add(x,residual)#global residual
        #out=self.tail(out)
        out  = self.reconstruct(out)
        return out


#model_4_2 refer to edsr
class edar2(nn.Module):
    def __init__(self):
        super(edar2, self).__init__()
        self.head= nn.Conv2d(3,64,3,1,1)
        #self.body = nn.ModuleList([ResidualBlock(64) for i in range(8)])
        self.body = self.make_layer(ResidualBlock, 8)
        self.tail=nn.Conv2d(64,64,3,1,1)
        self.reconstruct=nn.Conv2d(128,3,3,1,1)
        self.relu= nn.ReLU(inplace=True)

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
        x = self.head(x)
        residual=x#i'm afraid this may cause residual is x
        residual = self.body(residual)
        residual = self.relu(self.tail(residual))#
        out  = torch.cat([x,residual],1)#global residual
        #out=self.tail(out)
        out  = self.reconstruct(out)
        return out


#model_5 refer to SRDenseNet
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

        out = self.bottleneck(concat)

        out = self.reconstruction(out)
       
        return out


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features#It will dawnload the parameters,which will spend some time
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out




if __name__== '__main__':
    model=ARCNN().cuda()
    print('Model Structure:',model)
    
    params = list(model.parameters())
    for i in range(len(params)):
        print('layer:',i+1,params[i].size())
    print('parameters:', sum(param.numel() for param in model.parameters()))


