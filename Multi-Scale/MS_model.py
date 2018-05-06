import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as f



## 1)all relu =>> prelu

class Residual_Block(nn.Module):
    def __init__(self, Channel):
        super(Residual_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(Channel)
        self.bn2 = nn.BatchNorm2d(Channel)
        self.conv1 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.prelu(self.bn1(x1))
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = torch.add(x, x4)
        return x5


class IntraDeblocking(nn.Module):
    def __init__(self):
        super(IntraDeblocking,self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,1,3)
        self.prelu1 = nn.PReLU()
        self.downsample1 = nn.Conv2d(64,64,4,2,1)
        self.downsample2 = nn.Conv2d(64,64, 4, 2, 1)
        self.block0 = Residual_Block(64)
        self.block1 = Residual_Block(64)
        self.block2 = Residual_Block(64)
        self.block3 = Residual_Block(64)
        self.conv2 = nn.Conv2d(64,3,3,1,1)
        self.prelu2 = nn.PReLU()

        self.block4 = Residual_Block(64)
        self.block5 = Residual_Block(64)
        self.block6 = Residual_Block(64)
        self.block7 = Residual_Block(64)
        self.conv3 = nn.Conv2d(64,3,5,1,2)
        self.prelu3 = nn.PReLU()

        self.block8 = Residual_Block(64)
        self.block9 = Residual_Block(64)
        self.block10 = Residual_Block(64)
        self.block11 = Residual_Block(64)
        self.conv4 = nn.Conv2d(64,3,7,1,3)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(64,64,3,1,1)
        self.up1 = nn.PixelShuffle(2)
        self.prelu5 =nn.PReLU()

        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.up2 = nn.PixelShuffle(2)
        self.prelu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(80,64,3,1,1)
        self.prelu7 = nn.PReLU()
        self.conv8 = nn.Conv2d(80, 64, 3, 1, 1)
        self.prelu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(3, 64, 5, 1, 2)
        self.prelu9 = nn.PReLU()
        self.conv10 = nn.Conv2d(3, 64, 3, 1, 1)
        self.prelu10 = nn.PReLU()
    def forward(self,x,x1,x2):
        x0 = self.prelu1(self.conv1(x))
        ## decompose the feature into 3 scales
        x1 = self.prelu9(self.conv9(x1))
        x2 = self.prelu10(self.conv10(x2))
        ## the third scale
        x = self.block0(x2)
        x = self.block1(x)
        x = self.block2(x)
        xt1 = self.block3(x)
        x = self.conv5(xt1)
        x_cat1 = self.prelu5(self.up1(x))

        output3 = self.prelu2(self.conv2(xt1))
        ## the second scale
        x = torch.cat((x_cat1,x1),dim=1)
        x = self.prelu7(self.conv7(x))
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        xt2 = self.block7(x)
        output2 = self.prelu3(self.conv3(xt2))
        x_cat2 = self.prelu6(self.up2(self.conv6(xt2)))
        ## the first scale
        x = torch.cat((x_cat2,x0), dim=1)
        x = self.prelu8(self.conv8(x))
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        output1 = self.prelu4(self.conv4(x))


        return output1,output2,output3
if __name__== '__main__':
    x = Variable(torch.rand(8,3,64,64)).cuda()
    deblock = IntraDeblocking().cuda()
    y1,y2,y3 = deblock(x)
    print y1,y2,y3