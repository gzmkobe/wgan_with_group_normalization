import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

        


class Generator(nn.Module):
    def __init__(self, DIM):
        super(Generator, self).__init__()
        self.DIM = DIM
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * self.DIM),
            nn.BatchNorm2d(4 * 4 * 4 * self.DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 2, stride=2),
            nn.BatchNorm2d(2 * self.DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 2, stride=2),
            nn.BatchNorm2d(self.DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)

class Discriminator_with_group_norm(nn.Module):
    def __init__(self, DIM, group_num):
        super(Discriminator_with_group_norm, self).__init__()
        self.DIM = DIM
        main = nn.Sequential(
            nn.Conv2d(3, self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(self.DIM, group_num),
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(2 * self.DIM, group_num),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(4 * self.DIM, group_num),
            nn.LeakyReLU()   
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*self.DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output

    
class Discriminator_with_layer_norm(nn.Module):
    def __init__(self, DIM, group_num =1):
        super(Discriminator_with_layer_norm, self).__init__()
        self.DIM = DIM
        main = nn.Sequential(
            nn.Conv2d(3, self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(self.DIM, group_num),
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(2 * self.DIM, group_num),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            GroupBatchnorm2d(4 * self.DIM, group_num),
            nn.LeakyReLU()   
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*self.DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output       
    

class Discriminator(nn.Module):
    def __init__(self, DIM):
        super(Discriminator, self).__init__()
        self.DIM = DIM
        main = nn.Sequential(
            nn.Conv2d(3, self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*self.DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output

def ConvMeanPool(input_dim, output_dim, filter_size):
    return nn.Sequential(nn.Conv2d(input_dim,output_dim,filter_size,padding=int((filter_size-1)/2)),nn.AvgPool2d(2, stride=2))

def MeanPoolConv(input_dim, output_dim, filter_size):
    return nn.Sequential(nn.AvgPool2d(2, stride=2),nn.Conv2d(input_dim,output_dim,filter_size,padding=int((filter_size-1)/2)))

def UpSampleConv(input_dim, output_dim, filter_size):
    return(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(input_dim,output_dim,filter_size,padding=int((filter_size-1)/2))))

class ResidualBlock(nn.Module):
    """docstring for ResidualBlock
    """

    def __init__(self, input_dim, output_dim, filter_size, resample=None, no_dropout=False, labels=None):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.resample = resample
        if(resample != 'up'):
            self.bn1 = GroupBatchnorm2d(self.input_dim, 1)
            self.bn2 = GroupBatchnorm2d(self.output_dim, 1)
        else:
            self.bn1 = nn.BatchNorm2d(self.input_dim)
            self.bn2 = nn.BatchNorm2d(self.output_dim)
        self.conv2d_in_in = nn.Conv2d(self.input_dim,self.input_dim,self.filter_size,padding=int((self.filter_size-1)/2))
        self.conv2d_down = nn.Conv2d(self.input_dim,self.output_dim,1,padding=int((self.filter_size-1)/2))
        self.conv2d_in_out = nn.Conv2d(self.input_dim,self.output_dim,self.filter_size,padding=int((self.filter_size-1)/2))
        self.conv2d_out_out = nn.Conv2d(self.output_dim,self.output_dim,self.filter_size,padding=int((self.filter_size-1)/2))

        if(resample == None):
            self.conv_1 = self.conv2d_in_out
            self.conv_2 = self.conv2d_out_out
            self.conv_shortcut = self.conv2d_down 
        elif(resample == 'up'):
            self.conv_1 = UpSampleConv(self.input_dim,self.output_dim,self.filter_size)
            self.conv_2 = self.conv2d_out_out
            self.conv_shortcut = UpSampleConv(self.input_dim,self.output_dim,1)
        elif(resample == 'down'):
            self.conv_1 = self.conv2d_in_in
            self.conv_2 = ConvMeanPool(self.input_dim,self.input_dim,self.filter_size)
            self.conv_shortcut = ConvMeanPool(self.input_dim,self.output_dim,1)

    def forward(self,inputs):
        if(self.resample == None) and (self.output_dim==self.input_dim):
            shortcut = inputs
        else:
            shortcut = self.conv_shortcut(inputs)
        output = inputs
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = F.relu(output)
        output = self.conv_2(output)

        return output + shortcut

class OptimizedReslock(nn.Module):
    """docstring for OptimizedReslock"""
    def __init__(self, DIM):
        super(OptimizedReslock, self).__init__()
        self.DIM = DIM
        self.conv_1 = nn.Conv2d(3,self.DIM,3,padding=1)
        self.conv_2 = ConvMeanPool(self.DIM,self.DIM,3)
        self.conv_shortcut = MeanPoolConv(3,self.DIM,1)
    def forward(self,inputs):
        shortcut = self.conv_shortcut(inputs)
        output = self.conv_1(inputs)
        output = F.relu(output)
        output = self.conv_2(output)

        return shortcut+output
        

#ResNet Discriminator & Generator from Improved WGAN/ GP-WGAN
class Discriminator_with_ResNet(nn.Module):
    """docstring for Discriminator_with_ResNet

    """
    def __init__(self, DIM):
        super(Discriminator_with_ResNet, self).__init__()
        self.DIM = DIM

        self.ResidualBlock_Disc = nn.Sequential(ResidualBlock(self.DIM,self.DIM,3,resample ='down'),
                                            ResidualBlock(self.DIM,self.DIM,3,resample = None),
                                            ResidualBlock(self.DIM,self.DIM,3,resample = None))

        self.OptimizedReslock = OptimizedReslock(self.DIM)
        self.MeanPooling = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(self.DIM,1)
    def forward(self,inputs):
        output = inputs.view([-1,3,32,32])
        output = self.OptimizedReslock(output)
        output = self.ResidualBlock_Disc(output)
        output = F.relu(output)
        output = self.MeanPooling(output).view([-1,self.DIM])
        output = self.linear(output)
        return output

class Generator_with_ResNet(nn.Module):
    """docstring for Generator_with_ResNet"""
    def __init__(self, DIM):
        super(Generator_with_ResNet, self).__init__()
        self.DIM = DIM
        self.linear = nn.Linear(128,self.DIM*4*4)
        self.ResidualBlock_1 = ResidualBlock(self.DIM,self.DIM,3,resample= 'up')
        self.ResidualBlock_2 = ResidualBlock(self.DIM,self.DIM,3,resample= 'up')
        self.ResidualBlock_3 = ResidualBlock(self.DIM,self.DIM,3,resample= 'up')
        self.bn = nn.BatchNorm2d(self.DIM)
        self.conv2d = nn.Conv2d(self.DIM,3,3,padding=1)
        self.tanh = nn.Tanh()
    def forward(self,inputs):
        output = self.linear(inputs)
        output = output.view([-1,self.DIM,4,4])
        output = self.ResidualBlock_1(output)
        output = self.ResidualBlock_2(output)
        output = self.ResidualBlock_3(output)
        output = self.bn(output)
        output = F.relu(output)
        output = self.conv2d(output)
        output = self.tanh(output)
        return output.view([-1,3*32*32])
        
