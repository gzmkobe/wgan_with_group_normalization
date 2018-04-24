import torch
from torch import nn

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
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            GroupBatchnorm2d(4 * self.DIM, group_num)
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