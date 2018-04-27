import os, sys
sys.path.append(os.getcwd())
import argparse

import time
import tflib as lib
import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
import tflib.inception_score

from model.cifar10 import Discriminator, Generator, Discriminator_with_group_norm, Discriminator_with_ResNet, Generator_with_ResNet



# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
SAVE_PATH = './tmp/cifar10_groupnorm/'
DATA_DIR = '../data/cifar-10-batches-py/'

parser = argparse.ArgumentParser(description='Hyper-parameter of WGAN for CIFAR-10')

parser.add_argument('--MODE', type=str, default='wgan-gp', help='Valid options are dcgan, wgan, or wgan-gp')
parser.add_argument('--DIM', type=int, default=64, help = 'dimension of the fully-connected layer at front')
parser.add_argument('--LAMBDA', type=int, default=10, help = 'Gradient penalty lambda hyperparameter')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help = 'Batch size')
parser.add_argument('--CRITIC_ITERS', type=int, default=5, help = 'How many critic iterations per generator iteration')
parser.add_argument('--ITERS', type=int, default=200000, help = 'How many generator iterations to train for')
parser.add_argument('--OUTPUT_DIM', type = int, default = 3072, help = 'Number of pixels in CIFAR10 (3*32*32)')
parser.add_argument('--IS_CAL_ROUND', type = int, default = 1000, help = 'calculate the Inception score per IS_CAL_ROUND of epoch')
parser.add_argument('--IMAGE_SAVE_ROUND', type = int, default = 1000, help = 'save the generated images per IS_CAL_ROUND of epoch')
parser.add_argument('--BATCH_SIZE_IS', type = int, default = 64, help = 'BATCH_SIZE for inception score calculation')
parser.add_argument('--GROUP_NUM', type = int, default = 16, help = 'Number of groups in group Normalization')
parser.add_argument('--RESNET',type = bool, default = False, help = 'Whether use ResNet Discriminator and Generator')
parser.add_argument('--NORMALIZATION', type = str, default = 'none',  choices = ['layernorm', 'none', 'groupnorm'],help = 'Type of Normalization')

args = parser.parse_args()

MODE = args.MODE # Valid options are dcgan, wgan, or wgan-gp
DIM = args.DIM # This overfits substantially; you're probably better off with 64
LAMBDA = args.LAMBDA # Gradient penalty lambda hyperparameter
CRITIC_ITERS = args.CRITIC_ITERS # How many critic iterations per generator iteration
BATCH_SIZE = args.BATCH_SIZE # Batch size
ITERS = args.ITERS # How many generator iterations to train for
OUTPUT_DIM = args.OUTPUT_DIM # Number of pixels in CIFAR10 (3*32*32)
IS_CAL_ROUND = args.IS_CAL_ROUND
BATCH_SIZE_IS = args.BATCH_SIZE_IS
IMAGE_SAVE_ROUND = args.IMAGE_SAVE_ROUND
GROUP_NUM = args.GROUP_NUM
NORMALIZATION = args.NORMALIZATION
RESNET = args.RESNET

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

if os.path.exists(SAVE_PATH) == False:
    os.mkdir(SAVE_PATH)

if os.path.exists(os.path.join(SAVE_PATH, 'samples/')) == False:
    os.mkdir(os.path.join(SAVE_PATH, 'samples/'))

if(RESNET == True):
    netG = Generator_with_ResNet(128)
else:
    netG = Generator(DIM)

if(RESNET == True):
    netD = Discriminator_with_ResNet(128)
elif NORMALIZATION == 'groupnorm':
    netD = Discriminator_with_group_norm(DIM, GROUP_NUM)
elif NORMALIZATION == 'none':
    netD = Discriminator(DIM)

print(netG)
print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, SAVE_PATH + 'samples/' + str(frame) +'.jpg')

# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in range(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32))

    return lib.inception_score.inception_score(list(all_samples),cuda=use_cuda, batch_size = BATCH_SIZE_IS, resize = True, splits = 2)
  

# Dataset iterator
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)

def inf_train_gen():
    while True:
        for images in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images

gen = inf_train_gen()
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

for iteration in range(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in range(CRITIC_ITERS):

        #print('CRITIC_ITERS:', i)
        _data = next(gen)
        netD.zero_grad()

        # train with real
        _data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        D_real = netD(real_data_v)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        #print(Wasserstein_D.cpu().data.numpy())

        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(SAVE_PATH + 'train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot(SAVE_PATH + 'time', time.time() - start_time)
    lib.plot.plot(SAVE_PATH + 'train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot(SAVE_PATH + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate inception score every 1K iters
    # if False and iteration % 1000 == 999:
    #     inception_score = get_inception_score(netG)
    #     lib.plot.plot('./tmp/cifar10/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters

    if iteration % IS_CAL_ROUND == IS_CAL_ROUND - 1:
          inception_score = get_inception_score(netG)
          print("Inception score for iteration " +str(iteration)+" is "+str(inception_score))
          lib.plot.plot(SAVE_PATH + 'inception score', inception_score[0])


    if iteration % IMAGE_SAVE_ROUND == IMAGE_SAVE_ROUND -1 :
        dev_disc_costs = []
        for images in dev_gen():
            images = images.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            imgs = torch.stack([preprocess(item) for item in images])

            # imgs = preprocess(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot(SAVE_PATH + 'dev disc cost', np.mean(dev_disc_costs))

        generate_image(iteration, netG)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()
