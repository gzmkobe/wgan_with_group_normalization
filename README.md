# wgan-GP_with_group_normalization

The objective of the project is to combine Group Normalization with WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) to achieve better performance on image generation. For more reading, please click *[Report](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/Final_Report.pdf)


## Background
**Generative Adversarial Network**, estimated generative models by using an adversarial process.
**WGAN** further developed the model by introducing Wasserstein distance as a metric to optimize discriminator. 
As the model still appears slow to converge and hard to train, **improved WGAN-GP** with a gradient penalty in loss to achieve better learning result

![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture1.png "formula")

## Motivation & Introduction

**Issue with batch normalization (BN) in WGAN-GP** 
WGAN-GP penalizes the norm of the gradient to individual data input, not the entire batch of data
Batch normalization, however, maps whole batch of inputs for a batch of outputs.

**Layer normalization (LN )and instance normalization (IN)**
 Both don’t suffer from BN’s issues but didn’t perform well as expected.
 
**Group normalization (GN)**
GN divides the channels into groups and within each group computes the mean and variance and therefore overcomes the BN’s issues.

**Goal**: we embedded GN in WGAN-GP and DAGAN. Quality of image generation is evaluated by inception score

## Formula 
**General Normalization Formula**

![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture5.png "normalization1")
![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture6.png "normalization2")
![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture2.png "gp norm")

where N is the batch axis, C is the channel axis, and H and W are the spatial height and width axes;G is the number of groups, C/G is the number of channels per group

![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture3.png "types of norm")

## ARCHITECTURE & METHODOLOGY
**Design of Architecture** RESNET is used for generator and discriminator and below is the detail
![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/picture3.jpg "Architecture")

**Design of discriminator**
There are 6 normalization operation in discriminator: the discriminator has 3 residual blocks; each residual block includes 2 convolutional layers; each convolutional layer has an operation 

## EXPERIMENT RESULTS ON CIFAR-10
**Dataset: CIFAR-10**  32x32 color images in 10 classes; 50,000 training and 10,000 test
![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture4.png "cifar10")


![alt text](https://github.com/gzmkobe/wgan_with_group_normalization/blob/master/reference_imgs/Picture7.png "results")
Group Normalization improves inception score a little bit for both Simple-CNN and ResNet over baseline mode. 



