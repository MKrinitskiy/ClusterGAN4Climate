from __future__ import print_function

try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch

    from torchvision.models.resnet import BasicBlock, Bottleneck
    
    from itertools import chain as ichain

    from .utils import tlog, softmax, initialize_weights, calc_gradient_penalty

    # from torchsummary import summary
except ImportError as e:
    print(e)
    raise ImportError


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = [128, 4, 4]
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.l1 = nn.Linear(self.latent_dim + self.n_c, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.l2 = nn.Linear(1024, self.iels)
        self.bn2 = nn.BatchNorm1d(self.iels)

        x_channels = self.ishape[0]
        self.rnb_1 = BasicBlock(x_channels, x_channels)
        self.rnb_2 = BasicBlock(x_channels, x_channels)
        self.downsample_3 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(x_channels // 2))
        self.rnb_3 = BasicBlock(x_channels, x_channels // 2,
                                downsample=self.downsample_3)  # residual block (since using self.downsample_3 at the residual connection)

        x_channels = x_channels//2
        self.rnb_4 = BasicBlock(x_channels, x_channels)
        self.rnb_5 = BasicBlock(x_channels, x_channels)
        self.downsample_6 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(x_channels // 2))
        self.rnb_6 = BasicBlock(x_channels, x_channels // 2,
                                downsample=self.downsample_6)  # residual block (since using self.downsample_6 at the residual connection)

        x_channels = x_channels // 2
        self.rnb_7 = BasicBlock(x_channels, x_channels)
        self.rnb_8 = BasicBlock(x_channels, x_channels)
        self.downsample_9 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(x_channels // 2))
        self.rnb_9 = BasicBlock(x_channels, x_channels // 2,
                                downsample=self.downsample_9)  # residual block (since using self.downsample_9 at the residual connection)

        x_channels = x_channels//2
        self.rnb_10 = BasicBlock(x_channels, x_channels)
        self.rnb_11 = BasicBlock(x_channels, x_channels)
        self.downsample_12 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(x_channels // 2))
        self.rnb_12 = BasicBlock(x_channels, x_channels // 2,
                                 downsample=self.downsample_12)  # residual block (since using self.downsample_12 at the residual connection)

        x_channels = x_channels//2
        self.rnb_13 = BasicBlock(x_channels, x_channels)
        self.rnb_14 = BasicBlock(x_channels, x_channels)
        self.downsample_15 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
                nn.BatchNorm2d(x_channels // 2))
        self.rnb_15 = BasicBlock(x_channels, x_channels//2, downsample=self.downsample_15)

        x_channels = x_channels // 2
        self.rnb_16 = BasicBlock(x_channels, x_channels)
        self.rnb_17 = BasicBlock(x_channels, x_channels)
        self.downsample_18 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
                nn.BatchNorm2d(x_channels // 2))
        self.rnb_18 = BasicBlock(x_channels, x_channels//2, downsample=self.downsample_18)

        x_channels = x_channels // 2
        self.rnb_19 = BasicBlock(x_channels, x_channels)
        self.rnb_20 = BasicBlock(x_channels, x_channels)
        # self.downsample_21 = nn.Sequential(
        #         nn.Conv2d(x_channels, x_channels // 2, kernel_size=3, padding=1, stride=1, bias=True),
        #         nn.BatchNorm2d(x_channels // 2))
        self.rnb_21 = BasicBlock(x_channels, x_channels)

        self.Conv_gen_final = nn.Conv2d(x_channels, 1, 3, padding=1)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x = F.selu_(self.bn1(self.l1(z)))
        x = F.selu_(self.bn2(self.l2(x)))
        x = x.view([x.shape[0]] + self.ishape)

        x = self.rnb_1(x)
        x = self.rnb_2(x)
        x = self.rnb_3(x) # 64x4x4
        x = F.interpolate(x, scale_factor=2, mode='bicubic') # 64x8x8
        x = self.rnb_4(x)
        x = self.rnb_5(x)
        x = self.rnb_6(x) # 32x8x8
        x = F.interpolate(x, scale_factor=2, mode='bicubic') # 32x16x16
        x = self.rnb_7(x)
        x = self.rnb_8(x)
        x = self.rnb_9(x) # 16x16x16
        x = F.interpolate(x, scale_factor=2, mode='bicubic') # 16x32x32
        x = self.rnb_10(x)
        x = self.rnb_11(x)
        x = self.rnb_12(x) # 8x32x32
        x = F.interpolate(x, scale_factor=2, mode='bicubic') # 8x64x64
        x = self.rnb_13(x)
        x = self.rnb_14(x)
        x = self.rnb_15(x) # 4x64x64
        x = F.interpolate(x, scale_factor=2, mode='bicubic') # 4x128x128

        x = self.rnb_16(x)
        x = self.rnb_17(x)
        x = self.rnb_18(x)  # 2x128x128
        x = F.interpolate(x, scale_factor=2, mode='bicubic')  # 2x256x256

        x = self.rnb_19(x)
        x = self.rnb_20(x)
        x = self.rnb_21(x)  # 2x256x256
        x_gen = torch.sigmoid(self.Conv_gen_final(x)) # 1x256x256

        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (32, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.verbose = verbose

        x_channels = 1
        self.rnb_1 = BasicBlock(x_channels, x_channels)
        self.rnb_2 = BasicBlock(x_channels, x_channels)
        self.downsample_3 = nn.Sequential(nn.Conv2d(x_channels, 8, kernel_size=3, padding=1, stride=2, bias=True),
                                          nn.BatchNorm2d(8))
        self.rnb_3 = BasicBlock(x_channels, 8, downsample=self.downsample_3, stride=2) # 8x128x128
        x_channels = 8

        self.rnb_4 = BasicBlock(x_channels, x_channels)
        self.rnb_5 = BasicBlock(x_channels, x_channels)
        self.downsample_6 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels * 2, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(x_channels * 2))
        self.rnb_6 = BasicBlock(x_channels, x_channels * 2, downsample=self.downsample_6, stride=2) # 16x64x64
        x_channels = x_channels * 2

        self.rnb_7 = BasicBlock(x_channels, x_channels)
        self.rnb_8 = BasicBlock(x_channels, x_channels)
        self.downsample_9 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(x_channels))
        self.rnb_9 = BasicBlock(x_channels, x_channels, downsample=self.downsample_9, stride=2) # 16x32x32
        x_channels = x_channels

        self.rnb_10 = BasicBlock(x_channels, x_channels)
        self.rnb_11 = BasicBlock(x_channels, x_channels)
        self.downsample_12 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels * 2, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(x_channels * 2))
        self.rnb_12 = BasicBlock(x_channels, x_channels * 2, downsample=self.downsample_12, stride=2) # 32x16x16
        x_channels = x_channels * 2

        self.rnb_13 = BasicBlock(x_channels, x_channels)
        self.rnb_14 = BasicBlock(x_channels, x_channels)
        self.downsample_15 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(x_channels))
        self.rnb_15 = BasicBlock(x_channels, x_channels, downsample=self.downsample_15, stride=1) # 32x16x16
        x_channels = x_channels

        self.rnb_16 = BasicBlock(x_channels, x_channels)
        self.rnb_17 = BasicBlock(x_channels, x_channels)
        self.downsample_18 = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(x_channels))
        self.rnb_18 = BasicBlock(x_channels, x_channels, downsample=self.downsample_18, stride=2) # 32x8x8
        x_channels = x_channels

        self.linear_19 = nn.Linear(self.iels, 1024)
        self.bn_19 = nn.BatchNorm1d(1024)

        self.linear_20 = nn.Linear(1024, self.latent_dim + self.n_c)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, x):
        x = self.rnb_1(x)
        x = self.rnb_2(x)
        x = self.rnb_3(x)
        x = self.rnb_4(x)
        x = self.rnb_5(x)
        x = self.rnb_6(x)
        x = self.rnb_7(x)
        x = self.rnb_8(x)
        x = self.rnb_9(x)
        x = self.rnb_10(x)
        x = self.rnb_11(x)
        x = self.rnb_12(x)
        x = self.rnb_13(x)
        x = self.rnb_14(x)
        x = self.rnb_15(x)
        x = self.rnb_16(x)
        x = self.rnb_17(x)
        x = self.rnb_18(x)

        x = x.view((x.shape[0], -1))
        x = F.selu_(self.bn_19(self.linear_19(x)))
        z_img = self.linear_20(x)

        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = F.softmax(zc_logits, dim=1)

        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (32, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.wass = wass_metric
        self.verbose = verbose

        x_channels = 1
        self.rnb_1 = BasicBlock(x_channels, x_channels)
        self.downsample_3 = nn.Sequential(nn.Conv2d(x_channels, 8, kernel_size=3, padding=1, stride=2, bias=True),
                                          nn.BatchNorm2d(8))
        self.rnb_3 = BasicBlock(x_channels, 8, downsample=self.downsample_3, stride=2) # 8x128x128
        x_channels = 8

        self.rnb_4 = BasicBlock(x_channels, x_channels)
        self.downsample_6 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels * 2, kernel_size=3, padding=1, stride=2, bias=True),
                nn.BatchNorm2d(x_channels * 2))
        self.rnb_6 = BasicBlock(x_channels, x_channels * 2, downsample=self.downsample_6, stride=2) # 16x64x64
        x_channels = x_channels * 2

        self.rnb_7 = BasicBlock(x_channels, x_channels)
        self.downsample_9 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=2, bias=True),
                nn.BatchNorm2d(x_channels))
        self.rnb_9 = BasicBlock(x_channels, x_channels, downsample=self.downsample_9, stride=2) # 16x32x32
        x_channels = x_channels

        self.rnb_10 = BasicBlock(x_channels, x_channels)
        self.downsample_12 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels * 2, kernel_size=3, padding=1, stride=2, bias=True),
                nn.BatchNorm2d(x_channels * 2))
        self.rnb_12 = BasicBlock(x_channels, x_channels * 2, downsample=self.downsample_12, stride=2) # 32x16x16
        x_channels = x_channels * 2

        self.rnb_13 = BasicBlock(x_channels, x_channels)
        self.downsample_15 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=1, bias=True),
                nn.BatchNorm2d(x_channels))
        self.rnb_15 = BasicBlock(x_channels, x_channels, downsample=self.downsample_15, stride=1) # 32x16x16
        x_channels = x_channels

        self.rnb_16 = BasicBlock(x_channels, x_channels)
        self.downsample_18 = nn.Sequential(
                nn.Conv2d(x_channels, x_channels, kernel_size=3, padding=1, stride=2, bias=True),
                nn.BatchNorm2d(x_channels))
        self.rnb_18 = BasicBlock(x_channels, x_channels, downsample=self.downsample_18, stride=2) # 32x8x8 (=4096)
        x_channels = x_channels

        self.linear_19 = nn.Linear(self.iels, 1024)
        self.bn_19 = nn.BatchNorm1d(1024)

        self.linear_20 = nn.Linear(1024, 32)
        self.bn_20 = nn.BatchNorm1d(32)

        self.linear_21 = nn.Linear(32, 1)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, x):
        x = self.rnb_1(x)
        x = self.rnb_3(x)
        x = self.rnb_4(x)
        x = self.rnb_6(x)
        x = self.rnb_7(x)
        x = self.rnb_9(x)
        x = self.rnb_10(x)
        x = self.rnb_12(x)
        x = self.rnb_13(x)
        x = self.rnb_15(x)
        x = self.rnb_16(x)
        x = self.rnb_18(x)

        x = x.view((x.shape[0], -1))
        x = F.selu_(self.bn_19(self.linear_19(x)))
        x = F.selu_(self.bn_20(self.linear_20(x)))
        x = self.linear_21(x)

        if (not self.wass):
            x = torch.sigmoid(x)

        return x
