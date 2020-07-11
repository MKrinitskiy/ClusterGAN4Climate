from __future__ import print_function

try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty

    from torchsummary import summary
except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


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
        self.ishape = [128, 7, 7]
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.l1 = nn.Linear(self.latent_dim + self.n_c, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.l2 = nn.Linear(1024, self.iels)
        self.bn2 = nn.BatchNorm1d(self.iels)

        self.Conv_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.Conv_2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.Conv_3 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.Conv_4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)

        self.Conv_5 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(8)

        self.Conv_6 = nn.Conv2d(8, 1, 3, padding=1)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x = F.selu_(self.bn1(self.l1(z)))
        x = F.selu_(self.bn2(self.l2(x)))
        x = x.view([x.shape[0]] + self.ishape)
        x = F.selu_(self.bn3(self.Conv_1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = F.selu_(self.bn4(self.Conv_2(x)))
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = F.selu_(self.bn5(self.Conv_3(x)))
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = F.selu_(self.bn6(self.Conv_4(x)))
        x = F.interpolate(x, size=64, mode='bicubic')
        x = F.selu_(self.bn7(self.Conv_5(x)))
        x_gen = torch.sigmoid(self.Conv_6(x))
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

        self.Conv_1 = nn.Conv2d(self.channels, 8, 3, stride=2, padding=1)  # 32x32
        self.bn1 = nn.BatchNorm2d(8)

        self.Conv_2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)  # 16x16
        self.bn2 = nn.BatchNorm2d(16)

        self.Conv_3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 8x8
        self.bn3 = nn.BatchNorm2d(32)

        self.l4 = nn.Linear(self.iels, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.l5 = nn.Linear(1024, self.latent_dim + self.n_c)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, x):
        x = F.selu_(self.bn1(self.Conv_1(x)))
        x = F.selu_(self.bn2(self.Conv_2(x)))
        x = F.selu_(self.bn3(self.Conv_3(x)))
        x = x.view((x.shape[0], -1))
        x = F.selu_(self.bn4(self.l4(x)))
        z_img = self.l5(x)

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
        self.cshape = (64, 8, 8)
        self.iels = int(np.prod(self.cshape))
        self.wass = wass_metric
        self.verbose = verbose

        self.Conv_1 = nn.Conv2d(self.channels, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.bn1 = nn.BatchNorm2d(16)

        self.Conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.bn2 = nn.BatchNorm2d(32)

        self.Conv_3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 64 x 8 x 8
        self.bn3 = nn.BatchNorm2d(64)

        self.l4 = nn.Linear(self.iels, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.l5 = nn.Linear(1024, 32)
        self.bn5 = nn.BatchNorm1d(32)

        self.l6 = nn.Linear(32, 1)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, x):
        x = F.selu_(self.bn1(self.Conv_1(x)))
        x = F.selu_(self.bn2(self.Conv_2(x)))
        x = F.selu_(self.bn3(self.Conv_3(x)))
        x = x.view((x.shape[0], -1))
        x = F.selu_(self.bn4(self.l4(x)))
        x = F.selu_(self.bn5(self.l5(x)))
        x = self.l6(x)

        if (not self.wass):
            x = torch.sigmoid(x)

        return x
