from __future__ import print_function

try:
    import numpy as np
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    import imgaug as ia
    from imgaug import augmenters as iaa
except ImportError as e:
    print(e)
    raise ImportError


DATASET_FN_DICT = {'mnist' : datasets.MNIST,
                   'fashion-mnist' : datasets.FashionMNIST}


dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):
    """
    Convenience function for retrieving
    allowed datasets.
    Parameters
    ----------
    name : {'mnist', 'fashion-mnist'}
          Name of dataset
    Returns
    -------
    fn : function
         PyTorch dataset
    """
    if dataset_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dataset_name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, DATASET_FN_DICT.keys()))



def get_dataloader(data_dir='', batch_size=64, train_set=True, num_workers=1, augment = True):

    if augment:
        seq = iaa.Sequential([iaa.Resize((64,64)),
                              iaa.Affine(rotate=(-5, 5),
                                         translate_px=(-10, 10),
                                         shear=(-5, 5))],
                             random_order=False)
    else:
        seq = iaa.Resize((64, 64))

    dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir,
                           train=train_set,
                           download=True,
                           transform=transforms.Compose([np.array,
                                                         seq.augment_images,
                                                         transforms.ToTensor(),
                                                         ])),
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True)

    return dataloader




# class CustomDataset(Dataset):
#     def __init__(self, n_images, n_classes, transform=None):
#         self.images = np.random.randint(0, 255,
#                                         (n_images, 224, 224, 3),
#                                         dtype=np.uint8)
#         self.targets = np.random.randn(n_images, n_classes)
#         self.transform = transform
#
#     def __getitem__(self, item):
#         image = self.images[item]
#         target = self.targets[item]
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, target
#
#     def __len__(self):
#         return len(self.images)