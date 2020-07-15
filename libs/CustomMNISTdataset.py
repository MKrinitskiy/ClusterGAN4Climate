try:
    from .service_defs import DoesPathExistAndIsFile
    from torch.utils.data import DataLoader, Dataset
    import torch
    import numpy as np
    import imgaug as ia
    from imgaug import augmenters as iaa
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F
    from PIL import Image
except ImportError as e:
    print(e)
    raise ImportError

class CustomMNISTdataset(Dataset):
    def __init__(self,
                 mnist_fname='./mnist_data/mnist.npz',
                 train_set=True,
                 augment=True,
                 batch_size = 8):
        super(CustomMNISTdataset, self).__init__()
        assert DoesPathExistAndIsFile(mnist_fname)

        self.mnist_fname = mnist_fname
        self.batch_size = batch_size
        mnist_data = np.load(self.mnist_fname)
        if train_set:
            self.x = mnist_data['x_train']
            self.y = mnist_data['y_train']
        else:
            self.x = mnist_data['x_test']
            self.y = mnist_data['y_test']

        batches = list(self.x.shape)[0] // self.batch_size
        if batches * self.batch_size < list(self.x.shape)[0]:
            batches = batches + 1
        self.batches_number = batches
        self.dataset_indices = np.arange(batches)

        if augment:
            self.seq = iaa.Sequential([iaa.Resize((256,256)),
                                       iaa.Affine(rotate=(-5, 5),
                                                  translate_px=(-10, 10),
                                                  shear=(-5, 5))],
                                      random_order=False)
        else:
            self.seq = iaa.Resize((64, 64))


    def __getitem__(self, item):
        curr_batch = self.dataset_indices[item]
        images = self.x[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size]
        images = np.expand_dims(images, -1)
        labels = self.y[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size]

        # image = self.x[item]
        # label = self.y[item]

        images = self.seq.augment_images(images)
        images = images/255.0
        images = np.transpose(images, (0, 3, 1, 2)) # N,C,H,W
        images = torch.from_numpy(images).float()
        labels = torch.from_numpy(labels).float()

        return images, labels

    def __len__(self):
        return self.batches_number