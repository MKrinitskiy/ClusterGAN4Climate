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
                 augment=True):
                 # batch_size = 8):
        super(CustomMNISTdataset, self).__init__()
        assert DoesPathExistAndIsFile(mnist_fname)

        self.mnist_fname = mnist_fname
        # self.batch_size = batch_size
        mnist_data = np.load(self.mnist_fname)
        if train_set:
            self.x = mnist_data['x_train']
            self.y = mnist_data['y_train']
        else:
            self.x = mnist_data['x_test']
            self.y = mnist_data['y_test']

        # batches = list(self.x.shape)[0] // self.batch_size
        # if batches * self.batch_size < list(self.x.shape)[0]:
        #     batches = batches + 1
        # self.batches_number = batches
        # self.dataset_indices = np.arange(batches)

        if augment:
            self.seq = iaa.Sequential([iaa.Resize((256,256)),
                                       # iaa.GaussianBlur(sigma=10.0),
                                       iaa.Affine(rotate=(-5, 5),
                                                  translate_px=(-10, 10),
                                                  shear=(-5, 5))],
                                      random_order=False)
        else:
            self.seq = iaa.Sequential([iaa.Resize((256,256)),
                                       # iaa.GaussianBlur(sigma=10.0)
                                       ],
                                      random_order=False)


    def __getitem__(self, item):
        # curr_batch = self.dataset_indices[item]
        # images = self.x[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size]
        # images = np.expand_dims(images, -1)
        # labels = self.y[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size]

        image = self.x[item]
        label = self.y[item]

        image = self.seq.augment_image(image)
        image_max = image.max().astype(np.float)
        image = image.astype(np.float)/image_max
        image = np.expand_dims(image, 0) # C,H,W
        # images = np.transpose(images, (0, 3, 1, 2)) # N,C,H,W
        image = torch.from_numpy(image).float()
        # labels = torch.from_numpy(labels).float()
        label = torch.from_numpy(np.array([[label]])).float()

        return image, label

    def __len__(self):
        return self.x.shape[0]
        # return self.batches_number