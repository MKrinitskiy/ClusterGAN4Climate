try:
    import argparse
    import os
    import numpy as np
    import sys


    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import pandas as pd

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image

    from itertools import chain as ichain

    # from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from libs.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from libs.datasets import get_dataloader, dataset_list
    from libs.service_defs import find_files, EnsureDirectoryExists
    from tqdm import tqdm
    import re, pickle

except ImportError as e:
    print(e)
    raise ImportError


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Encoding a dataset using ClusterGAN-trained encoder")
    parser.add_argument('--run-name', dest='run_name', help="Training run directory")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--snapshot-final', dest='snapshot_final', action='store_true',
                       help='flag for using the final snapshot of the model (default behaviour)',
                       default=argparse.SUPPRESS)
    group.add_argument('--snapshot-stage', dest='snapshot_stage', type=int, default=argparse.SUPPRESS,
                       help='stage (epoch) of training for loading models snapshot (most close one will be involved '+
                            'in case there is no snapshots for this exact epoch);\n' +
                            '-1 means the last one except \'final\'')
    group.add_argument('--snapshot-all', dest='snapshot_all', action='store_true',
                       help='flag for encoding the data using all the snapshots made during training',
                       default=argparse.SUPPRESS)
    parser.add_argument('--num-examples', dest="num_examples", default=argparse.SUPPRESS, type=int, help="Number of samples")
    parser.add_argument('--batch-size', dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument('--dataset-train', dest='dataset_train', action='store_true', help='The flag indicating the encoding either of the training (if set) subset of data, or test subset (if not set, default)')
    parser.add_argument('--dataset-workers', dest='dataset_workers', type=int, default=1, help='number of workers preprocessing dataset')
    args = parser.parse_args(args)

    # num_examples = args.num_examples
    curr_run_name = args.run_name

    if 'snapshot_final' in args.__dict__.keys():
        snapshot_stage = 'final'
    elif 'snapshot_stage' in args.__dict__.keys():
        snapshot_stage = args.snapshot_stage
    elif 'snapshot_all' in args.__dict__.keys():
        snapshot_stage = 'all'

    DATASETS_DIR = os.path.join(os.path.abspath('./'), 'datasets')
    dataset_name = 'mnist'
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    logs_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name)
    models_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'models')
    encodings_dir = os.path.join(logs_dir, 'data_encodings')
    EnsureDirectoryExists(encodings_dir)


    #load parameters of the run
    try:
        # old behaviour
        train_df = pd.read_csv(os.path.join(logs_dir, 'training_details.csv'))
        latent_dim = train_df['latent_dim'][0]
        n_c = train_df['n_classes'][0]
    except:
        # new behaviour
        with open(os.path.join(logs_dir, 'train_details.pkl'), 'rb') as f:
            train_details = pickle.load(f)
        latent_dim = train_details['latent_dim']
        n_c = train_details['n_classes']

    cuda = True if torch.cuda.is_available() else False

    # Load encoder model

    if isinstance(snapshot_stage, int):
        enc_snapshot_fnames = [os.path.join(models_dir, 'ep%04d_encoder.pth.tar' % snapshot_stage)]
    elif snapshot_stage == 'final':
        enc_snapshot_fnames = [os.path.join(models_dir, '%s_encoder.pth.tar' % snapshot_stage)]
    elif snapshot_stage == 'all':
        enc_snapshot_fnames = find_files(models_dir, '*encoder*.pth.tar')
        enc_snapshot_fnames.sort()

    for snapshot_fname in tqdm(enc_snapshot_fnames):
        reex = r'(.+)_encoder\.pth\.tar'
        match = re.match(reex, os.path.basename(snapshot_fname))
        curr_stage = match.groups()[0]

        encoder = Encoder_CNN(latent_dim, n_c)
        encoder.load_state_dict(torch.load(snapshot_fname))
        encoder.cuda()
        encoder.eval()

        # Configure data loader
        train_set = (True if args.dataset_train else False)
        dataloader = get_dataloader(data_dir=data_dir, batch_size=args.batch_size, train_set=train_set, augment=False, num_workers=args.dataset_workers)
        num_examples = list(dataloader.dataset.data.shape)[0]
        if 'num_examples' in args.__dict__.keys():
            num_examples = args.num_examples
        STEPS_PER_EPOCH = num_examples // dataloader.batch_size
        if STEPS_PER_EPOCH * dataloader.batch_size < list(dataloader.dataset.data.shape)[0]:
            STEPS_PER_EPOCH = STEPS_PER_EPOCH + 1

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        #Get full data for encoding
        encoded = []
        labels = []
        zn = []
        zc_logits = []
        # for idx, (imgs, curr_labels) in tqdm(enumerate(dataloader), total=STEPS_PER_EPOCH):
        for idx, (imgs, curr_labels) in enumerate(dataloader):
            if idx >= STEPS_PER_EPOCH:
                break
            # curr_imgs, curr_labels = next(iter(dataloader))
            curr_c_imgs = Variable(imgs.type(Tensor), requires_grad=False)

            # Encode real images
            enc_zn, enc_zc, enc_zc_logits = encoder(curr_c_imgs)
            enc_zn = enc_zn.cpu().detach().numpy()
            enc_zc_logits = enc_zc_logits.cpu().detach().numpy()

            zn.append(enc_zn)
            zc_logits.append(enc_zc_logits)
            labels.append(curr_labels)

        labels = np.concatenate(labels, axis=0)
        zn = np.concatenate(zn, axis=0)
        zc_logits = np.concatenate(zc_logits, axis=0)


        fname = os.path.join(encodings_dir, '%s_%s_stage-%s_encoded.npz'%(dataset_name, ('train' if args.dataset_train else 'test'), curr_stage))
        np.savez(fname, zn=zn, zc_logits=zc_logits, labels=labels)
        print('embeddings saved to the file %s' % fname)


if __name__ == "__main__":
    main()
