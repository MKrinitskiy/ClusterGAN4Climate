from __future__ import print_function

try:
    import argparse, os, sys, pickle
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

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

    # from libs.definitions import *
    from libs.models_residual import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from libs.utils import save_models, calc_gradient_penalty, sample_z, cross_entropy
    # from libs.datasets import get_dataloader, dataset_list
    from libs.plots import plot_train_loss
    from tqdm import tqdm
    from libs.parse_args import *
    from libs.service_defs import *
    from os.path import join, isfile, isdir
    from libs.copy_tree import copytree_multi
    from torch.utils.tensorboard import SummaryWriter
    from libs.CustomMNISTdataset import *
except ImportError as e:
    print(e)
    raise ImportError

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    torch.autograd.set_detect_anomaly(True)

    curr_run_name = args.run_name
    EPOCHS = args.epochs
    if 'steps_per_epoch' in args:
        STEPS_PER_EPOCH = args.steps_per_epoch
    else:
        STEPS_PER_EPOCH = None

    #region preparations
    #region DATASETS_DIR
    DATASETS_DIR = os.path.join(os.path.abspath('./'), 'datasets')
    try:
        EnsureDirectoryExists(DATASETS_DIR)
    except:
        print('datasets directory couldn`t be found and couldn`t be created:\n%s' % DATASETS_DIR)
        raise FileNotFoundError('datasets directory couldn`t be found and couldn`t be created:\n%s' % DATASETS_DIR)
    #endregion

    #region data_dir
    dataset_name = 'mnist'
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    try:
        EnsureDirectoryExists(data_dir)
    except:
        print('data_dir couldn`t be found and couldn`t be created:\n%s' % data_dir)
        raise FileNotFoundError('data_dir couldn`t be found and couldn`t be created:\n%s' % data_dir)
    #endregion

    #region logs_dir
    logs_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name)
    try:
        EnsureDirectoryExists(logs_dir)
    except:
        print('logs directory couldn`t be found and couldn`t be created:\n%s' % logs_dir)
        raise FileNotFoundError('logs directory couldn`t be found and couldn`t be created:\n%s' % logs_dir)
    #endregion

    #region imgs_dir
    imgs_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'images')
    try:
        EnsureDirectoryExists(imgs_dir)
    except:
        print('output images directory couldn`t be found and couldn`t be created:\n%s' % imgs_dir)
        raise FileNotFoundError('output images directory couldn`t be found and couldn`t be created:\n%s' % imgs_dir)
    #endregion

    #region scripts_backup_dir
    scripts_backup_dir = os.path.join(os.path.abspath('./'), 'scripts_backup', curr_run_name)
    try:
        EnsureDirectoryExists(scripts_backup_dir)
    except:
        print('scripts backup directory couldn`t be found and couldn`t be created:\n%s' % scripts_backup_dir)
        raise FileNotFoundError('scripts backup directory couldn`t be found and couldn`t be created:\n%s' % scripts_backup_dir)
    #endregion

    #region models_dir
    models_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'models')
    try:
        EnsureDirectoryExists(models_dir)
    except:
        print('models snapshots directory couldn`t be found and couldn`t be created:\n%s' % models_dir)
        raise FileNotFoundError('models snapshots directory couldn`t be found and couldn`t be created:\n%s' % models_dir)
    #endregion

    # region models_dir
    tboard_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name, 'TBoard')
    try:
        EnsureDirectoryExists(tboard_dir)
    except:
        print('Tensorboard directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir)
        raise FileNotFoundError('Tensorboard directory directory couldn`t be found and couldn`t be created:\n%s' % tboard_dir)
    # endregion

    # endregion preparations

    #region backing up the scripts configuration
    ignore_func = lambda dir, files: [f for f in files if (isfile(join(dir, f)) and f[-3:] != '.py')] + [d for d in files if ((isdir(d)) & (d.endswith('scripts_backup') |
                                                                                                                                            d.endswith('.ipynb_checkpoints') |
                                                                                                                                            d.endswith('__pycache__') |
                                                                                                                                            d.endswith('build') |
                                                                                                                                            d.endswith('datasets') |
                                                                                                                                            d.endswith('logs') |
                                                                                                                                            d.endswith('runs') |
                                                                                                                                            d.endswith('snapshots')))]
    copytree_multi('./', scripts_backup_dir, ignore=ignore_func)
    #endregion




    #region setting training parameters
    num_workers = args.num_workers
    if args.debug:
        num_workers = 0


    batch_size = args.batch_size
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    n_skip_iter = 1 #5

    img_size = 256
    channels = 1
   
    # Latent space info
    latent_dim = args.latent_dim
    n_c = args.cat_num
    betan = 10
    betac = 10
   
    wass_metric = args.wass_metric
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'
    
    x_shape = (channels, img_size, img_size)
    #endregion
    
    cuda = True if torch.cuda.is_available() else False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if cuda:
        torch.cuda.set_device(0)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Initialize generator and discriminator
    generator = Generator_CNN(latent_dim, n_c, x_shape, verbose=True)
    encoder = Encoder_CNN(latent_dim, n_c)
    discriminator = Discriminator_CNN(wass_metric=wass_metric)
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    def collation(data):
        x_tensors = [d[0] for d in data]
        y_tensors = [d[1] for d in data]
        x_tensor = torch.cat(x_tensors, dim=0)
        y_tensor = torch.cat(y_tensors, dim=0)
        return x_tensor, y_tensor

    train_dataset = CustomMNISTdataset(mnist_fname=os.path.join(DATASETS_DIR, 'mnist', 'MNIST', 'mnist.npz'),
                                       train_set=True, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                  # collate_fn=collation)

    # dataloader = get_dataloader(data_dir=data_dir,
    #                             batch_size=batch_size,
    #                             num_workers=num_workers,
    #                             train_set=True,
    #                             augment=True)

    # Test data loader
    # test_dataloader = get_dataloader(data_dir=data_dir, batch_size=batch_size, train_set=False, augment=False)
    test_dataset = CustomMNISTdataset(mnist_fname=os.path.join(DATASETS_DIR, 'mnist', 'MNIST', 'mnist.npz'),
                                       train_set=False, augment=True)
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                 # collate_fn=collation)
   
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []
    
    c_zn = []
    c_zc = []
    c_i = []

    if STEPS_PER_EPOCH is None:
        STEPS_PER_EPOCH = len(train_dataset)//train_dataloader.batch_size
        if STEPS_PER_EPOCH * train_dataloader.batch_size < len(train_dataset):
            STEPS_PER_EPOCH = STEPS_PER_EPOCH+1

    EVAL_STEPS = len(test_dataset)//test_dataloader.batch_size
    if EVAL_STEPS*test_dataloader.batch_size < len(test_dataset):
        EVAL_STEPS = EVAL_STEPS+1

    #region saving train details for in-train-time analysis
    train_details = {'run_name': curr_run_name,
                     'EPOCHS': EPOCHS,
                     'STEPS_PER_EPOCH': STEPS_PER_EPOCH,
                     'EVAL_STEPS': EVAL_STEPS,
                     'learning_rate': lr,
                     'beta_1': b1,
                     'beta_2': b2,
                     'weight_decay': decay,
                     'n_skip_iter': n_skip_iter,
                     'latent_dim': latent_dim,
                     'n_classes': n_c,
                     'beta_n': betan,
                     'beta_c': betac,
                     'wass_metric': wass_metric,
                     }
    # this file will be rewritten lately
    with open(os.path.join(logs_dir, 'train_details.pkl'), 'wb') as f:
        pickle.dump(train_details, f)
    # region saving train details for in-train-time analysis

    writer = SummaryWriter(log_dir=tboard_dir)

    #region Training loop
    print('\nBegin training session with %i epochs...\n'%(EPOCHS))
    for epoch in range(EPOCHS):
        generator.train()
        encoder.train()
        for idx, (imgs, itruth_label) in tqdm(enumerate(train_dataloader), total=STEPS_PER_EPOCH):
            if idx >= STEPS_PER_EPOCH:
                break

            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            #region Train Generator + Encoder
            optimizer_GE.zero_grad()
            
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                      latent_dim=latent_dim,
                                      n_c=n_c,
                                      cuda_available=cuda)

            # Generate a batch of images
            gen_imgs = generator(zn, zc)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)

            # if args.debug:
            #     continue
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (idx % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)
    
                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
                #zc_loss = cross_entropy(enc_gen_zc_logits, zc)
    
                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    # Vanilla GAN loss
                    valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss
    
                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()
            #endregion


            #region Train Discriminator
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs, cuda_available=cuda)

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()
            #endregion

        # if args.debug:
        #     continue
        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())


        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp


        #region Cycle through test real -> enc -> gen
        print('cycle evaluation...')
        cycle_eval_mse_losses = []
        for idx, (eval_imgs, eval_labels) in tqdm(enumerate(test_dataloader), total=EVAL_STEPS):
            # test_imgs, test_labels = next(iter(testdata))
            eval_imgs = Variable(eval_imgs.type(Tensor))

            # t_imgs, t_label = test_imgs.data, test_labels

            # Encode sample real instances
            e_tzn, e_tzc, e_tzc_logits = encoder(eval_imgs)
            # Generate sample instances from encoding
            teg_imgs = generator(e_tzn, e_tzc)
            # Calculate cycle reconstruction loss
            img_mse_loss = mse_loss(eval_imgs, teg_imgs)
            # Save img reco cycle loss
            cycle_eval_mse_losses.append(img_mse_loss.item())
        c_i.append(np.mean(cycle_eval_mse_losses))
        #endregion
       

        #region Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=latent_dim,
                                                 n_c=n_c,
                                                 cuda_available=cuda)
        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)
        # Encode sample instances
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)
        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        #lat_xe_loss = cross_entropy(zc_e_logits, zc_samp)
        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())
        #endregion

        #region write losses to tensorboard
        writer.add_scalar('Loss/train/gen_enc_loss', ge_loss.item(), epoch)
        writer.add_scalar('Loss/train/dsc_loss', d_loss.item(), epoch)
        writer.add_scalar('Loss/train/cycle/lat_mse_loss', lat_mse_loss.item(), epoch)
        writer.add_scalar('Loss/train/cycle/lat_xe_loss', lat_xe_loss.item(), epoch)
        writer.add_scalar('Loss/train/cycle/img_mse_loss', img_mse_loss.item(), epoch)
        #endregion

        #region Save cycled and generated examples!
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        save_image(r_imgs.data[:n_samp],
                   '%s/real_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp],
                   '%s/reg_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp],
                   '%s/gen_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_sqrt_samp, normalize=True)
        #endregion
        
        #region Generate samples for specified classes
        stack_imgs = []
        for idx_gen in range(n_c):
            # Sample specific class
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                     latent_dim=latent_dim,
                                                     n_c=n_c,
                                                     fix_class=idx_gen,
                                                     cuda_available=cuda)

            # Generate sample instances
            gen_imgs_samp = generator(zn_samp, zc_samp)

            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        save_image(stack_imgs,
                   '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                   nrow=n_c, normalize=True)
        #endregion


        #region snapshots
        if args.snapshots_period == -1 or args.snapshots_period == 0:
            pass
        elif args.snapshots_period > 1:
            if epoch % args.snapshots_period == 0:
                model_list = [discriminator, encoder, generator]
                save_models(models=model_list, out_dir=models_dir, stage=epoch)

        #endregion

        print ('[Epoch %d/%d]' % (epoch, EPOCHS))
        print("\tModel Losses: [D: %f] [GE: %f]" % (d_loss.item(), ge_loss.item()))
        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]" % (img_mse_loss.item(), lat_mse_loss.item(), lat_xe_loss.item()))
    #endregion


    # if args.debug:
    #     sys.exit()

    # Save training results
    train_details = {'run_name'         : curr_run_name,
                     'EPOCHS'           : EPOCHS,
                     'STEPS_PER_EPOCH'  : STEPS_PER_EPOCH,
                     'learning_rate'    : lr,
                     'beta_1'           : b1,
                     'beta_2'           : b2,
                     'weight_decay'     : decay,
                     'n_skip_iter'      : n_skip_iter,
                     'latent_dim'       : latent_dim,
                     'n_classes'        : n_c,
                     'beta_n'           : betan,
                     'beta_c'           : betac,
                     'wass_metric'      : wass_metric,
                     'gen_enc_loss'     : {'label'  : 'G+E',
                                           'data'   : ge_l},
                     'disc_loss'        : {'label'  : 'D',
                                           'data'   : d_l},
                     'zn_cycle_loss'    : {'label'  : '$||Z_n-E(G(x))_n||$',
                                           'data'   : c_zn},
                     'zc_cycle_loss'    : {'label'  : '$||Z_c-E(G(x))_c||$',
                                           'data'   : c_zc},
                     'img_cycle_loss' : {'label'    : '$||X-G(E(x))||$',
                                         'data'     : c_i}
                     }
    with open(os.path.join(logs_dir, 'train_details.pkl'), 'wb') as f:
        pickle.dump(train_details, f)

    # Plot some training results
    plot_train_loss(train_details=train_details,
                    var_list=['gen_enc_loss', 'disc_loss'],
                    figname=os.path.join(logs_dir, 'training_model_losses.png')
                    )

    plot_train_loss(train_details=train_details,
                    var_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
                    figname=os.path.join(logs_dir, 'training_cycle_loss.png')
                    )


    # Save current state of trained models
    model_list = [discriminator, encoder, generator]
    save_models(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()
