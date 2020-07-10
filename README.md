ClusterGAN fork for climate data experiments.

This repository is mostly based on the repository [ClusterGAN: A PyTorch Implementation](https://github.com/zhampel/clusterGAN) which in turn is based on the [Tensorflow implementation](https://github.com/sudiptodip15/ClusterGAN) of [ClusterGAN](https://arxiv.org/abs/1809.03627)




## Requirements

This codebase is developed within the following environment:
```
python 3.6.9
pytorch 1.5.0
matplotlib 3.1.3
tqdm 4.47.0
numpy 1.18.1
seaborn 0.10.1
torchvision 0.6.0
```



## Run ClusterGAN on MNIST

We narrowed the scope of the applications to MNIST only. So the running of ClusterGAN on the MNIST dataset one may use the following command:
```bash
python train.py -r run_name -b 64 -n 300
```
where a directory `runs/mnist/run_name` will be made and contain the generated output (models, example generated instances, training figures) from the training run.

Options (the list will be extended soon):

 `-r` - run name,

`-b` - batch size

`-n` - number of training epochs.