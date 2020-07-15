from __future__ import print_function

try:
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt
    
except ImportError as e:
    print(e)
    raise ImportError


def plot_train_loss(train_details, var_list, figname='training_losses.png'):

    fig, ax = plt.subplots(figsize=(16,10))
    for arr_name in var_list:
        label = train_details[arr_name]['label']
        vals = train_details[arr_name]['data']
        epochs = range(0, len(vals))
        ax.plot(epochs, vals, label=r'%s'%(label))
    
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.set_title('Training Loss', fontsize=24)
    ax.grid()
    plt.legend(loc='upper right', numpoints=1, fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)
