try:
    import argparse, os, sys, re
    import numpy as np

    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd
    from itertools import chain as ichain
    from cuml import UMAP
    from libs.service_defs import DoesPathExistAndIsFile, DoesPathExistAndIsDirectory
except ImportError as e:
    print(e)
    raise ImportError


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="UMAP encoding script")
    parser.add_argument('--run-name', dest='run_name', help="Training run directory (for the plot to be placed in the right logs directory)", required=True)
    parser.add_argument('-f', '--file-path', dest='encodings_file_path', help="encodings file path", default=argparse.SUPPRESS, required=True)
    args = parser.parse_args(args)

    assert 'encodings_file_path' in args.__dict__.keys()
    assert DoesPathExistAndIsFile(args.encodings_file_path), f'path {args.encodings_file_path} does not exist or is not a file'

    curr_run_name = args.run_name
    logs_dir = os.path.join(os.path.abspath('./'), 'logs', curr_run_name)
    assert DoesPathExistAndIsDirectory(logs_dir), f'path {logs_dir} does not exist or not a directory'

    enc = np.load(args.encodings_file_path)
    zn = enc['zn']
    zc_logits = enc['zc_logits']
    labels = enc['labels']

    print('zn shape: ', zn.shape)
    print('zc logits shape: ', zc_logits.shape)
    print('true labels shape:', labels.shape)

    umap = UMAP(n_components=2, verbose=True, n_epochs=4096, learning_rate=0.1)
    umap_enc = umap.fit_transform(zn)
    classes = np.unique(labels)
    colors = cm.tab20(np.linspace(0.0, 1.0, len(classes)))
    colors = dict(zip(classes, colors))

    f = plt.figure(figsize=(6, 6), dpi=300)
    for label in classes:
        plt.scatter(umap_enc[labels == label, 0], umap_enc[labels == label, 1], s=0.5, color=colors[label], label=label)
    lgnd = plt.legend(fontsize=5)
    for hndl in lgnd.legendHandles:
        hndl.set_sizes([20])
    plt.axis('equal')
    plt.tight_layout()

    figname = os.path.join(logs_dir, 'umap-%s.png' % os.path.splitext(os.path.basename(args.encodings_file_path))[0])
    f.savefig(figname, dpi=300, bbox_inches=0, pad_inches=0, )
    plt.close()


if __name__ == "__main__":
    main()
