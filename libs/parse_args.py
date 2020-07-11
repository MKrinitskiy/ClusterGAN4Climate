import argparse, warnings, datetime, os


def parse_args(args):
    """ Parse the arguments.
        """
    parser = argparse.ArgumentParser(description='Simple training script for training an autoencoder for SPV clustering')

    # sail_parser = subparsers.add_parser('sail')
    # sail_parser.add_argument('annotations',         help='Path to pickle-file containing annotations for training.')
    # sail_parser.add_argument('classes',             help='Path to a CSV file containing class label mapping.')
    # sail_parser.add_argument('--train-data-base-path', help='Path to the directory with the train data itself', required=True)
    # sail_parser.add_argument('--train-masks-base-path', help='Path to the directory with the masks for train data', required=True)
    # sail_parser.add_argument('--val-annotations',   help='Path to pickle-file containing annotations for validation (optional).')
    # sail_parser.add_argument('--val-data-base-path', help='Path to the directory with the validation data itself')
    # sail_parser.add_argument('--val-masks-base-path', help='Path to the directory with the masks for validation data')

    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('--snapshot',                help='Resume training from a snapshot.')

    parser.add_argument('--run-name',           dest="run_name",            help='name for the current run (directories will be created based on this name)', default='test_run')
    parser.add_argument('--batch-size',         dest='batch_size',          help='Size of the batches.', default=64, type=int)
    parser.add_argument('--val-batch-size',     dest='val_batch_size',      help='Size of the batches for evaluation.', default=32, type=int)
    parser.add_argument('--epochs',             dest="epochs",              help='Number of epochs to train.', type=int, default=200)
    parser.add_argument('--wass-metric',        dest="wass_metric",         help="Flag for Wasserstein metric", action='store_true')
    parser.add_argument('-–gpu',                dest="gpu",                 help="GPU id to use", default=0, type=int)
    parser.add_argument('--num-workers',        dest="num_workers",         help="Number of dataset workers", default=1, type=int)
    parser.add_argument('--snapshots-period',   dest="snapshots_period",    help="Save model snapshots every N epochs; -1 (default) for final models only", default=-1, type=int)
    parser.add_argument('--latent-dim',         dest='latent_dim',          help='(real) embeddings dimensionality', type=int, default=32)
    parser.add_argument('--cat-num',            dest='cat_num',             help='(int) embeddings dimensionality: number of categories', type=int, default=10)
    # parser.add_argument('--steps-per-epoch',        help='Number of steps per epoch.', type=int)
    # parser.add_argument('--val-steps',              help='Number of steps per validation run.', type=int, default=100)
    # parser.add_argument('--no-snapshots',           help='Disable saving snapshots.', dest='snapshots', action='store_false')
    # parser.add_argument('--debug',                  help='launch in DEBUG mode', dest='debug', action='store_true')

    # parser.add_argument('--model-only',             help='compose model only and output its description', dest='model_only', action='store_true')
    # parser.add_argument('--generator-multiplier',   help='generator batches per discriminator batches for the sake of training stability', dest='generator_multiplier', default=4, type=int)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
        For example, batch_size < num_gpus
        Intended to raise errors prior to backend initialisation.

        Args
            parsed_args: parser.parse_args()

        Returns
            parsed_args
        """

    parsed_args.snapshot_path = os.path.join('./snapshots', parsed_args.run_name)
    parsed_args.tensorboard_dir = os.path.join('./logs', parsed_args.run_name)
    parsed_args.logs_path = os.path.join('./logs', parsed_args.run_name)

    return parsed_args