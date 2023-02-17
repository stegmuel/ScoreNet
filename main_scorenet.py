from source.trainers.scorenet_trainer_fast import ScoreNetTrainerFast
from source.utils.utils import get_data_loaders
from source.utils.dino_utils import bool_flag
from source.models.scorenet import ScoreNet
from source.utils.utils import get_logger
import source.utils.dino_utils as utils
from torch import nn
import argparse
import wandb
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Score-Net', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='scorenet', type=str, choices=['scorenet'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--large_patch_size', default=128, type=int, help="""Size in pixels
        of input square patches in high resolution - default 128 (for 128x128 patches).""")
    parser.add_argument('--small_patch_size', default=16, type=int, help="""Size in pixels
        of input square patches in low resolution- default 8 (for 8x8 patches).""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=15, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.01, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--main_savepath', default='output/patch_net.pth')
    parser.add_argument('--main_loadpath', default='backup/merge/patch_net_0_0.4_2_5_merge_final.pth')

    # Dataset parameters
    parser.add_argument('--train_queries',
                        default=['/media/thomas/Samsung_T5/BRACS/BRACS_RoI/previous_1/train/*/*.png'], type=str,
                        help='Please specify path to the training data.')
    parser.add_argument('--valid_queries',
                        default=['/media/thomas/Samsung_T5/BRACS/BRACS_RoI/previous_1/val/*/*.png'], type=str,
                        help='Please specify path to the validation data.')
    parser.add_argument('--test_queries',
                        default=['/media/thomas/Samsung_T5/BRACS/BRACS_RoI/previous_1/test/*/*.png'],
                        type=str, help='Please specify path to the test data.')
    parser.add_argument('--n_classes', default=7, type=int)

    # Perturbed top-k
    parser.add_argument('--n_patches', default=20, type=int, help='Number of patches the student model sees.')
    parser.add_argument('--n_estimations', default=100, type=int,
                        help='Number of noisy estimations of the top-k operation to be performed.')
    parser.add_argument('--sigma', default=1e-3, type=float, help='Level of the noise applied to the scores.')
    parser.add_argument('--decay_noise', default=False, type=bool_flag)
    parser.add_argument('--sigma_decay', default=0.7, type=float, help='Multiplicative decaying factor for the noise '
                                                                       'level.')

    # Misc
    parser.add_argument('--logger', default='logs/log.txt')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--display_rate', default=50, type=int)
    parser.add_argument('--train', default=True, type=bool_flag)
    parser.add_argument('--n_cls_scorer', default=2, type=int)
    parser.add_argument('--clipping_value', default=4., type=float)
    parser.add_argument('--n_cls_patch', default=1, type=int)
    parser.add_argument('--mix_prob', default=0.5, type=float)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--augmentation', default='scoremix', type=str)
    parser.add_argument('--dataset', default='bracs', type=str)
    parser.add_argument('--scale', default=8, type=int)
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    return parser


def main(args):
    # Login to wandb
    wandb.init(project='patch-net', entity='stegmuel', mode='disabled')
    wandb.config.update(args)
    args = wandb.config

    # Mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # Get the model
    model = ScoreNet(args=args)

    # Compute the number of parameters
    n_parameters = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(n_parameters))

    # Load models
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.small_patch_size)

    # Get the classifier
    classifier = nn.Sequential(
        nn.Linear(in_features=model.embed_dim, out_features=args.hidden_dim),
        nn.ReLU(),
        nn.Linear(in_features=args.hidden_dim, out_features=args.n_classes),
    )

    # Start the wandb logger
    wandb.watch(model, log='all', log_freq=1)

    # Get the data loaders
    train_loader, valid_loader, test_loader = get_data_loaders(args)
    print(len(train_loader.dataset))

    # Define the logging info
    logger = get_logger(args.logger)

    # Instantiate the trainer
    trainer = ScoreNetTrainerFast(
        model=model,
        classifier=classifier,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        savepath=args.main_savepath,
        args=args,
        logger=logger,
        loadpath=args.main_loadpath,
        display_rate=args.display_rate,
        decay_noise=args.decay_noise,
        fp16_scaler=fp16_scaler,
        mix_prob=args.mix_prob,
        test_freq=args.test_freq,
        augmentation=args.augmentation
    )

    # Start training
    if args.train:
        # Train the model
        print('Training with {} augmentation method.'.format(args.augmentation))
        trainer.train(args.epochs)

        # Test the model
        with torch.no_grad():
            trainer.test(None)


if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser(description='Trains a model on the BACH dataset.',
                                     parents=[get_args_parser()])

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the training
    main(args)
