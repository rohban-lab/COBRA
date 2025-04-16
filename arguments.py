from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of COBRA')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['cityscape', 'dagm', 'mvtecad', 'fashion-mnist', 'mnist', 'cifar10', 'cifar100', 'imagenet', 'svhn-10'],
                        default="cifar10", type=str)
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--model', help='Model',
                        choices=['resnet18', 'resnet18_imagenet'], default="resnet18", type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default="./cifar10_oc_class0.model", type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    parser.add_argument("--train_time_adv_evaluate", help='train_time_adv_evaluate',
                        action='store_true')
    parser.add_argument("--train_time_clean_evaluate", help='train_time_clean_evaluate',
                        action='store_true')

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=64, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=64, type=int)
    parser.add_argument('--timer', default=400000000000, type=int)
    
    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['COBRA'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'classification'],
                        default=['simclr', 'classification'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=10, type=int)
    parser.add_argument("--evaluate_save_step", help='',
                        default=20, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--K_classification", help='',
                        default=2, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", default=True, help='print quantiles of ood score',
                        action='store_true')


    parser.add_argument('--attack_type', type=str, default='linf',
                        help='adversarial l_p')
    parser.add_argument('--epsilon', type=float, default=0.0314,
                        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--k', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--random_start', type=bool, default=True,
                        help='True for PGD')
    parser.add_argument('--regularize_to', default='other', type=str, help='original/other')
    parser.add_argument('--loss_type', type=str, default='sim', help='loss type for Rep')

    parser.add_argument('--min', type=float, default=0.0, help='min for cliping image')
    parser.add_argument('--max', type=float, default=1.0, help='max for cliping image')
    parser.add_argument('--lamda', default=256, type=float)
    parser.add_argument("--in_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')
    parser.add_argument("--out_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')

    parser.add_argument('--eps', type=float, default=0.0314,
                        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    parser.add_argument('--steps', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')
    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()