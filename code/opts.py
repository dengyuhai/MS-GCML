import argparse
import utils

def get_parser():
    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `MS-GCML`,freeze MiT patch embedding'
                                     )
    
    parser.add_argument('--output_dir',
                        default='./output/debug_log',
                        help='Path to log folder'
                        )
    parser.add_argument('--data_root', default='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/collect',
                        help='global data path of root derectory')
    parser.add_argument('--dataset', default='coco-voc', help='Training dataset, e.g. cub, cars, SOP')

    parser.add_argument('--embedding_size', default=512, type=int,
                        help='Size of embedding that is appended to backbone model.'
                        )
    parser.add_argument('--bg_embedding_size', default=1024, type=int,
                        help='Size of embedding that is appended to backbone model.'
                        )
    parser.add_argument('--batch-size', default=5*15, type=int,
                        dest='sz_batch',
                        help='Number of samples per batch.'
                        )
    parser.add_argument('--loops', default=20, type=int,
                        dest='nb_loops',
                        help='Number of training loops.'
                        )
    parser.add_argument('--gpu-id', default=-1, type=int,
                        help='ID of GPU that is used for training, and "-1" means DataParallel'
                        )
    parser.add_argument('--workers', default=8, type=int,
                        dest='nb_workers',
                        help='Number of workers for dataloader.'
                        )
    parser.add_argument('--model', default='vit_MiT',#'vit_moco',#'vit_clip',#'googlenet',#'vit_dino'
                        help='Model for training'
                        )
    parser.add_argument('--task', default=None, type=str,
                        help='ovor task'
                        )
    parser.add_argument('--extractWay', default='SAM', type=str,
                        help='the way of box extract'
                        )
    parser.add_argument('--IoUthr', default='0.3', type=str,
                        help='the thr of IoU for corrected box recall'
                        )
    parser.add_argument('--step', default='val', type=str,
                        help='val or  test'
                        )
    parser.add_argument('--optimizer', default='adamp',
                        help='Optimizer setting'
                        )
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate setting'
                        )
    parser.add_argument('--emb-lr', default=3e-5, type=float,
                        help='Learning rate for embedding layer'
                        )
    parser.add_argument('--fix_lr', default=False, type=utils.bool_flag,
                        help='Learning rate Fixing'
                        )
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay setting'
                        )
    parser.add_argument('--num_neighbors', default=5, type=int,
                        help='For balanced sampling, the number of neighbors per query'
                        )
    parser.add_argument('--bn_freeze', default=0, type=int,
                        help='Batch normalization parameter freeze'
                        )
    parser.add_argument('--student_norm', default=0, type=int,
                        help='student L2 normalization'
                        )
    parser.add_argument('--teacher_norm', default=1, type=int,
                        help='teacher L2 normalization'
                        )
    parser.add_argument('--s_resume', default='',
                        help='Loading checkpoint'
                        )
    parser.add_argument('--t_resume', default='',help='Loading checkpoint'
                        )
    parser.add_argument('--view', default=2, type=int,
                        help='choose augmentation view'
                        )
    parser.add_argument('--delta', default=1, type=float,
                        help='Delta value of Relaxed Contrastive Loss'
                        )
    parser.add_argument('--sigma', default=3, type=float,
                        help='Sigma value of Relaxed Contrastive Loss'
                        )
    parser.add_argument('--momentum', default=0.999, type=float,
                        help='Momentum Update Parameter'
                        )
    parser.add_argument('--pretrained', default=True, type=utils.bool_flag,
                        help='Training with ImageNet pretrained model'
                        )
    parser.add_argument('--swav', default=False, type=utils.bool_flag,
                        help='Training with SwAV pretrained model'
                        )

    parser.add_argument('--remark', default='',
                        help='Any reamark'
                        )
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--val_interval', default=200, type=int,
                        help='every how many batch tested on the validation set ')
    parser.add_argument('--loss_interval', default=1, type=int,
                        help='Number of intervals for monitoring loss')
    parser.add_argument('--cluster_num', default='100', type=str,
                        help='Number of clusters for clustering loss'
                        )
    parser.add_argument('--group_num', default=4, type=int,
                        help='Number of groups for GCL'
                        )
    parser.add_argument('--fz_backbone', default=False, type=bool,
                        help='freeze backbone or not'
                        )
    parser.add_argument('--random_sample', default=True, type=bool,
                        help='random_sample or not'
                        )
    parser.add_argument('--iter_per_loop', default=900, type=int,
                        help='How many itereations to update the distance matrix'
                        )
    # parser.add_argument('--fz_momentum', default=1000, type=int,
    #                     help='How many itereations to stop updating the teachers network'
    #                     )
    parser.add_argument('--momentum_batch', default=10, type=int,
                        help='momentum update teacher network interval'
                        )
    parser.add_argument('--loss_type', default='GCL', type=str,
                        help='Loss of ablation experiments'
                        )
    return parser
