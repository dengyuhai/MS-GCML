import os
import torch, math, time, argparse, random, warnings
import utils, dataset, loss
import numpy as np
import wandb
from net.googlenet import googlenet
from net.MiT import MiT_gcl
from net.clip import clip_gcl
from net.moco import moco_gcl
from torch import nn
from dataset import sampler
from dataset.sam_groups import GCL
from dataset.eval import validate_set
from tqdm import tqdm
def get_parser():
    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `MS-GCML`'
                                     )
    parser.add_argument('--LOG_DIR',
                        default='./debug_log',
                        help='Path to log folder'
                        )
    parser.add_argument('--DATA_DIR', default='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/debug',
                        help='Path of data')
    # parser.add_argument('--DATA_DIR', default='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata',
    #                     help='Path of data')
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
    parser.add_argument('--val_interval', default=100, type=int,
                        help='every how many batch tested on the validation set ')
    parser.add_argument('--loss_interval', default=1, type=int,
                        help='Number of intervals for monitoring loss')
    parser.add_argument('--cluster_num', default='100', type=str,
                        help='Number of clusters for clustering loss'
                        )
    parser.add_argument('--group_num', default=4, type=int,
                        help='Number of groups for GCL'
                        )
    parser.add_argument('--fz_backbone', default=True, type=bool,
                        help='freeze backbone or not'
                        )
    parser.add_argument('--random_sample', default=True, type=bool,
                        help='random_sample or not'
                        )
    parser.add_argument('--iter_per_loop', default=900, type=int,
                        help='How many itereations to update the distance matrix'
                        )
    parser.add_argument('--fz_momentum', default=1000, type=int,
                        help='How many itereations to stop updating the teachers network'
                        )
    parser.add_argument('--momentum_batch', default=100, type=int,
                        help='momentum update teacher network interval'
                        )
    return parser

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
    parser = get_parser()
    args = parser.parse_args()
    group_num = args.group_num
    inter_per_loop = args.iter_per_loop
    train_count = 1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
    # Directory for Log
    model_name = '{}_embSize{}_lr{}_batch{}'.format(args.model, args.embedding_size, args.lr, args.sz_batch)
    LOG_DIR = args.LOG_DIR + '/{}_{}/{}'.format(args.extractWay, args.dataset, model_name)
    os.makedirs(LOG_DIR,exist_ok=True)
    wandb.init(project='finetune{}_{}_all_{}'.format(args.dataset, args.extractWay, args.IoUthr), notes=LOG_DIR,
               name='{}_{}'.format(args.LOG_DIR, args.model))
    wandb.config.update(args)
    
    LOG_DIR = os.path.abspath(LOG_DIR)
    os.makedirs(LOG_DIR,exist_ok=True)
    DATA_DIR = os.path.abspath(args.DATA_DIR)
    results_dir = os.path.join(LOG_DIR, 'validate_results')
    detailed_base_dir = './output/{}'.format(args.LOG_DIR)
    os.makedirs(results_dir,exist_ok=True)
    pth_save_dir = os.path.join(LOG_DIR, 'pth')
    os.makedirs(pth_save_dir, exist_ok=True)
    # Student Model
    if args.model.find('googlenet') + 1:
        model_student = googlenet(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm,
                                    True, group_num=group_num)
        center_file = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/100center_feat_googlenet_init.pkl'
    elif args.model.find('vit_clip') + 1:
        model_student = clip_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm,
                                    True, group_num=group_num)
        center_file = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/100center_feat_clip_l2.pkl'
    elif args.model.find('vit_moco') + 1:
        model_student = moco_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm,
                                    True, group_num=group_num)
        center_file = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/100center_feat_moco_l2.pkl'
    elif args.model.find('MiT') + 1:
        print('student  ===> ViT_MiT')
        model_student = MiT_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.student_norm,
                                    True, group_num=group_num)
        center_file = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/100center_feat_MiT_l2.pkl'
    
    # Teacher Model
    if args.model.find('googlenet') + 1:
        model_teacher = googlenet(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm,
                                    False)
    elif args.model.find('vit_clip') + 1:
        model_teacher = clip_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm,
                                    False)
    elif args.model.find('vit_moco') + 1:
        model_teacher = moco_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm,
                                    False)
    elif args.model.find('MiT') + 1:
        print('teacher  ===> ViT_MiT')
        model_teacher = MiT_gcl(args.embedding_size, args.bg_embedding_size, args.pretrained, args.teacher_norm,
                                    False)
    
    
    if args.model.split('_')[1] == 'clip' or args.model.split('_')[1] == 'moco' or args.model.split('_')[1]=='MiT':
        print('use model.prepcocess')
        defined_transform = model_student.preprocess
    else:
        defined_transform = None
    is_inception = (args.model == 'bn_inception' or args.model == 'googlenet')
    model_student = model_student.cuda()
    model_teacher = model_teacher.cuda()
    for param in list(set(model_teacher.parameters())):
        param.requires_grad = False

    if os.path.isfile(args.t_resume):
        print('=> teacher Loading Checkpoint {}'.format(args.t_resume))
        checkpoint = torch.load(args.t_resume, map_location='cpu'.format(0))
        model_teacher.load_state_dict(checkpoint['model_state_dict'])

    else:
        print('=> teacher No Checkpoint {}!!!!!!!!!!!!!'.format(args.t_resume))

    if os.path.isfile(args.s_resume):
        print('=> student Loading Checkpoint {}'.format(args.s_resume))
        checkpoint = torch.load(args.s_resume, map_location='cpu'.format(0))
        model_student.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('=> student No Checkpoint {}!!!!!!!!!!!!!'.format(args.s_resume))
    if args.gpu_id == -1:
        # model_teacher = nn.DataParallel(model_teacher,output_device=1)
        # model_student = nn.DataParallel(model_student,output_device=1)
        model_teacher = nn.DataParallel(model_teacher)
        model_student = nn.DataParallel(model_student)
        
    dl_samplings = []
    detailed_info = []
    train_list = []
    for j in range(group_num):
        dataset_sampling = GCL(
        root=DATA_DIR,
        mode='train',
        transform=dataset.utils.Transform_for_Sampler(
            is_train=False,
            is_inception=is_inception,defined_transform=defined_transform),
        train_list=[],
        detail_info=[],
        group_name='group{}'.format(j+1),
        IoUthr=args.IoUthr,
        group_num=group_num,
        txt_id=train_count
        )
        detailed_info.append(dataset_sampling.detail_info)
        train_list.append(dataset_sampling.train_list)
        dl_sampling = torch.utils.data.DataLoader(
        dataset_sampling,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=False)
        dl_samplings.append(dl_sampling)
    train_count += 1
    print('\n*********the number of dl_samplings is{}********\n'.format(len(dl_samplings)))
    
    trn_dataset = GCL(
        root=DATA_DIR,
        mode='train',
        transform=dataset.utils.MultiTransforms(
            is_train=True,
            is_inception=is_inception,
            view=args.view,defined_transform=defined_transform),
        train_list=train_list,
        detail_info= detailed_info,
        group_name=None,
        IoUthr=args.IoUthr,
        group_num=group_num)

    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=False
    )
   
    query_dataset = validate_set(
        root=DATA_DIR,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=True,defined_transform=defined_transform),
    )
    gallery_dataset = validate_set(
        root=DATA_DIR,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=True,defined_transform=defined_transform)
    )
    dl_pred = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=False
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=False
    )
    stml_criterion = loss.STML_GCL_loss(delta=args.delta, sigma=args.sigma, view=args.view, disable_mu=args.student_norm,logger=None,
                                    topk=args.num_neighbors * args.view,cluster_num=args.cluster_num,center_file=center_file).cuda()
    # Momentum Update
    momentum_update = loss.Momentum_Update(momentum=args.momentum).cuda()
    
    # Train Parameters
    fc_layer_lr = args.lr#args.emb_lr if args.emb_lr else args.lr
    if args.gpu_id != -1:
        # for j in range(group_num):
        #     embedding_param = embedding_param + list(model_student.model.embedding_fs[j].parameters()) + list(
        #     model_student.model.embedding_gs[j].parameters())
        embedding_param = list(model_student.model.embedding_f1.parameters()) + list(
            model_student.model.embedding_g1.parameters())+\
        list(model_student.model.embedding_f2.parameters()) + list(
            model_student.model.embedding_g2.parameters()) +\
        list(model_student.model.embedding_f3.parameters()) + list(
            model_student.model.embedding_g3.parameters()) +\
        list(model_student.model.embedding_f4.parameters()) + list(
            model_student.model.embedding_g4.parameters())
    else:
        embedding_param = list(model_student.module.model.embedding_f1.parameters()) + list(
            model_student.module.model.embedding_g1.parameters())+ \
                          list(model_student.module.model.embedding_f2.parameters()) + list(
            model_student.module.model.embedding_g2.parameters()) +\
                          list(model_student.module.model.embedding_f3.parameters()) + list(
                    model_student.module.model.embedding_g3.parameters()) + \
                          list(model_student.module.model.embedding_f4.parameters()) \
                          + list(model_student.module.model.embedding_g4.parameters())
    if args.fz_backbone:
        print('****************  freeze backbone  ************************\n')
        if args.gpu_id != -1:
            backbone_modules = list(set(model_student.model.parameters()).difference(set(embedding_param)))
        else:
            backbone_modules = list(set(model_student.module.model.parameters()).difference(set(embedding_param)))
        for param in backbone_modules:
            param.requires_grad = False
        param_groups = [
            {'params': embedding_param, 'lr': fc_layer_lr, 'weight_decay': float(args.weight_decay)},
        ]
    else:
        param_groups = [
            {'params': list(set(model_student.parameters()).difference(set(embedding_param))) if args.gpu_id != -1 else
            list(set(model_student.module.parameters()).difference(set(embedding_param)))},
            {'params': embedding_param, 'lr': fc_layer_lr, 'weight_decay': float(args.weight_decay)},
        ]
    # Optimizer Setting
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=0.9)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    elif args.optimizer == 'adamp':
        from adamp import AdamP

        opt = AdamP(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay), nesterov=True)
    batch_num = args.nb_loops * args.iter_per_loop
    if not args.fix_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=batch_num)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=batch_num)
    all_metric = utils.evaluate_euclid_batch_4_task(model_student, dl_query, dl_pred, [1, 2, 4, 8],
                                                    res_save_dir=None)
    for task_count, task_metric in enumerate(all_metric):
        box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
        with open(os.path.join(results_dir, 'init.txt'), 'a') as f:
            f.write('{}_{} validate results\n'.format('init', task_count))
            f.write('box_recall@1  {:.4f}\n'.format(box_recalls[0] * 100))
            f.write('box_mAP的     {:.4f}\n'.format(box_mAP * 100))
            f.write('img_recall@1  {:.4f}\n'.format(img_recalls[0] * 100))
            f.write('img_mAP的      {:.4f}\n'.format(img_mAP * 100))
            f.write('unkbox_recall @1  {:.4f}\n'.format(unk_box_recall[0] * 100))
            f.write('unkbox_mAP的    {:.4f}\n'.format(unk_box_mAP * 100))
            f.write('unkimg_recall@1  {:.4f}\n'.format(unk_img_recall[0] * 100))
            f.write('unkimg_mAP的  {:.4f}\n'.format(unk_img_mAP * 100))
            f.write('allimg_mAP的  {:.4f}\n'.format(all_img_mAP * 100))
            f.write('allbox_mAP的  {:.4f}\n\n'.format(all_box_mAP * 100))
    iteration = 0
    for loop in range(args.nb_loops):
        if loop % 1 == 0:
            # dl_samplings = [dl_sampling1,dl_sampling2,dl_sampling3]
            if args.random_sample and loop != 0:
                detailed_info = []
                train_list = []
                dl_samplings = []
                for j in range(group_num):
                    dataset_sampling = GCL(
                        root=DATA_DIR,
                        mode='train',
                        transform=dataset.utils.Transform_for_Sampler(
                            is_train=False,
                            is_inception=is_inception, defined_transform=defined_transform),
                        train_list=[],
                        detail_info=[],
                        group_name='group{}'.format(j + 1),
                        IoUthr=args.IoUthr,
                        group_num=group_num,
                        # txt_id=train_count
                    )
                    detailed_info.append(dataset_sampling.detail_info)
                    train_list.append(dataset_sampling.train_list)
                    dl_sampling = torch.utils.data.DataLoader(
                        dataset_sampling,
                        batch_size=args.sz_batch,
                        shuffle=False,
                        num_workers=args.nb_workers,
                        pin_memory=False)
                    dl_samplings.append(dl_sampling)
                print('\n*********the number of dl_samplings is {}********\n'.format(len(dl_samplings)))
                train_count += 1
                trn_dataset = GCL(
                    root=DATA_DIR,
                    mode='train',
                    transform=dataset.utils.MultiTransforms(
                        is_train=True,
                        is_inception=is_inception,
                        view=args.view, defined_transform=defined_transform),
                    train_list=train_list,
                    detail_info=detailed_info,
                    group_name=None,
                    IoUthr=args.IoUthr,
                    group_num=group_num)
            balanced_sampler = sampler.NNBatchSampler(trn_dataset, model_student, dl_samplings, args.sz_batch,
                                                      args.num_neighbors, True,group_num=group_num)
            dl_tr = torch.utils.data.DataLoader(trn_dataset, num_workers=args.nb_workers, pin_memory=False,
                                                batch_sampler=balanced_sampler)

        # batch_long = len(dl_tr)
        # if not args.fix_lr:
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=batch_long)
        # else:
        #     scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=batch_long)
        print('************loop {}*********\n'.format(loop))
        model_student.train()
        model_teacher.eval()
        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model_student.model.modules() if args.gpu_id != -1 else model_student.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        pbar = enumerate(tqdm(dl_tr))
        # pbar1 = enumerate(tqdm(dl_tr))
        # pbar2 = enumerate(tqdm(dl_tr))
        # for ii in range(2):
        #     if ii == 0:
        #         pbar = pbar1
        #     else:
        #         pbar = pbar2
        for batch_idx, data in pbar:
            if batch_idx > inter_per_loop:
                break
            x, _, y, idx, groups = data
            # y = y.squeeze().cuda(non_blocking=True)
            idx = idx.squeeze()#.cuda(non_blocking=True)

            # N = len(y)
            # y = torch.cat([y] * args.view)
            idx = torch.cat([idx] * args.view).cuda(non_blocking=True)
            groups = np.concatenate([groups] * args.view)
            groups = torch.from_numpy(groups)

            x = torch.cat(x, dim=0)
            x_s, x_t = x, x
            x_s_cuda = x_s.squeeze().cuda(non_blocking=True)
            x_t_cuda = x_t.squeeze().cuda(non_blocking=True)
            with torch.no_grad():
                t_g, t_idx = model_teacher(x_t_cuda, groups)
            s_g, s_f, s_idx = model_student(x_s_cuda, groups)


            all_loss = stml_criterion(s_f, s_g, t_g, s_idx, iteration)

            loss = all_loss.pop('loss')
            RC_loss = all_loss['RC'].cpu()
            KL_groups_loss = all_loss['KL_groups'].cpu()
            KL_loss = all_loss['KL'].cpu()
            opt.zero_grad()
            loss.backward()
            if iteration % args.momentum_batch == 0:
                # print('\nupdate teacher\n')
                momentum_update(model_student, model_teacher)
            opt.step()
            scheduler.step()

            del s_g, s_f, s_idx, t_g, t_idx, x, idx, x_s, x_t, x_s_cuda, x_t_cuda
            torch.cuda.empty_cache()

            # if ((iteration % args.val_interval == 0) and iteration != 0) or iteration == 150:
            # if (iteration % args.val_interval == 0) or iteration == 150:
            if ((iteration % args.val_interval == 0) and iteration != 0):
                batch_name = str(iteration).rjust(7, '0')
                torch.save({
                    'model_state_dict': model_student.state_dict() if args.gpu_id != -1 else model_student.module.state_dict()},
                    '{}/loop{}batch_{}.pth'.format(pth_save_dir, loop, iteration))
                torch.save({
                    'model_state_dict': model_teacher.state_dict() if args.gpu_id != -1 else model_teacher.module.state_dict()},
                    '{}/teacher_loop{}batch_{}.pth'.format(pth_save_dir, loop, iteration))
                detail_dir = None#os.path.join(detailed_base_dir, batch_name)
                with torch.no_grad():
                    all_metric = utils.evaluate_euclid_batch_4_task(model_student, dl_query, dl_pred, [1, 2, 4, 8],
                                                                    res_save_dir=detail_dir)
                    for task_count, task_metric in enumerate(all_metric):
                        box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
                        wandb.log({'{}box_recall'.format(task_count): box_recalls[0] * 100,
                                   '{}box_mAP'.format(task_count): box_mAP * 100,
                                   '{}img_recall'.format(task_count): img_recalls[0] * 100,
                                   '{}img_mAP'.format(task_count): img_mAP * 100,
                                   '{}unk_box_recall'.format(task_count): unk_box_recall[0] * 100,
                                   '{}unk_box_mAP'.format(task_count): unk_box_mAP * 100,
                                   '{}unk_img_recalls'.format(task_count): unk_img_recall[0] * 100,
                                   '{}unk_img_mAP'.format(task_count): unk_img_mAP * 100,
                                   '{}all_box_mAP'.format(task_count): all_box_mAP * 100,
                                   '{}all_img_mAP'.format(task_count): all_img_mAP * 100},
                                  commit=False)
                        with open(os.path.join(results_dir, str(iteration) + '_test.txt'), 'a') as f:
                            f.write('{}_{}validate results\n'.format(iteration, task_count))
                            f.write('box_recall@1  {:.4f}\n'.format(box_recalls[0] * 100))
                            f.write('box_mAP的     {:.4f}\n'.format(box_mAP * 100))
                            f.write('img_recall@1  {:.4f}\n'.format(img_recalls[0] * 100))
                            f.write('img_mAP的      {:.4f}\n'.format(img_mAP * 100))
                            f.write('unkbox_recall @1  {:.4f}\n'.format(unk_box_recall[0] * 100))
                            f.write('unkbox_mAP的   {:.4f}\n'.format(unk_box_mAP * 100))
                            f.write('unkimg_recall@1  {:.4f}\n'.format(unk_img_recall[0] * 100))
                            f.write('unkimg_mAP的 {:.4f}\n'.format(unk_img_mAP * 100))
                            f.write('allimg_mAP的  {:.4f}\n'.format(all_img_mAP * 100))
                            f.write('allbox_mAP的  {:.4f}\n\n'.format(all_box_mAP * 100))
            iteration += 1
            if iteration % args.loss_interval == 0:
                wandb.log({'learning_rate': opt.param_groups[0]["lr"],
                           'loss': loss, 'RC_loss': RC_loss,
                           'group_loss': KL_groups_loss, 'KL_loss': KL_loss},
                          step=int((iteration + 1) / args.loss_interval))

        del dl_tr,trn_dataset,balanced_sampler
        # scheduler.step()
        batch_name = str(iteration).rjust(7, '0')
        torch.save({
            'model_state_dict': model_student.state_dict() if args.gpu_id != -1 else model_student.module.state_dict()},
            '{}/loop{}batch_{}.pth'.format(pth_save_dir, loop, iteration))
        torch.save({
            'model_state_dict': model_teacher.state_dict() if args.gpu_id != -1 else model_teacher.module.state_dict()},
            '{}/teacher_loop{}batch_{}.pth'.format(pth_save_dir, loop, iteration))
        detail_dir = None  # os.path.join(detailed_base_dir, batch_name)
        with torch.no_grad():
            all_metric = utils.evaluate_euclid_batch_4_task(model_student, dl_query, dl_pred, [1, 2, 4, 8],
                                                            res_save_dir=detail_dir)
            for task_count, task_metric in enumerate(all_metric):
                box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
                wandb.log({'{}box_recall'.format(task_count): box_recalls[0] * 100,
                           '{}box_mAP'.format(task_count): box_mAP * 100,
                           '{}img_recall'.format(task_count): img_recalls[0] * 100,
                           '{}img_mAP'.format(task_count): img_mAP * 100,
                           '{}unk_box_recall'.format(task_count): unk_box_recall[0] * 100,
                           '{}unk_box_mAP'.format(task_count): unk_box_mAP * 100,
                           '{}unk_img_recalls'.format(task_count): unk_img_recall[0] * 100,
                           '{}unk_img_mAP'.format(task_count): unk_img_mAP * 100,
                           '{}all_box_mAP'.format(task_count): all_box_mAP * 100,
                           '{}all_img_mAP'.format(task_count): all_img_mAP * 100},
                          commit=False)
                with open(os.path.join(results_dir, str(iteration) + '_test.txt'), 'a') as f:
                    f.write('{}_{}的测试情况\n'.format(iteration, task_count))
                    f.write('box_recall@1值为 {:.4f}\n'.format(box_recalls[0] * 100))
                    f.write('box_mAP的值为    {:.4f}\n'.format(box_mAP * 100))
                    f.write('img_recall@1值为 {:.4f}\n'.format(img_recalls[0] * 100))
                    f.write('img_mAP的值为     {:.4f}\n'.format(img_mAP * 100))
                    f.write('unkbox_recall @1值为 {:.4f}\n'.format(unk_box_recall[0] * 100))
                    f.write('unkbox_mAP的值为   {:.4f}\n'.format(unk_box_mAP * 100))
                    f.write('unkimg_recall@1值为 {:.4f}\n'.format(unk_img_recall[0] * 100))
                    f.write('unkimg_mAP的值为 {:.4f}\n'.format(unk_img_mAP * 100))
                    f.write('allimg_mAP的值为 {:.4f}\n'.format(all_img_mAP * 100))
                    f.write('allbox_mAP的值为 {:.4f}\n\n'.format(all_box_mAP * 100))
  
    