import argparse
import os
import pickle

import dataset
import torch,wandb
from net.inception import inception_v1
from net.googlenet import googlenet
from net.clip import clip_gcl
from net.moco import moco_gcl
from net.MiT_STML import MiT_gcl as STML_MiT
from net.MiT import MiT_gcl 
from PIL import Image
from utils import evaluate_euclid_batch_4_task
from dataset.eval import OWOD,OWOD_test
from dataset.eval_VG import eval_VG
from dataset.eval_lvis import eval_lvis
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from tqdm import tqdm
from utils import T1_id,T2_id,T3_id,T4_id,calc_distance_batch,calc_all_metric
import gc
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_VGdataset(transform,thr=0.3):
    args_sz_batch = 120
    image_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/VG/VG_100K'
    args_nb_workers = 8
    query_dataset = eval_VG(mode='query',
                            image_dir=image_dir,
                            transform=transform)
    pred_dataset = eval_VG(mode='pred',
                            image_dir=image_dir,
                            transform=transform)
    dl_pred = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    return dl_pred, dl_query

def get_validate_dataset(transform):
    args_sz_batch = 120
    DATA_DIR = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata'
    args_nb_workers = 8
    query_dataset = OWOD(
        root=DATA_DIR,
        mode='query',
        transform=transform
    )
    pred_dataset = OWOD(
        root=DATA_DIR,
        mode='eval',
        transform=transform
    )
    dl_pred = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    return dl_pred, dl_query


def validate_single(model_student,save_path,detailed_dir,defined_transform=None):
    if defined_transform == None:
        transform = dataset.utils.make_transform(
            is_train=False,
            is_inception=True)
    else:
        transform = defined_transform
    k_list = [1, 2, 4, 8]
    # dl_pred,dl_query = get_validate_dataset(transform)
    dl_pred, dl_query = get_validate_dataset(transform)

    with torch.no_grad():
        model_student.eval()
        all_metric = evaluate_euclid_batch_4_task(model_student, dl_query, dl_pred, k_list, res_save_dir=detailed_dir,vis=True)
        for task_count, task_metric in enumerate(all_metric):
            box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
            with open(save_path, 'a') as f:
                f.write('{}_task{}的测试情况\n'.format('final', task_count))
                f.write('box_recall@1值为 {:.4f}\n'.format(box_recalls[0] * 100))
                f.write('box_mAP的值为    {:.4f}\n'.format(box_mAP * 100))
                f.write('img_recall@1值为 {:.4f}\n'.format(img_recalls[0] * 100))
                f.write('img_mAP的值为     {:.4f}\n'.format(img_mAP * 100))
                f.write('unkbox_recall @1值为 {:.4f}\n'.format(unk_box_recall[0] * 100))
                f.write('unkbox_mAP的值为   {:.4f}\n'.format(unk_box_mAP * 100))
                f.write('unkimg_recall@1值为 {:.4f}\n'.format(unk_img_recall[0] * 100))
                f.write('unkimg_mAP的值为 {:.4f}\n'.format(unk_img_mAP * 100))
                f.write('all_img_mAP的值为 {:.4f}\n'.format(all_img_mAP * 100))
                f.write('all_box_mAP的值为 {:.4f}\n\n'.format(all_box_mAP * 100))

def get_lvis_dataset(transform):
    DATA_DIR = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/LVIS'
    args_sz_batch = 120
    args_nb_workers = 4
    query_dataset = eval_lvis(
        root=DATA_DIR,
        mode='query',
        transform=transform
    )
    pred_dataset = eval_lvis(
        root=DATA_DIR,
        mode='eval',
        transform=transform
    )
    dl_pred = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    return dl_pred, dl_query
    

def get_test_dataset(transform):
    args_sz_batch = 120
    DATA_DIR = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata'
    args_nb_workers = 8
    query_dataset = OWOD_test(
        root=DATA_DIR,
        mode='query',
        transform=transform,
        task='t4',
        IoUthr='0.3'
    )
    pred_dataset = OWOD_test(
        root=DATA_DIR,
        mode='eval',
        transform=transform,
        task='t4',
        IoUthr='0.3'
    )
    dl_pred = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args_sz_batch,
        shuffle=False,
        num_workers=args_nb_workers,
        pin_memory=True
    )
    return dl_pred, dl_query

def test_single(model_student,save_path,detailed_dir,defined_transform=None):
    if defined_transform == None:
        transform = dataset.utils.make_transform(
            is_train=False,
            is_inception=True)
    else:
        transform = defined_transform
    k_list = [1, 2, 4, 8]
    dl_pred, dl_query = get_test_dataset(transform)

    model_student.eval()
    all_metric = evaluate_euclid_batch_4_task(model_student, dl_query, dl_pred, k_list, res_save_dir=detailed_dir)
    for task_count, task_metric in enumerate(all_metric):
        box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
        with open(save_path, 'a') as f:
            f.write('{}_task{}的测试情况\n'.format('final', task_count))
            f.write('box_recall@1值为 {:.4f}\n'.format(box_recalls[0] * 100))
            f.write('box_mAP的值为    {:.4f}\n'.format(box_mAP * 100))
            f.write('img_recall@1值为 {:.4f}\n'.format(img_recalls[0] * 100))
            f.write('img_mAP的值为     {:.4f}\n'.format(img_mAP * 100))
            f.write('unkbox_recall @1值为 {:.4f}\n'.format(unk_box_recall[0] * 100))
            f.write('unkbox_mAP的值为   {:.4f}\n'.format(unk_box_mAP * 100))
            f.write('unkimg_recall@1值为 {:.4f}\n'.format(unk_img_recall[0] * 100))
            f.write('unkimg_mAP的值为 {:.4f}\n'.format(unk_img_mAP * 100))
            f.write('all_img_mAP的值为 {:.4f}\n'.format(all_img_mAP * 100))
            f.write('all_box_mAP的值为 {:.4f}\n\n'.format(all_box_mAP * 100))
def test_save_feat(model_student,save_dir,dl_pred,defined_transform=None):
    os.makedirs(save_dir,exist_ok=True)
    pred_calc_batch = 400
    save_count = 0

    pred_F = []
    pred_ImgFr = []
    pred_box_count = []
    pred_C = []
    model_student.eval()
    with torch.no_grad():
        for pred_batch_count, pred_batch in enumerate(tqdm(dl_pred)):
            for i, J in enumerate(pred_batch):
                if i == 0:
                    _, J = model_student(J.cuda(non_blocking=True))
                    for j in J:
                        # pred_F.append(j.cpu())
                        pred_F.append(j)
                elif i == 1:
                    for j in J:
                        pred_box_count.append(int(j.split('/')[-1].split('-')[-1].split('.')[0]))
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 1000000000 + int(j[2:])
                            #j = 0
                        j = int(j)
                        pred_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        # pred_C.append(j.cpu())
                        pred_C.append(j)
                else:
                    continue
            if (pred_batch_count + 1) % pred_calc_batch == 0:
                pred_F = torch.stack(pred_F).cpu()
                pred_C = torch.stack(pred_C).numpy()
                pred_box_count = np.array(pred_box_count, dtype=np.int32)
                pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
                save_pkl = '{}/{:03}.pkl'.format(save_dir, save_count)
                save_count += 1
                with open(save_pkl, "wb") as file:
                    pickle.dump([pred_F, pred_C, pred_box_count, pred_ImgFr], file)
                pred_F = []
                pred_ImgFr = []
                pred_box_count = []
                pred_C = []
                gc.collect()
        # calc pred distance last loop:
        pred_F = torch.stack(pred_F).cpu()
        pred_C = torch.stack(pred_C).cpu().numpy()
        pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
        pred_box_count = np.array(pred_box_count, dtype=np.int32)
        save_pkl = '{}/{:03}.pkl'.format(save_dir, save_count)
        save_count += 1
        with open(save_pkl, "wb") as file:
            pickle.dump([pred_F, pred_C, pred_box_count, pred_ImgFr], file)

def test_use_saved_feat(feat_dir,res_save_txt,model,dl_query,dl_pred,detailed_dir=None,vis=False,iou_plus=100,dataset_name='ovor'):
    k_list = [1,2,4,8]
    pred_calc_batch = 400
    query_calc_batch = 50
    return_k = 2000
    model.eval()
    if dataset_name == 'ovor':
        known_cat1 = T1_id
        known_cat2 = T1_id + T2_id
        known_cat3 = T1_id + T2_id + T3_id
        known_cat4 = T1_id + T2_id + T3_id + T4_id
    elif dataset_name == 'lvis':
        known_cat1=known_cat2=known_cat3=known_cat4=list(range(1,1500))
    elif dataset_name == 'VG':
        known_cat1=known_cat2=known_cat3=known_cat4=list(range(1,iou_plus-1000))
    pred_img_num = len(dl_pred.dataset)
    query_img_num = len(dl_query.dataset)
    print('query_box_img number是{}\n'.format(query_img_num))
    if pred_img_num % (pred_calc_batch * 120) > return_k:
        left_sample_pred_num = return_k
    else:
        left_sample_pred_num = pred_img_num % (pred_calc_batch * 120)
    sample_pred_num_total = (pred_img_num // (pred_calc_batch * 120)) * 2000 + left_sample_pred_num
    metric1 = [[] for i in range(10)]
    metric2 = [[] for i in range(10)]
    metric3 = [[] for i in range(10)]
    metric4 = [[] for i in range(10)]
    final_metric1 = []
    final_metric2 = []
    final_metric3 = []
    final_metric4 = []
    pkl_list = os.listdir(feat_dir)
    pkl_list.sort()
    list_len = len(pkl_list)

    query_F = []
    query_ImgFr = []
    query_C = []
    query_box_count = []
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for query_batch_count, batch in enumerate(tqdm(dl_query)):
            for i, J in enumerate(batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        query_F.append(j)
                elif i == 1:
                    for j in J:
                        query_box_count.append(int(j.split('/')[-1].split('-')[-1].split('.')[0]))
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 1000000000 + int(j[2:])
                        j = int(j)
                        query_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        query_C.append(j)
                else:
                    continue
            if (query_batch_count + 1) % query_calc_batch == 0:
                query_F = torch.stack(query_F)
                query_C = np.array(torch.stack(query_C).cpu(), dtype=np.int32)
                query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
                query_box_count = np.array(query_box_count, dtype=np.int32)


                all_d = torch.ones([query_calc_batch * 120, sample_pred_num_total])
                all_C = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)
                sort_C = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)
                all_ImgFr = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)
                all_box_count = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)
                sort_ImgFr = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)
                sort_box_count = np.ones([query_calc_batch * 120, sample_pred_num_total], dtype=np.int32)

                for count, pkl_name in enumerate(pkl_list):
                    with open('{}/{}'.format(feat_dir,pkl_name),"rb") as pkl_file:
                        pred_F, pred_C, pred_box_count, pred_ImgFr = pickle.load(pkl_file)
                    pred_F = pred_F.cuda()
                    batch_d, batch_C, batch_ImgFr, batch_box_count = calc_distance_batch(query_F,pred_F,pred_ImgFr,pred_C,pred_box_count,return_k)
                    if count != list_len - 1:
                        all_d[:, count * return_k: (count + 1) * return_k] = batch_d
                        all_C[:, count * return_k: (count + 1) * return_k] = batch_C
                        all_ImgFr[:, count * return_k: (count + 1) * return_k] = batch_ImgFr
                        all_box_count[:, count * return_k: (count + 1) * return_k] = batch_box_count
                    else:
                        all_d[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_d
                        all_C[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_C
                        all_ImgFr[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_ImgFr
                        all_box_count[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_box_count
                    del pred_F, pred_C, pred_ImgFr, pred_box_count
                    gc.collect()
                # all_d = all_d.cuda()
                top_index = all_d.topk(len(all_C[0, :]), largest=False)[1].cpu().numpy()
                for i in range(np.shape(sort_C)[0]):
                    # for j in range(np.shape(sort_C)[1]):
                    sort_C[i, :] = all_C[i, top_index[i, :]]
                    sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
                    sort_box_count[i, :] = all_box_count[i, top_index[i, :]]
                    if vis == True:
                        model_name = res_save_txt.split('/')[-1].split('.')[0]
                        vis_dir = '/home/msi/PycharmProjects/STML-GCL/experiments/test_ovor/top5/{}'.format(model_name)
                        os.makedirs(vis_dir, exist_ok=True)
                        with open('{}/{}-{}.txt'.format(vis_dir, query_ImgFr[i],query_box_count[i]), 'w') as file:
                            for jj in range(10):
                                file.write('{}-{}\n'.format(sort_ImgFr[i, jj], sort_box_count[i, jj]))
                temp_metric1 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat1,
                                               query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                               res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
                temp_metric2 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat2,
                                               query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                               res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
                temp_metric3 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat3,
                                               query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                               res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
                temp_metric4 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat4,
                                               query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                               res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
                del query_F, query_C, query_ImgFr, query_box_count, all_d, all_C, all_ImgFr, all_box_count, sort_ImgFr, sort_C
                gc.collect()
                for i, ele in enumerate(temp_metric1):
                    metric1[i].append(ele)
                for i, ele in enumerate(temp_metric2):
                    metric2[i].append(ele)
                for i, ele in enumerate(temp_metric3):
                    metric3[i].append(ele)
                for i, ele in enumerate(temp_metric4):
                    metric4[i].append(ele)

                query_F = []
                query_C = []
                query_ImgFr = []
                query_box_count = []

        # query last loop
        query_F = torch.stack(query_F)
        query_C = np.array(torch.stack(query_C).cpu(), dtype=np.int32)
        query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
        query_box_count = np.array(query_box_count, dtype=np.int32)
        all_d = torch.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total])
        all_C = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)
        sort_C = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)
        all_ImgFr = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)
        all_box_count = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)
        sort_ImgFr = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)
        sort_box_count = np.ones([query_img_num % (query_calc_batch * 120), sample_pred_num_total], dtype=np.int32)

        for count, pkl_name in enumerate(pkl_list):
            with open('{}/{}'.format(feat_dir, pkl_name), "rb") as pkl_file:
                pred_F, pred_C, pred_box_count, pred_ImgFr = pickle.load(pkl_file)
            pred_F = pred_F.cuda()
            batch_d, batch_C, batch_ImgFr, batch_box_count = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C,
                                                                                 pred_box_count, return_k)
            if count != list_len - 1:
                all_d[:, count * return_k: (count + 1) * return_k] = batch_d
                all_C[:, count * return_k: (count + 1) * return_k] = batch_C
                all_ImgFr[:, count * return_k: (count + 1) * return_k] = batch_ImgFr
                all_box_count[:, count * return_k: (count + 1) * return_k] = batch_box_count
            else:
                all_d[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_d
                all_C[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_C
                all_ImgFr[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_ImgFr
                all_box_count[:, (pred_img_num // (pred_calc_batch * 120)) * 2000:] = batch_box_count
            del pred_F, pred_C, pred_ImgFr, pred_box_count
            gc.collect()
        all_d = all_d.cuda()
        top_index = all_d.topk(len(all_C[0, :]), largest=False)[1].cpu().numpy()
        for i in range(np.shape(sort_C)[0]):
            # for j in range(np.shape(sort_C)[1]):
            sort_C[i, :] = all_C[i, top_index[i, :]]
            sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
            sort_box_count[i, :] = all_box_count[i, top_index[i, :]]
            if vis == True:
                        model_name = res_save_txt.split('/')[0].split('.')[0]
                        vis_dir = '/home/msi/PycharmProjects/STML-GCL/experiments/test_ovor/top5/{}'.format(model_name)
                        os.makedirs(vis_dir, exist_ok=True)
                        with open('{}/{}-{}.txt'.format(vis_dir, query_ImgFr[i],query_box_count[i]), 'w') as file:
                            for jj in range(5):
                                file.write('{}-{}\n'.format(sort_ImgFr[i, jj], sort_box_count[i, jj]))
        temp_metric1 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat1,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
        temp_metric2 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat2,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
        temp_metric3 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat3,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
        temp_metric4 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat4,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=detailed_dir, pred_box_counts=sort_box_count,iou_plus=iou_plus)
        del query_F, query_C, query_ImgFr, query_box_count, all_d, all_C, all_ImgFr, all_box_count, sort_ImgFr, sort_C
        gc.collect()
        for i, ele in enumerate(temp_metric1):
            metric1[i].append(ele)
        for i, ele in enumerate(temp_metric2):
            metric2[i].append(ele)
        for i, ele in enumerate(temp_metric3):
            metric3[i].append(ele)
        for i, ele in enumerate(temp_metric4):
            metric4[i].append(ele)
    for idx,met in enumerate(metric1):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric1.append(np.average(met,axis=1))
            else:
                final_metric1.append(np.average(met))
        else:
            final_metric1.append(np.average(met))
    for idx,met in enumerate(metric2):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric2.append(np.average(met,axis=1))
            else:
                final_metric2.append(np.average(met))
        else:
            final_metric2.append(np.average(met))
    for idx,met in enumerate(metric3):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric3.append(np.average(met,axis=1))
            else:
                final_metric3.append(np.average(met))
        else:
            final_metric3.append(np.average(met))
    for idx,met in enumerate(metric4):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric4.append(np.average(met,axis=1))
            else:
                final_metric4.append(np.average(met))
        else:
            final_metric4.append(np.average(met))
    all_metric = [final_metric1, final_metric2, final_metric3,final_metric4]
    for task_count, task_metric in enumerate(all_metric):
        box_recalls, unk_box_recall, img_recalls, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP, all_box_mAP, all_img_mAP = task_metric
        with open(res_save_txt, 'a') as f:
            f.write('{}_task{}的测试情况\n'.format('final', task_count))
            f.write('box_recall@1值为 {:.4f}\n'.format(box_recalls[0] * 100))
            f.write('box_mAP的值为    {:.4f}\n'.format(box_mAP * 100))
            f.write('img_recall@1值为 {:.4f}\n'.format(img_recalls[0] * 100))
            f.write('img_mAP的值为     {:.4f}\n'.format(img_mAP * 100))
            f.write('unkbox_recall @1值为 {:.4f}\n'.format(unk_box_recall[0] * 100))
            f.write('unkbox_mAP的值为   {:.4f}\n'.format(unk_box_mAP * 100))
            f.write('unkimg_recall@1值为 {:.4f}\n'.format(unk_img_recall[0] * 100))
            f.write('unkimg_mAP的值为 {:.4f}\n'.format(unk_img_mAP * 100))
            f.write('all_img_mAP的值为 {:.4f}\n'.format(all_img_mAP * 100))
            f.write('all_box_mAP的值为 {:.4f}\n\n'.format(all_box_mAP * 100))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--pth_dir', type=str, default='/home/msi/PycharmProjects/STML-GCL/experiments/test/pth',
                        help='pth_dir')
    parser.add_argument('--dataset_name',type=str, default='lvis',
                        help='dataset_name')
    parser.add_argument('--resume_name',type=str,default='MiT_500.pth',
                        help='resume_name')
    parser.add_argument('--save_feat',type=int,default=1,
                        help='save_feat or calclate metrics')
    parser.add_argument('--vis',type=int,default=0,
                        help='vis or not')
    parser.add_argument('--gpu_id',type=int,default=1,
                        help='gpu_id')
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu_id)
    
    pth_dir = args.pth_dir
    dataset_name = args.dataset_name
    model_name = args.resume_name
    print('***********{}*************'.format(model_name))
    vis_flag = args.vis
    if model_name[:3] == "gcl":
        model_student = googlenet(512, 1024, True, 1, True)
    elif model_name.split('_')[0] == 'clip':
        model_student = clip_gcl(512,1024,True,1,True)
    elif model_name.split('_')[0] == 'moco':
        model_student = moco_gcl(512,1024,True,1,True)
    elif model_name.split('_')[0] == 'STMLMiT':
        model_student = STML_MiT(512,1024,True,1,True)
    elif model_name.split('_')[0] == 'MiT':
        model_student = MiT_gcl(512,1024,True,1,True)
    else:
        model_student = inception_v1(512, 1024, True, 1, True)
    if model_name.split('_')[0] == 'clip' or model_name.split('_')[0] == 'moco' or model_name.split('_')[0] == 'STMLMiT' or model_name.split('_')[0] == 'MiT':
        defined_transform = model_student.preprocess
    else:
        defined_transform = dataset.utils.make_transform(
            is_train=False,
            is_inception=True)
        
        
    if dataset_name == 'ovor':
        dl_pred, dl_query = get_test_dataset(transform=defined_transform)
        iou_plus=100
    elif dataset_name == 'lvis':
        dl_pred, dl_query = get_lvis_dataset(transform=defined_transform)
        iou_plus=2000
    elif dataset_name == 'VG':
        dl_pred, dl_query = get_VGdataset(transform=defined_transform)
        iou_plus=90000
    save_dir = '/home/msi/PycharmProjects/STML-GCL/experiments/test/test_{}'.format(dataset_name)
    feat_save_base_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/test_feat/{}'.format(dataset_name)
    detailed_base_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/test_detailed/{}'.format(dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    if model_name.split('_')[-1] != 'ori':
        resume_model = os.path.join(pth_dir, model_name)
        print('load from{}'.format(resume_model))
        checkpoint = torch.load(resume_model, map_location='cpu'.format(0))
        model_student.load_state_dict(checkpoint['model_state_dict'])
    model_student = model_student.cuda()
       
    detailed_dir = '{}/{}'.format(detailed_base_dir,model_name.split('.')[0])
    feat_save_dir = '{}/{}'.format(feat_save_base_dir,model_name.split('.')[0])
    if args.save_feat:
        test_save_feat(model_student,feat_save_dir,dl_pred,defined_transform)
    else:
        save_path = '{}/{}.txt'.format(save_dir,model_name.split('.')[0])
        test_use_saved_feat(feat_save_dir,save_path,model_student,dl_query,dl_pred,vis=vis_flag,iou_plus=iou_plus,dataset_name=dataset_name,detailed_dir=detailed_dir)



