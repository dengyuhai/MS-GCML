import gc
import os

import numpy as np
import torch
import json
from tqdm import tqdm
import argparse
import pickle
import faiss
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import _LRScheduler
# from torch.optim.optimizer import Optimizer, required

T1_id=[1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
T2_id=[8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 78, 79, 80, 81, 82]
T3_id=[34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
T4_id=[46, 47, 48, 49, 50, 51, 65, 70, 73, 74, 75, 76, 77, 84, 85, 86, 87, 88, 89, 90]

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k,known_cat,inf_cat,query_ImgFR=None,query_box_count=None,res_save_dir =None):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    if len(known_cat) ==20:
        task = 1
    elif len(known_cat) ==40:
        task = 2
    elif len(known_cat) ==60:
        task = 3
    else:
        task = 4
    s = 0
    us=0
    known_cat = set(known_cat)
    if res_save_dir != None:
        recall_save_dir = os.path.join(res_save_dir, 'recall/' + str(k))
        os.makedirs(recall_save_dir, exist_ok=True)
        for t, y, cc, img_fr, box_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count):
            with open(os.path.join(recall_save_dir, str(img_fr) + '_' + str(box_count) + '.txt'), 'a') as f:
                if t in y[:k]:
                    if cc in known_cat:
                        s += 1
                    else:
                        us += 1
                    f.write('task{} cat{} recall@{} 1\n'.format(task, cc, k))
                else:
                    f.write('task{} cat{} recall@{} 0\n'.format(task, cc, k))
    else:
        for t, y, cc, img_fr, box_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count):
            if t in y[:k]:
                if cc in known_cat:
                    s += 1
                else:
                    us += 1
    return s / (1. * len(T)),us / (1. * len(T))

def calc_recall_at_k_GT(T, Y, k,known_cat,inf_cat):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    us=0
    if k == 1:
        k = 2
    for t,y,cc in zip(T,Y,inf_cat):
        if t in y[1:k]:
        # if t in y[:k]:
            if cc in known_cat:
                s += 1
            else:
                us+=1
    return s / (1. * len(T)),us / (1. * len(T))

def calc_img_recall_at_k(T, Y, k,known_cat,inf_cat,query_ImgFR=None,query_box_count=None,res_save_dir=None,iou_plus = 100):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    if len(known_cat) == 20:
        task = 1
    elif len(known_cat) == 40:
        task = 2
    elif len(known_cat) == 60:
        task = 3
    else:
        task = 4
    known_cat = set(known_cat)
    s = 0
    us=0
    if res_save_dir != None:
        recall_save_dir = os.path.join(res_save_dir, 'img_recall/' + str(k))
        os.makedirs(recall_save_dir, exist_ok=True)
        for t, y, cc, img_fr, box_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count):
            with open(os.path.join(recall_save_dir, str(img_fr) + '_' + str(box_count) + '.txt'), 'a') as f:
                if t in y[:k] or (t + iou_plus) in y[:k]:
                    f.write('task{} cat{} recall@{} 1\n'.format(task, cc, k))
                    if cc in known_cat:
                        s += 1
                    else:
                        us += 1
                else:
                    f.write('task{} cat{} recall@{} 0\n'.format(task, cc, k))
    else:
        for t, y, cc, img_fr, box_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count):
            if t in y[:k] or (t + iou_plus) in y[:k]:
                if cc in known_cat:
                    s += 1
                else:
                    us += 1
    return s / (1. * len(T)),us / (1. * len(T))

def calc_img_recall_at_k_GT(T, Y, k,known_cat,inf_cat):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    us=0
    if k == 1:
        k = 2
    for t,y,cc in zip(T,Y,inf_cat):
        if t in y[1:k] or (t+100) in y[1:k]:
            if cc in known_cat:
                s += 1
            else:
                us+=1
    return s / (1. * len(T)),us / (1. * len(T))


def get_train_img_feat(model, dataloader):
    model_is_training = model.training
    model.eval()
    feat_num = len(dataloader.sampler)
    feat=np.zeros([feat_num,512],dtype=np.float32)
    img_names = np.ones(feat_num,dtype='<U16')
    total_count1 = 0
    total_count2 = 0
    with torch.no_grad():
        for tmp_count,batch in enumerate(tqdm(dataloader)):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        feat[total_count1,:] = j.cpu().numpy()
                        total_count1 += 1
                elif i==1:
                    for j in J:
                        j = j.split('/')[-1].split('.')[0]
                        img_names[total_count2] = j
                        total_count2 +=1
                else:
                    continue

    model.train()
    model.train(model_is_training)  # revert to previous training state
    return feat, img_names
def faiss_kmeans_MM(feat,cat_num,img_names=None,cat_save_dir=None):
    os.makedirs(cat_save_dir,exist_ok=True)
    start_time=time.time()
    dim=np.shape(feat)[1]
    kmeans=faiss.Kmeans(dim,cat_num)
    kmeans.train(feat)
    index=faiss.IndexFlatL2(dim)
    index.add(kmeans.centroids)
    _,I=index.search(feat,1)
    I=np.array(I)[:,0]
    sort_img = [[] for i in range(cat_num)]
    for i, label in enumerate(I):
        sort_img[label].append(str(img_names[i])[2:-2]+ '.jpg')
    for i in range(cat_num):
        with open(os.path.join(cat_save_dir,str(i)+'.txt'), 'w+') as f:
            for img_name in sort_img[i]:
                f.write(img_name + "\n")

def predict_logo_batchwise(model,dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    A0 = []
    A1 = []
    A2 = []
    box_I_list = []
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1 img_path
                # i = 2: sz_batch * labels
                # i = 3: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    #                     J = model(J.cuda(), q_eval=True)
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        A0.append(j)
                elif i == 1:
                    continue
                elif i == 2:
                    for j in J:
                        A2.append(j)
                else:
                    continue


    model.train()
    model.train(model_is_training)  # revert to previous training state


    return torch.stack(A0), torch.stack(A2).cpu().numpy()

def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    # A = [[] for i in range(len(ds[0]))]
    A0 = []
    A1 = []
    A2 = []
    box_I_list=[]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1 img_path
                # i = 2: sz_batch * labels
                # i = 3: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
#                     J = model(J.cuda(), q_eval=True)
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J :
                        A0.append(j)
                elif i == 1:
                    for j in J:
                        box_index = j.split('/')[-1].split('-')[1].split('.')[0]
                        box_I_list.append(int(box_index))
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 0
                        j = int(j)
                        A1.append(j)
                elif i == 2:
                    for j in J :
                        A2.append(j)
                else:
                    continue
                

    model.train()
    model.train(model_is_training) # revert to previous training state
    
    # return [torch.stack(A[i]) for i in range(len(A))]
    # res= [torch.stack(A[i]).cpu().numpy() for i in [0,2,3]]

    # res = [torch.stack(A[0])]
    # res.append(torch.stack(A[2]).cpu().numpy())
    # res.append(np.array(box_I_list,dtype=np.int32))
    # res.append(np.array(A[1],dtype=np.int32))
    # return res
    return torch.stack(A0),torch.stack(A2).cpu().numpy(),np.array(box_I_list,dtype=np.int32),np.array(A1,dtype=np.int32)

def calc_distance_batch(query_F,pred_F,pred_ImgFr,pred_C,pred_box_count = [],return_k=2000):
    
    dist_emb = query_F.pow(2).sum(1) + (-2) * pred_F.mm(query_F.t())
    dist_emb = pred_F.pow(2).sum(1) + dist_emb.t()

    if len(pred_C) <= return_k:
        return_k = len(pred_C)
    d,top_index = dist_emb.topk(return_k, largest=False)
    d=d.cpu()
    top_index=top_index.cpu()
    if len(pred_box_count) == 0:
        return d,pred_C[top_index],pred_ImgFr[top_index]
    else:
        return d,pred_C[top_index],pred_ImgFr[top_index],pred_box_count[top_index]

def calc_logo_distance_batch(query_F,pred_F,pred_C,pred_idx,return_k=2000):
    
    dist_emb = query_F.pow(2).sum(1) + (-2) * pred_F.mm(query_F.t())
    dist_emb = pred_F.pow(2).sum(1) + dist_emb.t()

    d,top_index = dist_emb.topk(return_k, largest=False)
    d=d.cpu()
    top_index=top_index.cpu()
    return d,pred_C[top_index],pred_idx[top_index]


def calc_metric_batch(model, pred_dataloader,query_dataloader,k_list,known_cat,pred_calc_batch=400,query_calc_batch=50,return_k=2000):
    batch_size_ = pred_dataloader.batch_size
    device = "cuda"
    pred_img_num=len(pred_dataloader.dataset)
    query_img_num=len(query_dataloader.dataset)
    left_sample_pred_num = 0
    if pred_img_num % (pred_calc_batch * batch_size_) > return_k:
        left_sample_pred_num = return_k
    else:
        left_sample_pred_num = pred_img_num % (pred_calc_batch * batch_size_)
    sample_pred_num_total = (pred_img_num // (pred_calc_batch * batch_size_)) * 2000 + left_sample_pred_num
    metric = [[] for i in range(10)]
    final_metric = []
    model_is_training = model.training
    model.eval()

    query_F = []
    query_ImgFr = []
    query_C = []
    # batch_count=1
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for query_batch_count, batch in enumerate(tqdm(query_dataloader)):
            for i, J in enumerate(batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        query_F.append(j)
                elif i == 1:
                    for j in J:
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 0
                        j = int(j)
                        query_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        query_C.append(j)
                else:
                    continue
            if (query_batch_count + 1) % query_calc_batch == 0:
                query_F = torch.stack(query_F)
                query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
                query_ImgFr = np.array(query_ImgFr, dtype=np.int32)

                pred_F=[]
                pred_ImgFr=[]
                pred_C=[]
                all_d=torch.ones([query_calc_batch*batch_size_,sample_pred_num_total])
                all_C = np.ones([query_calc_batch*batch_size_,sample_pred_num_total],dtype=np.int32)
                sort_C=np.ones([query_calc_batch*batch_size_,sample_pred_num_total],dtype=np.int32)
                all_ImgFr = np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                sort_ImgFr=np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                for pred_batch_count,pred_batch in enumerate(pred_dataloader):
                    for i, J in enumerate(pred_batch):
                        if i == 0:
                            _, J = model(J.cuda(non_blocking=True))
                            # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                            for j in J:
                                pred_F.append(j)
                        elif i == 1:
                            for j in J:
                                j = j.split('/')[-1].split('-')[0]
                                if j[:2] == 'no':
                                    j = 0
                                j = int(j)
                                pred_ImgFr.append(j)
                        elif i == 2:
                            for j in J:
                                pred_C.append(j)
                        else:
                            continue
                    if (pred_batch_count+1) % pred_calc_batch == 0:
                        sample_id=int((pred_batch_count+1) / pred_calc_batch)
                        pred_F=torch.stack(pred_F)
                        pred_C=torch.stack(pred_C).cpu().numpy()
                        pred_ImgFr=np.array(pred_ImgFr, dtype=np.int32)
                        gc.collect()
                        batch_d,batch_C,batch_Img_Fr=calc_distance_batch(query_F,pred_F,pred_ImgFr,pred_C,return_k)
                        all_d[:,(sample_id-1) * return_k : sample_id * return_k]=batch_d
                        all_C[:,(sample_id-1) * return_k : sample_id * return_k]=batch_C
                        all_ImgFr[:,(sample_id-1) * return_k : sample_id * return_k]=batch_Img_Fr
                        pred_F = []
                        pred_ImgFr = []
                        pred_C = []
                        gc.collect()
                # calc pred distance last loop:
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
                gc.collect()
                batch_d, batch_C, batch_Img_Fr = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C, return_k)
                all_d[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_d
                all_C[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_C
                all_ImgFr[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_Img_Fr
                del pred_F,pred_C,pred_ImgFr

                top_index=all_d.topk(len(all_C[0,:]),largest=False)[1].cpu().numpy()
                for i in range(np.shape(sort_C)[0]):
                    # for j in range(np.shape(sort_C)[1]):
                    sort_C[i, :] = all_C[i, top_index[i, :]]
                    sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
                temp_metric = calc_all_metric(query_C, query_ImgFr,sort_C, sort_ImgFr, k_list, known_cat)
                del query_F, query_C, query_ImgFr,all_d,all_C,all_ImgFr,sort_ImgFr,sort_C
                gc.collect()
                query_F = []
                query_C = []
                query_ImgFr = []
                for i, ele in enumerate(temp_metric):
                    metric[i].append(ele)

        # query batch last loop
        query_F = torch.stack(query_F)
        query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
        query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
        pred_F = []
        pred_ImgFr = []
        pred_C = []
        all_d = torch.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total])
        all_C = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_C = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        all_ImgFr = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_ImgFr = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        for pred_batch_count, pred_batch in enumerate(pred_dataloader):
            for i, J in enumerate(pred_batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        pred_F.append(j)
                elif i == 1:
                    for j in J:
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 0
                        j = int(j)
                        pred_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        pred_C.append(j)
                else:
                    continue
            if (pred_batch_count + 1) % pred_calc_batch == 0:
                sample_id = int((pred_batch_count + 1) / pred_calc_batch)
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
                batch_d, batch_C, batch_Img_Fr = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C, return_k)
                all_d[:,(sample_id-1) * return_k : sample_id * return_k] = batch_d
                all_C[:,(sample_id-1) * return_k : sample_id * return_k] = batch_C
                all_ImgFr[:,(sample_id-1) * return_k : sample_id * return_k] = batch_Img_Fr
                pred_F = []
                pred_ImgFr = []
                pred_C = []
                gc.collect()
        # calc pred distance last loop:
        pred_F = torch.stack(pred_F)
        pred_C = torch.stack(pred_C).cpu().numpy()
        pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
        batch_d, batch_C, batch_Img_Fr = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C, return_k)
        all_d[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_d
        all_C[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_C
        all_ImgFr[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_Img_Fr
        del pred_F, pred_C, pred_ImgFr

        top_index = all_d.topk(len(all_C[0,:]), largest=False)[1].cpu()
        for i in range(np.shape(sort_C)[0]):
            # for j in range(np.shape(sort_C)[1]):
            sort_C[i, :] = all_C[i, top_index[i, :]]
            sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
        temp_metric = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat)
        del query_F, query_C, query_ImgFr, all_d, all_C, all_ImgFr
        gc.collect()
        for i, ele in enumerate(temp_metric):
            metric[i].append(ele)


    model.train()
    model.train(model_is_training)  # revert to previous training state

    for idx,met in enumerate(metric):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric.append(np.average(met,axis=1))
            else:
                final_metric.append(np.average(met))
        else:
            final_metric.append(np.average(met))
    print(final_metric)
    return final_metric

def calc_metric_batch_4_task(model, pred_dataloader,query_dataloader,k_list,res_save_dir=None,pred_calc_batch=400,query_calc_batch=50,return_k=2000,vis=False):
    # device = model.device
    batch_size_ = pred_dataloader.batch_size
    known_cat1 = T1_id
    known_cat2 = T1_id + T2_id
    known_cat3 = T1_id + T2_id + T3_id
    known_cat4 = T1_id + T2_id + T3_id + T4_id
    pred_img_num=len(pred_dataloader.dataset)
    query_img_num=len(query_dataloader.dataset)
    print('the number of query_box_img is {}\n'.format(query_img_num))
    if pred_img_num % (pred_calc_batch * batch_size_) > return_k:
        left_sample_pred_num = return_k
    else:
        left_sample_pred_num = pred_img_num % (pred_calc_batch * batch_size_)
    sample_pred_num_total = (pred_img_num // (pred_calc_batch * batch_size_)) * 2000 + left_sample_pred_num
    metric1 = [[] for i in range(10)]
    metric2 = [[] for i in range(10)]
    metric3 = [[] for i in range(10)]
    metric4 = [[] for i in range(10)]
    final_metric1 = []
    final_metric2 = []
    final_metric3 = []
    final_metric4 = []
    model_is_training = model.training
    model.eval()

    query_F = []
    query_ImgFr = []
    query_C = []
    query_box_count = []
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for query_batch_count, batch in enumerate(tqdm(query_dataloader)):
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
                            j = 0
                        j = int(j)
                        query_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        query_C.append(j)
                else:
                    continue
            if (query_batch_count + 1) % query_calc_batch == 0:
                query_F = torch.stack(query_F)
                query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
                query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
                query_box_count = np.array(query_box_count, dtype=np.int32)

                pred_F=[]
                pred_ImgFr=[]
                pred_box_count = []
                pred_C=[]
                all_d=torch.ones([query_calc_batch*batch_size_,sample_pred_num_total])
                all_C = np.ones([query_calc_batch*batch_size_,sample_pred_num_total],dtype=np.int32)
                sort_C=np.ones([query_calc_batch*batch_size_,sample_pred_num_total],dtype=np.int32)
                all_ImgFr = np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                all_box_count = np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                sort_ImgFr=np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                sort_box_count = np.ones([query_calc_batch*batch_size_,sample_pred_num_total], dtype=np.int32)
                for pred_batch_count,pred_batch in enumerate(pred_dataloader):
                    for i, J in enumerate(pred_batch):
                        if i == 0:
                            _, J = model(J.cuda(non_blocking=True))
                            # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                            for j in J:
                                pred_F.append(j)
                        elif i == 1:
                            for j in J:
                                pred_box_count.append(int(j.split('/')[-1].split('-')[-1].split('.')[0]))
                                j = j.split('/')[-1].split('-')[0]
                                if j[:2] == 'no':
                                    j = 0
                                j = int(j)
                                pred_ImgFr.append(j)
                        elif i == 2:
                            for j in J:
                                pred_C.append(j)
                        else:
                            continue
                    if (pred_batch_count+1) % pred_calc_batch == 0:
                        sample_id=int((pred_batch_count+1) / pred_calc_batch)
                        pred_F=torch.stack(pred_F)
                        pred_C=torch.stack(pred_C).cpu().numpy()
                        pred_box_count = np.array(pred_box_count,dtype=np.int32)
                        pred_ImgFr=np.array(pred_ImgFr, dtype=np.int32)
                        gc.collect()
                        batch_d,batch_C,batch_Img_Fr,batch_box_count=calc_distance_batch(query_F,pred_F,pred_ImgFr,pred_C,pred_box_count,return_k)
                        all_d[:,(sample_id-1) * return_k : sample_id * return_k]=batch_d
                        all_C[:,(sample_id-1) * return_k : sample_id * return_k]=batch_C
                        all_ImgFr[:,(sample_id-1) * return_k : sample_id * return_k]=batch_Img_Fr
                        all_box_count[:,(sample_id-1) * return_k : sample_id * return_k]=batch_box_count
                        pred_F = []
                        pred_ImgFr = []
                        pred_box_count = []
                        pred_C = []
                        gc.collect()
                # calc pred distance last loop:
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
                pred_box_count = np.array(pred_box_count, dtype=np.int32)
                gc.collect()
                batch_d, batch_C, batch_Img_Fr,batch_box_count = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C, pred_box_count,return_k)
                all_d[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_d
                all_C[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_C
                all_ImgFr[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_Img_Fr
                all_box_count[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000: ] = batch_box_count
                del pred_F,pred_C,pred_ImgFr,pred_box_count

                top_index=all_d.topk(len(all_C[0,:]),largest=False)[1].cpu().numpy()
                for i in range(np.shape(sort_C)[0]):
                    # for j in range(np.shape(sort_C)[1]):
                    sort_C[i, :] = all_C[i, top_index[i, :]]
                    sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
                    sort_box_count[i, :] = all_box_count[i, top_index[i, :]]
                    if vis == True:
                        os.makedirs('/home/msi/PycharmProjects/STML-GCL/vis/top100',exist_ok=True)
                        vis_save_txt = '/home/msi/PycharmProjects/STML-GCL/vis/top100/{}_{}.txt'.format(query_ImgFr[i],query_box_count[i])
                        with open(vis_save_txt,'w') as file:
                            for jj in range(100):
                                file.write('{}-{}\n'.format(sort_ImgFr[i][jj],sort_box_count[i][jj]))
                        # with open('/home/msi/PycharmProjects/STML-GCL/vis/retrieval.txt','a') as f:
                        #     f.write('{}-{} {}-{} {}-{} {}-{} {}-{} {}-{}\n'.format(query_ImgFr[i],query_box_count[i],
                        #                                                            sort_ImgFr[i][0],sort_box_count[i][0],
                        #                                                            sort_ImgFr[i][1],sort_box_count[i][1],
                        #                                                            sort_ImgFr[i][2],sort_box_count[i][2],
                        #                                                            sort_ImgFr[i][3],sort_box_count[i][3],
                        #                                                            sort_ImgFr[i][4],sort_box_count[i][4]))
                temp_metric1 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat1,
                                               query_ImgFR=query_ImgFr,query_box_count=query_box_count,
                                               res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
                temp_metric2 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat2,
                                               query_ImgFR=query_ImgFr,query_box_count=query_box_count,
                                               res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
                temp_metric3 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat3,
                                               query_ImgFR=query_ImgFr,query_box_count=query_box_count,
                                               res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
                temp_metric4 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat4,
                                               query_ImgFR=query_ImgFr,query_box_count=query_box_count,
                                               res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
                del query_F, query_C, query_ImgFr, query_box_count,all_d, all_C, all_ImgFr,all_box_count,sort_ImgFr,sort_C
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

        # query batch last loop
        query_F = torch.stack(query_F)
        query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
        query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
        query_box_count = np.array(query_box_count, dtype=np.int32)
        pred_F = []
        pred_ImgFr = []
        pred_box_count = []
        pred_C = []
        all_d = torch.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total])
        all_C = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_C = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        all_ImgFr = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        all_box_count = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_ImgFr = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_box_count = np.ones([query_img_num % (query_calc_batch*batch_size_), sample_pred_num_total], dtype=np.int32)
        for pred_batch_count, pred_batch in enumerate(pred_dataloader):
            for i, J in enumerate(pred_batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        pred_F.append(j)
                elif i == 1:
                    for j in J:
                        pred_box_count.append(int(j.split('/')[-1].split('-')[-1].split('.')[0]))
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 0
                        j = int(j)
                        pred_ImgFr.append(j)
                elif i == 2:
                    for j in J:
                        pred_C.append(j)
                else:
                    continue
            if (pred_batch_count + 1) % pred_calc_batch == 0:
                sample_id = int((pred_batch_count + 1) / pred_calc_batch)
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
                pred_box_count = np.array(pred_box_count, dtype=np.int32)
                batch_d, batch_C, batch_Img_Fr,batch_box_count = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C,pred_box_count, return_k)
                all_d[:,(sample_id-1) * return_k : sample_id * return_k] = batch_d
                all_C[:,(sample_id-1) * return_k : sample_id * return_k] = batch_C
                all_ImgFr[:,(sample_id-1) * return_k : sample_id * return_k] = batch_Img_Fr
                all_box_count[:,(sample_id-1) * return_k : sample_id * return_k] = batch_box_count
                pred_F = []
                pred_ImgFr = []
                pred_box_count = []
                pred_C = []
                gc.collect()
        # calc pred distance last loop:
        pred_F = torch.stack(pred_F)
        pred_C = torch.stack(pred_C).cpu().numpy()
        pred_ImgFr = np.array(pred_ImgFr, dtype=np.int32)
        pred_box_count = np.array(pred_box_count, dtype=np.int32)
        batch_d, batch_C, batch_Img_Fr,batch_box_count = calc_distance_batch(query_F, pred_F, pred_ImgFr, pred_C,pred_box_count, return_k)
        all_d[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_d
        all_C[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_C
        all_ImgFr[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_Img_Fr
        all_box_count[:,(pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_box_count
        del pred_F, pred_C, pred_ImgFr,pred_box_count

        top_index = all_d.topk(len(all_C[0,:]), largest=False)[1].cpu()
        for i in range(np.shape(sort_C)[0]):
            # for j in range(np.shape(sort_C)[1]):
            sort_C[i, :] = all_C[i, top_index[i, :]]
            sort_ImgFr[i, :] = all_ImgFr[i, top_index[i, :]]
            sort_box_count[i, :] = all_box_count[i, top_index[i, :]]
            if vis == True:
                os.makedirs('/home/msi/PycharmProjects/STML-GCL/vis/top100', exist_ok=True)
                vis_save_txt = '/home/msi/PycharmProjects/STML-GCL/vis/top100/{}_{}.txt'.format(query_ImgFr[i],
                                                                                                query_box_count[i])
                with open(vis_save_txt, 'w') as file:
                    for jj in range(100):
                        file.write('{}-{}\n'.format(sort_ImgFr[i][jj], sort_box_count[i][jj]))
                # with open('/home/msi/PycharmProjects/STML-GCL/vis/retrieval.txt', 'a') as f:
                #     f.write('{}-{} {}-{} {}-{} {}-{} {}-{} {}-{}\n'.format(query_ImgFr[i], query_box_count[i],
                #                                                            sort_ImgFr[i][0], sort_box_count[i][0],
                #                                                            sort_ImgFr[i][1], sort_box_count[i][1],
                #                                                            sort_ImgFr[i][2], sort_box_count[i][2],
                #                                                            sort_ImgFr[i][3], sort_box_count[i][3],
                #                                                            sort_ImgFr[i][4], sort_box_count[i][4]))

        temp_metric1 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat1,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
        temp_metric2 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat2,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
        temp_metric3 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat3,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
        temp_metric4 = calc_all_metric(query_C, query_ImgFr, sort_C, sort_ImgFr, k_list, known_cat4,
                                       query_ImgFR=query_ImgFr, query_box_count=query_box_count,
                                       res_save_dir=res_save_dir,pred_box_counts=sort_box_count)
        del query_F, query_C, query_ImgFr, all_d, all_C, all_ImgFr,all_box_count
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
    model.train()
    model.train(model_is_training)  # revert to previous training state
    return [final_metric1, final_metric2, final_metric3,final_metric4]


def query_batchwise(model, query_dataloader,pred_F,pred_C,pred_ImgFr,k_list,calc_batch=50,known_cat=None):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    metric=[[] for i in range(8)]
    final_metric=[]

    ds = query_dataloader.dataset
    query_F=[]
    query_ImgFr=[]
    query_C=[]
    # batch_count=1
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch_count,batch in enumerate(tqdm(query_dataloader)):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    #                     J = model(J.cuda(), q_eval=True)
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        query_F.append(j)
                elif i==1:
                    for j in J:
                        j = j.split('/')[-1].split('-')[0]
                        if j[:2] == 'no':
                            j = 0
                        j = int(j)
                        query_ImgFr.append(j)
                elif i==2:
                    for j in J:
                        query_C.append(j)
                else:
                    continue
            if (batch_count+1) % calc_batch == 0:
                query_F = torch.stack(query_F)
                query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
                query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
                temp_metric = calc_metric(query_F, query_C, query_ImgFr, pred_F, pred_C, pred_ImgFr, k_list, known_cat)
                del query_F, query_C, query_ImgFr
                query_F=[]
                query_C=[]
                query_ImgFr=[]
                for i, ele in enumerate(temp_metric):
                    metric[i].append(ele)

        # last loop
        query_F = torch.stack(query_F)
        query_C = np.array(torch.stack(query_C).cpu(),dtype=np.int32)
        query_ImgFr = np.array(query_ImgFr, dtype=np.int32)
        temp_metric = calc_metric(query_F, query_C, query_ImgFr, pred_F, pred_C, pred_ImgFr, k_list, known_cat)
        for i, ele in enumerate(temp_metric):
            metric[i].append(ele)
    for idx,met in enumerate(metric):
        if idx<4:
            if len(met)>1:
                met=np.stack(met,axis=1)
                final_metric.append(np.average(met,axis=1))
            else:
                final_metric.append(np.average(met))
        else:
            final_metric.append(np.average(met))
    print(final_metric)
    return final_metric

def calc_all_metric(query_C,query_ImgFr,pred_C,pred_ImgFr,k_list,known_cat,
                    query_ImgFR=None,query_box_count=None,res_save_dir = None,pred_box_counts=None,iou_plus=100):
    # calculate recall @ K
    box_recall = []
    unk_box_recall = []
    for k in k_list:
        r_at_k, unk_r_k = calc_recall_at_k(query_C, pred_C[:,:k], k, known_cat, query_C,
                                           query_ImgFR=query_ImgFR,query_box_count=query_box_count,res_save_dir=res_save_dir)
        box_recall.append(r_at_k)
        unk_box_recall.append(unk_r_k)
    # box_mAP, unk_box_mAP,all_mAP = calc_mAP(query_C, pred_C, known_cat, query_C,
    #                                         query_ImgFR=query_ImgFR,query_box_count=query_box_count,res_save_dir=res_save_dir)
    box_mAP, unk_box_mAP, all_mAP = calc_mAP(query_C, pred_C, known_cat, query_C,
                                             query_ImgFR=query_ImgFR, query_box_count=query_box_count,
                                             res_save_dir=res_save_dir,pred_img_FR=pred_ImgFr,pred_box_count=pred_box_counts)
    img_recall = []
    unk_img_recall = []
    for k in k_list:
        r_at_k, unk_r_k = calc_img_recall_at_k(query_C, pred_C[:,:k], k, known_cat, query_C,
                                               query_ImgFR=query_ImgFR,query_box_count=query_box_count,res_save_dir=res_save_dir,iou_plus=iou_plus)#calc_recall_at_k(query_ImgFr, pred_ImgFr[:,:k], k, known_cat, query_C)
        img_recall.append(r_at_k)
        unk_img_recall.append(unk_r_k)

    # img_mAP, unk_img_mAP = calc_mAP(query_ImgFr, pred_ImgFr, known_cat, query_C)
    img_mAP, unk_img_mAP,all_img_mAP = calc_img_mAP(query_ImgFr, pred_ImgFr, known_cat, query_C,
                                                    query_ImgFR=query_ImgFR,query_box_count=query_box_count,res_save_dir=res_save_dir)

    return np.array(box_recall), np.array(unk_box_recall), np.array(img_recall), np.array(
        unk_img_recall), box_mAP, unk_box_mAP, img_mAP, unk_img_mAP,all_mAP,all_img_mAP



def calc_metric(query_F, query_C, query_ImgFr,pred_F,pred_C,pred_ImgFr,k_list,known_cat):
    Y, img_Y, mAP_Y, img_mAP_Y = direct_query(query_F, query_C, query_ImgFr, pred_F, pred_C, pred_ImgFr, k_list,len(query_C))

    # calculate recall @ K
    box_recall = []
    unk_box_recall = []
    for k in k_list:
        r_at_k, unk_r_k = calc_recall_at_k(query_C, Y, k, known_cat, query_C)
        box_recall.append(r_at_k)
        unk_box_recall.append(unk_r_k)
    box_mAP, unk_box_mAP,all_mAP = calc_mAP(query_C, mAP_Y, known_cat, query_C)
    img_recall = []
    unk_img_recall = []
    for k in k_list:
        # r_at_k, unk_r_k = calc_recall_at_k(query_ImgFr, img_Y, k, known_cat, query_C)
        r_at_k, unk_r_k = calc_img_recall_at_k(query_C,Y,k,known_cat,query_C)
        img_recall.append(r_at_k)
        unk_img_recall.append(unk_r_k)

    # img_mAP, unk_img_mAP = calc_mAP(query_ImgFr, img_mAP_Y, known_cat, query_C)
    img_mAP, unk_img_mAP,all_img_mAP = calc_img_mAP(query_ImgFr, img_mAP_Y, known_cat, query_C)

    return np.array(box_recall),np.array(unk_box_recall),np.array(img_recall),np.array(unk_img_recall),box_mAP, unk_box_mAP,img_mAP, unk_img_mAP


def calc_dist_nn_matrix(model, dataloader,calc_batch=50,query_feat=None):
    i1=i2=j1=j2=0
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = []
    B=[[] for i in range(3)]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        batch_size = dataloader.batch_size
        # dist_nn=[]
        be_query_num=len(ds.I)
        query_num=np.shape(query_feat)[0]
        dist_nn=torch.zeros((query_num,be_query_num))
        for batch_count,batch in enumerate(tqdm(dataloader)):

            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * image_names
                # i = 2: sz_batch * labels
                # i = 3: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    #                     J = model(J.cuda(), q_eval=True)
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        j2+=1
                        A.append(j)
                else:
                    for j in J:
                        if i == 1:
                            j = j.split('/')[-1].split('-')[0]
                            if j[:2] == 'no':
                                j = 99999999
                                raise RuntimeError('cannot exist no')
                            j = int(j)
                        B[i-1].append(j)

            # tmp=(int(batch_count)+1) % calc_batch
            if ((int(batch_count)+1) % calc_batch) == 0:
                be_queried_f=torch.stack(A)
                query_f=[]
                dist_batch=[]
                for idx,QF in enumerate(query_feat):
                    query_f.append(QF)
                    i2+=1
                    if (idx+1)%(batch_size*calc_batch)==0:
                        query_f=torch.stack(query_f,dim=0)
                        dist_emb=query_f.pow(2).sum(1)+(-2)*be_queried_f.mm(query_f.t())
                        dist_emb=be_queried_f.pow(2).sum(1)+dist_emb.t()
                        dist_nn[i1:i2,j1:j2]=dist_emb
                        # dist_batch.append(dist_emb)
                        del query_f
                        query_f=[]
                        i1=i2

                
                query_f = torch.stack(query_f, dim=0)
                dist_emb = query_f.pow(2).sum(1) + (-2) * be_queried_f.mm(query_f.t())
                dist_emb = be_queried_f.pow(2).sum(1) + dist_emb.t()
                dist_nn[i1:i2, j1:j2] = dist_emb
                # dist_batch.append(dist_emb)
                del query_f
                i1=i2=0

                # dist_batch=torch.stack(dist_batch,dim=0)
                # dist_nn.append(dist_batch)
                del be_queried_f,A
                A=[]
                j1=j2
        be_queried_f = torch.stack(A)
        query_f = []
        for idx, QF in enumerate(query_feat):
            query_f.append(QF)
            i2+=1
            if (idx + 1) % (batch_size * calc_batch) == 0:
                query_f = torch.stack(query_f, dim=0)
                dist_emb = query_f.pow(2).sum(1) + (-2) * be_queried_f.mm(query_f.t())
                dist_emb = be_queried_f.pow(2).sum(1) + dist_emb.t()
                dist_nn[i1:i2, j1:j2] = dist_emb
                # dist_batch.append(dist_emb)
                del query_f
                i1=i2
                query_f = []

        
        query_f = torch.stack(query_f, dim=0)
        dist_emb = query_f.pow(2).sum(1) + (-2) * be_queried_f.mm(query_f.t())
        dist_emb = be_queried_f.pow(2).sum(1) + dist_emb.t()
        dist_nn[i1:i2, j1:j2] = dist_emb
        # dist_batch.append(dist_emb)
        del query_f
        # dist_batch = torch.stack(dist_batch, dim=0)
        # dist_nn.append(dist_batch)
        del be_queried_f

    model.train()
    model.train(model_is_training)  # revert to previous training state

    # return [torch.stack(A[i]) for i in range(len(A))]

    return dist_nn

def logo_evaluate(model, pred_dataloader,query_dataloader,pred_calc_batch,query_calc_batch,return_k,detailed_save_txt):
    batch_size_ = pred_dataloader.batch_size
    pred_img_num = len(pred_dataloader.dataset)
    query_img_num = len(query_dataloader.dataset)
    if pred_img_num % (pred_calc_batch * batch_size_) > return_k:
        left_sample_pred_num = return_k
    else:
        left_sample_pred_num = pred_img_num % (pred_calc_batch * batch_size_)
    sample_pred_num_total = (pred_img_num // (pred_calc_batch * batch_size_)) * 2000 + left_sample_pred_num
    model_is_training = model.training
    model.eval()

    query_F = []
    query_C = []
    all_mAP = []
    # batch_count=1
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for query_batch_count, batch in enumerate(query_dataloader):
            for i, J in enumerate(batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        query_F.append(j)
                elif i == 1:
                    continue
                elif i == 2:
                    for j in J:
                        query_C.append(j)
                else:
                    continue
            if (query_batch_count + 1) % query_calc_batch == 0:
                query_F = torch.stack(query_F)
                query_C = np.array(torch.stack(query_C).cpu(), dtype=np.int32)

                pred_F = []
                pred_C = []
                pred_idx = []
                all_d = torch.ones([query_calc_batch * batch_size_, sample_pred_num_total])
                all_C = np.ones([query_calc_batch * batch_size_, sample_pred_num_total], dtype=np.int32)
                all_idx = np.ones([query_calc_batch * batch_size_, sample_pred_num_total], dtype=np.int32)
                sort_C = np.ones([query_calc_batch * batch_size_, sample_pred_num_total], dtype=np.int32)
                sort_idx = np.ones([query_calc_batch * batch_size_, sample_pred_num_total], dtype=np.int32)
                for pred_batch_count, pred_batch in enumerate(pred_dataloader):
                    for i, J in enumerate(pred_batch):
                        if i == 0:
                            _, J = model(J.cuda(non_blocking=True))
                            # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                            for j in J:
                                pred_F.append(j)
                        elif i == 1:
                            for j in J:
                                pred_idx.append(int(j.split('/')[-1].split('_')[-1].split('.')[0]))
                        elif i == 2:
                            for j in J:
                                pred_C.append(j)
                        else:
                            continue
                    if (pred_batch_count + 1) % pred_calc_batch == 0:
                        sample_id = int((pred_batch_count + 1) / pred_calc_batch)
                        pred_F = torch.stack(pred_F)
                        pred_C = torch.stack(pred_C).cpu().numpy()
                        pred_idx = np.array(pred_idx)
                        gc.collect()
                        batch_d, batch_C,batch_idx = calc_logo_distance_batch(query_F, pred_F,  pred_C,pred_idx,
                                                                             return_k)
                        all_d[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_d
                        all_C[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_C
                        all_idx[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_idx
                        pred_F = []
                        pred_C = []
                        pred_idx = []
                        gc.collect()
                # calc pred distance last loop:
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_idx = np.array(pred_idx)
                gc.collect()
                batch_d, batch_C,batch_idx = calc_logo_distance_batch(query_F, pred_F, pred_C,pred_idx,
                                                            return_k)
                all_d[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_d
                all_C[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_C
                all_idx[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_idx
                del pred_F, pred_C

                top_index = all_d.topk(len(all_C[0, :]), largest=False)[1].cpu().numpy()
                for i in range(np.shape(sort_C)[0]):
                    # for j in range(np.shape(sort_C)[1]):
                    sort_C[i, :] = all_C[i, top_index[i, :]]
                    sort_idx[i, :] = all_idx[i, top_index[i, :]]

                # temp_mAP = calc_logo_mAP(query_C,sort_C)
                # all_mAP.append(temp_mAP)
                with open(detailed_save_txt,'a+') as f:
                    for i in range(np.shape(sort_C)[0]):
                        f.write("{}  {}_{}.jpg\n".format(query_C[i],sort_C[i][0],sort_idx[i][0]))
                del query_F, query_C, all_d, all_C
                gc.collect()
                query_F = []
                query_C = []


        # query batch last loop
        query_F = torch.stack(query_F)
        query_C = np.array(query_C)
        pred_F = []
        pred_C = []
        pred_idx = []
        temp_111 = query_img_num % (query_calc_batch * batch_size_)
        all_d = torch.ones([query_img_num % (query_calc_batch * batch_size_), sample_pred_num_total])
        all_C = np.ones([query_img_num % (query_calc_batch * batch_size_), sample_pred_num_total], dtype=np.int32)
        all_idx = np.ones([query_img_num % (query_calc_batch * batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_C = np.ones([query_img_num % (query_calc_batch * batch_size_), sample_pred_num_total], dtype=np.int32)
        sort_idx = np.ones([query_img_num % (query_calc_batch * batch_size_), sample_pred_num_total], dtype=np.int32)
        for pred_batch_count, pred_batch in enumerate(tqdm(pred_dataloader)):
            for i, J in enumerate(pred_batch):
                if i == 0:
                    _, J = model(J.cuda(non_blocking=True))
                    # J = torch.nn.functional.normalize(J, p=2, eps=1e-12, dim=1)
                    for j in J:
                        pred_F.append(j)
                elif i == 1:
                   for j in J:
                       pred_idx.append(int(j.split('/')[-1].split('_')[-1].split('.')[0]))
                elif i == 2:
                    for j in J:
                        pred_C.append(j)
                else:
                    continue
            if (pred_batch_count + 1) % pred_calc_batch == 0:
                sample_id = int((pred_batch_count + 1) / pred_calc_batch)
                pred_F = torch.stack(pred_F)
                pred_C = torch.stack(pred_C).cpu().numpy()
                pred_idx = np.array(pred_idx)
                batch_d, batch_C,batch_idx = calc_logo_distance_batch(query_F, pred_F, pred_C,pred_idx, return_k)
                all_d[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_d
                all_C[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_C
                all_idx[:, (sample_id - 1) * return_k: sample_id * return_k] = batch_idx
                pred_F = []
                pred_C = []
                pred_idx = []
                gc.collect()
        # calc pred distance last loop:
        pred_F = torch.stack(pred_F)
        pred_C = torch.stack(pred_C).cpu().numpy()
        pred_idx = np.array(pred_idx)
        batch_d, batch_C,batch_idx = calc_logo_distance_batch(query_F, pred_F, pred_C, pred_idx,return_k)
        all_d[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_d
        all_C[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_C
        all_idx[:, (pred_img_num // (pred_calc_batch * batch_size_)) * 2000:] = batch_idx
        del pred_F, pred_C

        top_index = all_d.topk(len(all_C[0, :]), largest=False)[1].cpu()
        for i in range(np.shape(sort_C)[0]):
            # for j in range(np.shape(sort_C)[1]):
            sort_C[i, :] = all_C[i, top_index[i, :]]
            sort_idx[i, :] = all_idx[i, top_index[i, :]]


        with open(detailed_save_txt,'a+') as f:
            for i in range(np.shape(sort_C)[0]):
                f.write("{}  {}_{}.jpg\n".format(query_C[i],sort_C[i][0],sort_idx[i][0]))
        temp_mAP = calc_logo_mAP(query_C,sort_C,detailed_save_txt)
        print(temp_mAP)
        del query_F, query_C, all_d, all_C,
        gc.collect()
        all_mAP.append(temp_mAP)

    model.train()
    model.train(model_is_training)  # revert to previous training state


    return np.average(all_mAP)

def calc_logo_mAP(T,Y,detailed_txt=None):
    gt_dict = dict()
    gt_txt = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/object_detection_data/BelgaLogos/dataset/set1_gt.txt'
    with open(gt_txt) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        splits = line.split()
        logo_name = splits[0][:-1]
        img_name = splits[2].split('.')[0]
        gt = int(splits[3])
        if not int(img_name) in gt_dict:
            temp_dict = {}
            temp_dict[logo_name] = gt
            gt_dict[int(img_name)] = temp_dict
        else:
            gt_dict[int(img_name)][logo_name] = gt

    mAP = []
    logo_name = []
    # start_time = time.time()
    for t, yy in zip(T, Y):
        temp = t[:-1]
        logo_name.append(t)
        for i in range(len(yy)):
            yy[i] = gt_dict[yy[i]][t[:-1]]
        index = list(np.where(yy == 1)[0])
        # index += list(np.where(yy == (t + 100))[0])
        # index.sort()
        if len(index) == 0:
            AP = 0
        else:
            # index = np.array(index)+1
            # accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
            # AP = accumulate / index
            AP = 1 / (index[0] + 1)
        mAP.append(np.average(AP))
    print(mAP)
    print(logo_name)
    if detailed_txt != None:
        with open(detailed_txt, 'a') as f:
            for logo, AP in zip(logo_name, mAP):
                f.write(f'{logo}  {AP}\n')
            f.write(f'average: {np.average(mAP)}')
    mAP = np.average(mAP)
    return mAP

def ori_evaluate_euclid(model, dataloader, k_list, num_buffer=5000):
    X, T, _, img_name = predict_batchwise(model, dataloader)
    img_name = torch.tensor(img_name, device='cpu')
    # img_name=np.array(img_name)

    # get predictions by assigning nearest K neighbors with Euclidean distance
    mAP_K = len(T)
    K = max(k_list)
    img_Y = []  # 
    img_mAP_Y = []
    Y = []
    mAP_Y = []
    xs = []
    for x in X:
        if len(xs) < num_buffer:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs, dim=0)

            dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
            dist_emb = X.pow(2).sum(1) + dist_emb.t()

            # top_index = dist_emb.topk(1+K, largest=False)[1]
            top_index = dist_emb.topk(mAP_K, largest=False)[1]  # .cpu()
            y = T[top_index[:, 1:1 + K]]
            mAP_y = T[top_index[:, 1:]]
            top_index_cpu = top_index.cpu()
            img_y = img_name[top_index_cpu[:, 1:1 + K]]
            img_mAP_y = img_name[top_index_cpu[:, 1:]]

            Y.append(y.float().cpu())
            mAP_Y.append(mAP_y.float().cpu())
            img_Y.append(img_y)
            img_mAP_Y.append(img_mAP_y)
            xs = []

    # Last Loop
    xs = torch.stack(xs, dim=0)
    dist_emb = xs.pow(2).sum(1) + (-2) * X.mm(xs.t())
    dist_emb = X.pow(2).sum(1) + dist_emb.t()

    # top_index=dist_emb.topk(mAP_K, largest=False)[1]
    # top_index = dist_emb.topk(1+K, largest=False)[1]
    top_index = dist_emb.topk(mAP_K, largest=False)[1]  # .cpu()
    y = T[top_index[:, 1:1 + K]]
    mAP_y = T[top_index[:, 1:]]
    top_index_cpu = top_index.cpu()
    img_y = img_name[top_index_cpu[:, 1:1 + K]]
    img_mAP_y = img_name[top_index_cpu[:, 1:]]

    Y.append(y.float().cpu())
    mAP_Y.append(mAP_y.float().cpu())
    img_Y.append(img_y)
    img_mAP_Y.append(img_mAP_y)

    Y = torch.cat(Y, dim=0)
    mAP_Y = torch.cat(mAP_Y, dim=0)
    img_mAP_Y = np.concatenate(img_mAP_Y, axis=0)  # torch.cat(img_mAP_Y, dim=0)
    img_Y = np.concatenate(img_Y, axis=0)  # torch.cat(img_Y, dim=0)

    # calculate recall @ K
    box_recall = []
    for k in k_list:
        r_at_k,unk_r_at_k = calc_recall_at_k(T, Y, k)
        box_recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    box_mAP = calc_mAP(T, mAP_Y)
    img_recall = []
    for k in k_list:
        r_at_k = calc_recall_at_k(img_name, img_Y, k)
        img_recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    img_mAP = calc_mAP(img_name, img_mAP_Y)
    return box_recall, box_mAP, img_recall, img_mAP

def direct_query(query_F, query_C, query_ImgFr,pred_F,pred_C,pred_ImgFr,k_list,num_buffer=5000):
    i1=i2=0
    mAP_K=len(pred_C) 
    K=max(k_list)
    img_query_Y=np.zeros([len(query_C),mAP_K],dtype=np.int32)
    query_Y=np.zeros([len(query_C),mAP_K],dtype=np.int32)
    calc_col=[]

    for x in query_F:
        if len(calc_col)<num_buffer:
            calc_col.append(x)
            i2+=1
        else:
            calc_col.append(x)
            i2+=1

            calc_col = torch.stack(calc_col, dim=0)
            dist_emb = calc_col.pow(2).sum(1) + (-2) * pred_F.mm(calc_col.t())
            dist_emb = pred_F.pow(2).sum(1) + dist_emb.t()

            top_index=dist_emb.topk(mAP_K,largest=False)[1].cpu()
            # y = pred_C[top_index[:, : K]]
            query_Y[i1:i2,:]= pred_C[top_index]
            # img_y = pred_ImgFr[top_index_cpu[:, 1:1 + K]]
            img_query_Y[i1:i2,:] = pred_ImgFr[top_index]
            del calc_col
            calc_col=[]
            i1=i2
    calc_col = torch.stack(calc_col, dim=0)
    dist_emb = calc_col.pow(2).sum(1) + (-2) * pred_F.mm(calc_col.t())
    dist_emb = pred_F.pow(2).sum(1) + dist_emb.t()

    top_index = dist_emb.topk(mAP_K, largest=False)[1].cpu()
    # y = pred_C[top_index[:, : K]]
    query_Y[i1:i2, :] = pred_C[top_index]
    # img_y = pred_ImgFr[top_index_cpu[:, 1:1 + K]]
    img_query_Y[i1:i2, :] = pred_ImgFr[top_index]

    return query_Y[:, :K], img_query_Y[:, :K], query_Y, img_query_Y

def evaluate_euclid_batch(model, query_dataloader, pred_dataloader,k_list,task='t1',res_save_dir='/home/msi/PycharmProjects/STML-CVPR22-main/TEMP'):
    if task =='t1':
        known_cat = T1_id
    elif task =='t2':
        known_cat = T1_id + T2_id
    elif task == 't3':
        known_cat = T1_id + T2_id + T3_id
    elif task == 't4':
        known_cat = T1_id + T2_id + T3_id + T4_id

    box_recall, unk_box_recall, img_recall, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP,all_box_mAP,all_img_mAP \
        = calc_metric_batch_4_task(model,pred_dataloader,query_dataloader,k_list,res_save_dir=res_save_dir)
    return box_recall, unk_box_recall, img_recall, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP,all_box_mAP,all_img_mAP

def evaluate_euclid_batch_4_task(model, query_dataloader, pred_dataloader,k_list,res_save_dir=None,vis=False):
    # box_recall, unk_box_recall, img_recall, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP,all_box_mAP,all_img_mAP = calc_metric_batch_4_task(model,pred_dataloader,query_dataloader,
    #                                                                                                                        k_list)
    return calc_metric_batch_4_task(model,pred_dataloader,query_dataloader,k_list,res_save_dir=res_save_dir,vis=vis)

def evaluate_euclid(model, query_dataloader, pred_dataloader,k_list, task='t1'):
    if task=='t1':
        known_cat = T1_id
    elif task=='t2':
        known_cat = T1_id+T2_id
    elif task=='t3':
        known_cat = T1_id+T2_id+T3_id
    elif task=='t4':
        known_cat = T1_id+T2_id+T3_id+T4_id
    pred_F,pred_C,_,pred_ImgFr=predict_batchwise(model,pred_dataloader)
    # query_F, query_C, _, query_ImgFr = predict_batchwise(model, query_dataloader)

    start_time=time.time()
    # q_num,dim=np.shape(query_F)
    # query_F[:,0]+=np.arange(q_num)/1000.
    # base_F=pred_F
    # b_num=np.shape(pred_F)[0]
    # base_F[:, 0] += np.arange(b_num) / 1000.
    # index=faiss.IndexFlatL2(dim)
    # index.add(base_F)
    # _,top_index=index.search(query_F,q_num)
    #
    #
    # K = max(k_list)
    #
    # Y = pred_C[top_index[:, :K]]
    # mAP_Y = pred_C[top_index]
    # img_Y = pred_ImgFr[top_index[:, :K]]
    # img_mAP_Y = pred_ImgFr[top_index]
    # Y,img_Y,mAP_Y,img_mAP_Y=direct_query(query_F,query_C,query_ImgFr,pred_F,pred_C,pred_ImgFr,k_list)
    box_recall, unk_box_recall, img_recall, unk_img_recall, box_mAP, unk_box_mAP, img_mAP, unk_img_mAP=query_batchwise(model, query_dataloader,
                                                                                                                       pred_F,pred_C,pred_ImgFr,
                                                                                                                       k_list,50,known_cat)
   
    return box_recall,box_mAP,img_recall,img_mAP,unk_box_recall,unk_box_mAP,unk_img_recall,unk_img_mAP

def evaluate_euclid_GT_as_pred(model, query_dataloader, pred_dataloader,k_list, task='t1'):
    if task=='t1':
        known_cat = T1_id
    elif task=='t2':
        known_cat = T1_id+T2_id
    elif task=='t3':
        known_cat = T1_id+T2_id+T3_id
    elif task=='t4':
        known_cat = T1_id+T2_id+T3_id+T4_id
    # pred_F,pred_C,_,pred_ImgFr=predict_batchwise(model,pred_dataloader)
    start_time = time.time()
    query_F, query_C, _, query_ImgFr = predict_batchwise(model, query_dataloader)

    Y,img_Y,mAP_Y,img_mAP_Y=direct_query(query_F,query_C,query_ImgFr,query_F,query_C,query_ImgFr,k_list)
    box_recall = []
    unk_box_recall=[]
    for k in k_list:
        r_at_k,unk_r_k = calc_recall_at_k_GT(query_C, Y, k,known_cat,query_C)
        box_recall.append(r_at_k)
        unk_box_recall.append(unk_r_k)
    box_mAP,unk_box_mAP=calc_mAP_GT(query_C,mAP_Y,known_cat,query_C)
    img_recall=[]
    unk_img_recall=[]
    for k in k_list:
        r_at_k ,unk_r_k= calc_img_recall_at_k_GT(query_C, Y, k,known_cat,query_C)
        img_recall.append(r_at_k)
        unk_img_recall.append(unk_r_k)
    img_mAP,unk_img_mAP=calc_img_mAP_GT(query_ImgFr,img_mAP_Y,known_cat,query_C)
    return box_recall, box_mAP, img_recall, img_mAP, unk_box_recall, unk_box_mAP, unk_img_recall, unk_img_mAP

def calc_mAP(T,Y,known_cat,inf_cat,query_ImgFR=None,query_box_count=None,res_save_dir=None,pred_img_FR=None,pred_box_count=None,vis=False):
    """
        T : [nb_samples] (target labels)
        Y : [nb_samples x test_samples_num] 
    """

    if len(known_cat) == 20:
        task = 1
    elif len(known_cat) == 40:
        task = 2
    elif len(known_cat) == 60:
        task = 3
    else:
        task = 4
    known_cat = set(known_cat)
    mAP=[]
    unk_mAP=[]
    all_mAP = []
    if vis == False:
        if res_save_dir != None:
            AP_save_dir = os.path.join(res_save_dir, 'AP')
            os.makedirs(AP_save_dir, exist_ok=True)
        for t, yy, cc, img_fr, box_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count):
            index = np.where(yy == t)
            if len(index[0]) == 0:
                AP = 0
            else:
                index = np.array(index)
                index = index[0] + 1  #
                accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
                AP = accumulate / index
            if res_save_dir != None:
                with open(os.path.join(AP_save_dir, str(img_fr) + '_' + str(box_count) + '.txt'), 'a') as f:
                    f.write('task{} cat{} AP {:.4f}\n'.format(task, cc, np.average(AP)))
            all_mAP.append(np.average(AP))
            if cc in known_cat:
                mAP.append(np.average(AP))
            else:
                unk_mAP.append(np.average(AP))
    else:
        for t, yy, cc, img_fr, box_count,pred_FR,pred_count in zip(T, Y, inf_cat, query_ImgFR, query_box_count,pred_img_FR,pred_box_count):
            index = np.where(yy == t)
            if len(index[0]) == 0:
                AP = 0
            else:
                index_ = index[0]
                ret_pred_box_count = pred_count[index_]
                ret_pred_imgFR  = pred_FR[index_]
                count = 0
                os.makedirs('/home/msi/PycharmProjects/STML-GCL/vis/ori-clip-mAP-rank',exist_ok=True)
                with open('/home/msi/PycharmProjects/STML-GCL/vis/ori-clip-mAP-rank/{}_{}.txt'.format(img_fr,box_count),'w') as f:
                    for box,FR,rank in zip(ret_pred_box_count,ret_pred_imgFR,index_):
                        f.write('{}-{}-{}-{}\n'.format(box,FR,rank,count))
                        count+=1

                index = np.array(index)
                index = index[0] + 1  #
                accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
                AP = accumulate / index
            all_mAP.append(np.average(AP))
            if cc in known_cat:
                mAP.append(np.average(AP))
            else:
                unk_mAP.append(np.average(AP))
    mAP=np.array(mAP)
    return np.average(mAP),np.average(unk_mAP),np.average(all_mAP)

def calc_mAP_GT(T,Y,known_cat,inf_cat):
    """
        T : [nb_samples] (target labels)
        Y : [nb_samples x test_samples_num] 
    """
    mAP=[]
    unk_mAP=[]
    # start_time = time.time()
    for t,yy,cc in zip(T,Y,inf_cat):
        index=np.where(yy==t)
        if len(index[0])==0:
            AP=0
        else:
            index = index[0][1:]
            if len(index) == 0:
                AP = 0
            else:
                index = index + 1  #
                accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
                AP = accumulate / index

        if cc in known_cat:
            mAP.append(np.average(AP))
        else:
            unk_mAP.append(np.average(AP))
    mAP=np.array(mAP)
    return np.average(mAP),np.average(unk_mAP)

def calc_img_mAP(T,Y,known_cat,query_cat,query_ImgFR=None,query_box_count=None,res_save_dir=None):
    mAP = []
    unk_mAP = []
    all_mAP = []

    if res_save_dir != None:
        AP_save_dir = os.path.join(res_save_dir, 'img_AP')
        os.makedirs(AP_save_dir, exist_ok=True)
    if len(known_cat) == 20:
        task = 1
    elif len(known_cat) == 40:
        task = 2
    elif len(known_cat) == 60:
        task = 3
    else:
        task = 4
    known_cat = set(known_cat)
    # start_time = time.time()
    for t, yy, cc, img_fr, box_count in zip(T, Y, query_cat, query_ImgFR, query_box_count):
        index = list(np.where(yy == t)[0])
        # index += list(np.where(yy == (t + 100))[0])
        # index.sort()
        if len(index) == 0:
            AP = 0
        else:
            # index = np.array(index)+1
            # accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
            # AP = accumulate / index
            AP = 1/(index[0]+1)
        if res_save_dir != None:
            with open(os.path.join(AP_save_dir, str(img_fr) + '_' + str(box_count) + '.txt'), 'a') as f:
                f.write('task{} cat{} AP {:.4f}\n'.format(task, cc, np.average(AP)))
        all_mAP.append(np.average(AP))
        if cc in known_cat:
            mAP.append(np.average(AP))
        else:
            unk_mAP.append(np.average(AP))
    mAP = np.array(mAP)
    return np.average(mAP), np.average(unk_mAP),np.average(all_mAP)

def calc_img_mAP_GT(T,Y,known_cat,query_cat):
    mAP = []
    unk_mAP = []
    # start_time = time.time()
    for t, yy, cc in zip(T, Y, query_cat):
        index = list(np.where(yy == t)[0])
        # index += list(np.where(yy == (t + 100))[0])
        # index.sort()
        if len(index) == 0 or len(index) == 1:
            AP = 0
        else:
            # index = np.array(index)+1
            # accumulate = np.arange(len(index)) + 1  # Add 1 to make the number start at 1
            # AP = accumulate / index
            AP = 1/(index[1]+1)
        if cc in known_cat:
            mAP.append(np.average(AP))
        else:
            unk_mAP.append(np.average(AP))
    mAP = np.array(mAP)
    return np.average(mAP), np.average(unk_mAP)


def save_recall(T, Y, k,known_cat,inf_cat,save_path):
    s = 0
    us = 0
    for t, y, cc in zip(T, Y, inf_cat):
        if t in y[:k]:
            if cc in known_cat:
                s += 1
            else:
                us += 1
                with open(save_path,'a+') as f:
                    f.write(f"{t}\n")
    return s / (1. * len(T)), us / (1. * len(T))



