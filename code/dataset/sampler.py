import gc
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
from tqdm import *
from scipy import linalg
import faiss
import time
import pickle

class ClassBalancedBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, data_source, batch_size, images_per_class=3, ignore_index=None):
        self.targets = data_source.ys
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.ignore_index = ignore_index
        self.reverse_index, self.ignored = self._build_reverse_index()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target == self.ignore_index:
                ignored.append(i)
                continue
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self):
        # Real batch size is self.images_per_class * (self.batch_size // self.images_per_class)
        num_classes = self.batch_size // self.images_per_class
        sampled_classes = np.random.choice(list(self.reverse_index.keys()), num_classes, replace=False)

        sampled_indices = []
        for cls in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls],
                                                    self.images_per_class,
                                                    replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.batch_size
    
class NNBatchSampler(Sampler):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, data_source, model, seen_dataloaders, batch_size, nn_per_image = 5, using_feat = True, is_norm = False,save_feat=None,group_num = 4,debug_logger=None):
        self.batch_size = batch_size
        self.nn_per_image = nn_per_image
        self.using_feat = using_feat
        self.is_norm = is_norm
        self.num_samples = data_source.__len__()
        self.group_num = group_num
        # self.nn_matrix, self.dist_matrix = self._build_nn_matrix(model, seen_dataloader,save_feat=save_feat)

        self.group_len = np.array(data_source.group_len)
        mean_area = np.array(data_source.mean_area)
        self.area_ratio = mean_area / np.sum(mean_area)

        sample_ratio = self.group_len * self.area_ratio
        total = np.sum(sample_ratio)
        num_image = self.batch_size // self.nn_per_image
        num_image_list = []
        for i in range(len(sample_ratio)):
            sample_ratio_ = max(0.1, sample_ratio[i] / total)
            sample_num = int(sample_ratio_ * num_image)
            num_image_list.append(sample_num)
        max_id = np.argmax(num_image_list)
        num_image_list[max_id] = num_image - int(np.sum(num_image_list)) + num_image_list[max_id]
        self.num_image_list = num_image_list
        self.nn_matrixs = []
        self.dist_matrixs = []
        self.debug_logger = debug_logger
        # for j in range(self.group_num):
        #     self.nn_matrix, self.dist_matrix = self.build_NN_mati_using_faiss(model, seen_dataloaders[j],
        #                                                                         len(seen_dataloaders[j].dataset),
        #                                                                         save_feat=save_feat)
        #     self.nn_matrixs.append(self.nn_matrix)
        #     self.dist_matrixs.append(self.dist_matrix)
        self.nn_matrix1, self.dist_matrix1 =self.build_NN_mati_using_faiss(model, seen_dataloaders[0],len(seen_dataloaders[0].dataset),save_feat=save_feat)
        self.nn_matrix2, self.dist_matrix2 = self.build_NN_mati_using_faiss(model, seen_dataloaders[1],
                                                                            len(seen_dataloaders[1].dataset),
                                                                            save_feat=save_feat)
        self.nn_matrix3, self.dist_matrix3 = self.build_NN_mati_using_faiss(model, seen_dataloaders[2],
                                                                            len(seen_dataloaders[2].dataset),
                                                                            save_feat=save_feat)
        self.nn_matrix4, self.dist_matrix4 = self.build_NN_mati_using_faiss(model, seen_dataloaders[3],
                                                                            len(seen_dataloaders[3].dataset),
                                                                            save_feat=save_feat)
        torch.cuda.empty_cache()


    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()


    def _predict_batchwise(self, model, seen_dataloader,save_feat=None):
        model_is_training = model.training
        model.eval()

        ds = seen_dataloader.dataset
        # A = [[] for i in range(len(ds[0]))]
        A0=[]
        batch_count=0
        with torch.no_grad():
            # extract batches (A becomes list of samples)
            for batch in tqdm(seen_dataloader):
                # if batch_count>800:
                #     break
                batch_count+=1
                for i, J in enumerate(batch):
                    # i = 0: sz_batch * images
                    # i = 1: sz_batch * labels
                    # i = 2: sz_batch * indices
                    # i = 3 : sz_batch * img_path
                    # i = 4 : sz_batch * group
                    if i == 0:
                        # move images to device of model (approximate device)
                        if save_feat==True:
                            feat=model(J.cuda(),save_feat=save_feat)
                            img_names=[names.split('/')[-1].split('.')[0] for names in batch[1]]
                            save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/coco_feat'
                            os.makedirs(save_dir,exist_ok=True)
                            save_pkl=os.path.join(save_dir,str(batch_count)+'.pkl')
                            with open(save_pkl, "wb") as f:
                                pickle.dump([img_names, feat], f)
                        if self.using_feat:
                            f_J, _ = model(J.cuda())
                        else:
                            _, f_J = model(J.cuda())
                        f_J = f_J.cpu()

                        for j in f_J:
                            A0.append(j)
                    else:
                        continue
        model.train()
        model.train(model_is_training) # revert to previous training state

        # all_A0_ = torch.cat(all_A0,dim=0)
        # return all_A0_#, torch.stack(A2),torch.stack(A3) #[torch.stack(A[i]) for i in [0,2,3]]
        return torch.stack(A0)

    def build_NN_mati_using_faiss(self,model, seen_dataloader,img_num,save_feat=None):
        X = self._predict_batchwise(model, seen_dataloader, save_feat=save_feat)
        print('\n********teacher embedding计算完毕*********\n')
        X = X.numpy()
        X = np.array(X,dtype=np.float16)

        if self.using_feat:
            dim = 1024
        else:
            dim = 512
        M = 32
        ef_Construction = 60
        ef_search = 128

        # M = 16
        # ef_Construction = 40
        # ef_search = 64

        faiss_index = faiss.IndexHNSWFlat(dim, M)

        faiss_index.hnsw.efConstruction = ef_Construction
        faiss_index.hnsw.efSearch = ef_search

        # faiss_index = faiss.IndexFlatL2(1024)
        start_time =time.time()
        faiss_index.add(X)
        print('construction time  {}\n'.format(time.time()-start_time))
        K = self.nn_per_image * 1
        nn_matrix = np.ones([img_num,self.nn_per_image],dtype=np.int32)
        dist_matrix = np.ones([img_num,self.nn_per_image])
        i1 = 0
        i2 = 0
        xs = []

        for x in tqdm(X):
            if len(xs) < 5000:
                xs.append(x)
                i2+=1
            else:
                xs.append(x)
                i2+=1
                xs = np.stack(xs, axis=0)
                dist, ind = faiss_index.search(xs, K)
                nn_matrix[i1:i2,:] = ind
                dist_matrix[i1:i2,:] = dist
                i1 = i2
                xs = []
                del ind

        # Last Loop
        xs = np.stack(xs, axis=0)
        dist, ind = faiss_index.search(xs, K)
        nn_matrix[i1:i2, :] = ind
        dist_matrix[i1:i2, :] = dist
        #
        nn_matrix = torch.from_numpy(nn_matrix)
        dist_matrix = torch.from_numpy(dist_matrix)
        del X
        gc.collect()

        return nn_matrix, dist_matrix


    def sample_batch(self):
        # sampled_queries = np.random.choice(self.num_samples, num_image, replace=False)
        # sampled_indices = self.nn_matrix[sampled_queries].view(-1)

        sampled_queries = []
        sampled_indices = []
        for i in range(len(self.group_len)):
            if i == 0:
                # group_len = self.group_len[0]  #一次只采样24张
                choice_range = range(self.group_len[0])
            else:
                choice_range = range(self.group_len[i]-self.group_len[i-1])
            sample = np.random.choice(choice_range, self.num_image_list[i], replace=False)
            sampled_queries.append(sample)

        # *******************一次只采样24张**************************
        #     sample_num = int(group_len/self.group_len[-1] * num_image)
        #     sample_num_sum += sample_num
        #     sample = np.random.choice(choice_range,sample_num,replace=False)
        #     sampled_queries.append(sample)
        # if sample_num_sum < num_image:
        #     curr_queries = np.concatenate(sampled_queries)
        #     left_range = np.setdiff1d(np.arange(self.num_samples),curr_queries)
        #     sampled_queries.append(np.random.choice(left_range,num_image-sample_num_sum,replace=False))
        # *******************一次只采样24张**************************

        # sampled_queries = np.concatenate(sampled_queries)
        # sampled_indices = self.nn_matrix[sampled_queries].view(-1)
        # for j in range(self.group_num):
        #     if j == 0:
        #         sampled_indices.append(self.nn_matrixs[0][sampled_queries[0]].view(-1))
        #     else:
        #         temp = self.nn_matrixs[j][sampled_queries[j]] + self.group_len[j - 1]
        #         sampled_indices.append(temp.view(-1))

        sampled_indices.append(self.nn_matrix1[sampled_queries[0]].view(-1))
        temp = self.nn_matrix2[sampled_queries[1]]+self.group_len[0]
        sampled_indices.append(temp.view(-1))
        temp = self.nn_matrix3[sampled_queries[2]]+self.group_len[1]
        sampled_indices.append(temp.view(-1))
        temp = self.nn_matrix4[sampled_queries[3]] + self.group_len[2]
        sampled_indices.append(temp.view(-1))
        # sampled_indices = torch.stack(sampled_indices).view(-1)
        sampled_indices = torch.cat(sampled_indices)

        return sampled_indices

    def __len__(self):
        return self.num_samples // self.batch_size


