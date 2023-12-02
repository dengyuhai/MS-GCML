import random

import numpy as np

from .base import *
import os


class GCL(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None, k_fold_eval=False, fold_idx=0, train_list=[], detail_info=[],
                 task=None, group_name=None, IoUthr=None, cat_names=[], start_number=[],group_num = 4,txt_id=-1):

        self.root = root + '/OWOD/sam/'
        # train_img_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/img'
        train_img_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/sort_img'

        if task == None:
            txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/size_group_{}'.format(group_num)
            # txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/debug/OWOD/sam/train/all/size_group_{}'.format(group_num)
        else:
            txt_dir = '/mnt/temp'
            # txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/{}/size_group_{}'.format(task,group_num)
            # txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/debug/OWOD/sam/train/{}/size_group_{}'.format(task,group_num)
        train_txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/train_gcl_txt2'
        
        # print(txt_dir)

        self.mode = mode
        self.transform = transform
        self.k = 4096
        self.ys, self.im_paths, self.I, self.group = [], [], [], []
        self.group_len = []
        self.mean_area = []
        self.train_list = []
        self.detail_info = {}

        # BaseDataset.__init__(self, self.root, self.mode, self.transform, k_fold_eval, fold_idx)
        img_count = 0
        no_img_list = []
        img_num = 0
        if self.mode == 'train':
            if group_name == None:
                if len(train_list) != 0:
                    for group_id,img_paths in enumerate(train_list):
                        for img_path in img_paths:
                            self.im_paths.append(img_path)
                            self.ys += [600]
                            self.group.append(group_id + 1)
                            self.I.append(img_num)
                            img_num += 1
                        self.group_len.append(detail_info[group_id]['group_len'])
                        self.mean_area.append(detail_info[group_id]['mean_area'])
                    self.group_len = [sum(self.group_len[:i+1]) for i in range(len(self.group_len))]
                # else:
                #     for i in range(group_num):
                #         area_list = []
                #         with open(os.path.join(txt_dir,'group{}.txt'.format(i+1))) as f:
                #         # with open(os.path.join(txt_dir, '{}.txt'.format(task))) as f:
                #             lines = f.readlines()
                #         thr_num = int(0.5 * len(lines))

                #         for idx, line in enumerate(lines):
                #             if idx > thr_num:
                #                 break
                #             img_name, area = line.split(',')
                #             area = int(area)
                #             area_list.append(area)  #用于计算group mean area
                #             img_path = os.path.join(train_img_path, img_name)
                #             if os.path.isfile(img_path):
                #                 self.im_paths.append(img_path)
                #                 self.ys += [600]
                #                 self.group.append(i + 1)
                #                 self.I.append(img_num)
                #                 img_num += 1
                #             else:
                #                 no_img_list.append(img_name)
                #         self.group_len.append(img_num)
                #         self.mean_area.append(np.mean(area_list))
            else:
                if txt_id != -1:
                    area_list = []
                    train_list = os.listdir(train_txt_dir)
                    train_list.sort()
                    cur_txt_dir = '{}/{}'.format(train_txt_dir, train_list[txt_id])
                    with open(os.path.join(cur_txt_dir, '{}.txt'.format(group_name))) as f:
                        lines = f.readlines()
                    for temp_idx,line in enumerate(lines):
                        # if temp_idx > 1000:
                        #     break
                        crop_name,cat,area,*box = line.split()
                        img_path = os.path.join(train_img_path, crop_name + '.jpg')
                        if os.path.isfile(img_path):
                            self.im_paths.append(img_path)
                            self.train_list.append(img_path)
                            self.ys += [600]
                            self.I.append(img_num)
                            img_num += 1
                            self.group.append(int(group_name[-1]))
                            area_list.append(int(area))  # 用于计算group mean area
                        else:
                            no_img_list.append(crop_name)
                    self.group_len.append(img_num)
                    self.mean_area.append(np.mean(area_list))
                    self.detail_info['group_len'] = img_num
                    self.detail_info['mean_area'] = np.mean(area_list)
                else:
                    area_list = []
                    # with open(os.path.join(txt_dir,'group{}.txt'.format(i+1))) as f:
                    with open(os.path.join(txt_dir, '{}.txt'.format(group_name))) as f:
                        lines = f.readlines()
                    thr_num = int(0.5 * len(lines))

                    sample_lines = random.sample(lines, thr_num)
                    for idx, line in enumerate(sample_lines):

                    # for idx, line in enumerate(lines):
                    #     if idx > thr_num:
                    #         break
                        img_name, area = line.split(',')
                        area = int(area)
                        area_list.append(area)  # 用于计算group mean area
                        img_path = os.path.join(train_img_path, img_name)
                        if os.path.isfile(img_path):
                            self.im_paths.append(img_path)
                            self.train_list.append(img_path)
                            self.ys += [600]
                            self.group.append(int(group_name[-1]))
                            self.I.append(img_num)
                            img_num += 1
                        else:
                            no_img_list.append(img_name)
                    self.group_len.append(img_num)
                    self.mean_area.append(np.mean(area_list))
                    self.detail_info['group_len'] = img_num
                    self.detail_info['mean_area'] = np.mean(area_list)
            print('-----{}Number of files that do not exist: {}-----'.format(mode,len(no_img_list)))

        elif self.mode == 'gallery':
            pred_path = root + '/OWOD/sam/val/pred/' + IoUthr
            for i, txt in enumerate(os.listdir(os.path.join(pred_path, 'dataset'))):
                # if img_count>120000:
                #     break
                current_path = os.path.join(pred_path, 'dataset')
                with open(os.path.join(current_path, txt)) as f:
                    lines = f.readlines()
                box_count = 1
                for line in lines:
                    img_path = os.path.join(pred_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    img_no_path = os.path.join(pred_path + '/img',
                                               'no' + txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    box_count += 1
                    if os.path.isfile(img_path):
                        self.ys += [int(line.split(' ', 4)[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)
                    elif os.path.isfile(img_no_path):
                        self.ys += [int(line.split(' ', 4)[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_no_path)
                    else:
                        no_img_list.append(txt.split('.')[0])
            print('------the number of {} is {}------'.format(mode, img_count))

        elif self.mode == 'query':
            query_path = root + '/OWOD/MM/val/query'
            for i, txt in enumerate(os.listdir(os.path.join(query_path, 'dataset'))):
                # if img_count>8000:
                #     break
                current_path = os.path.join(query_path, 'dataset')
                with open(os.path.join(current_path, txt)) as f:
                    lines = f.readlines()
                box_count = 1
                for line in lines:
                    img_path = os.path.join(query_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    box_count += 1
                    if not os.path.isfile(img_path):
                        no_img_list.append(txt.split('.')[0])
                    else:
                        self.ys += [int(line.split(' ', 4)[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)
            print('------the number of {} is {}------'.format(mode,img_count))

    def nb_classes(self):
        debug_list = list(set(self.ys))
        # assert set(self.ys) == set(self.classes)
        return len(debug_list), debug_list

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            # print(index,'\n')
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1: im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]
        paths = self.im_paths[index]
        group = self.group[index]

        return im, paths, target, index, group

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
        self.group = [self.group[i] for i in I]
