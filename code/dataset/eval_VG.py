from __future__ import print_function
from __future__ import division

import os
import shutil
import cv2
import torch
from os.path import join
import torchvision
import numpy as np
import PIL.Image
from tqdm import tqdm


class eval_VG(torch.utils.data.Dataset):
    def __init__(self,mode,image_dir,transform,iou_thr=0.3):
        self.image_base_dir = image_dir
        self.transform = transform
        self.ys, self.I = [], []
        self.image_paths = []
        self.mode = mode
        no_img_list = []

        if mode == 'query':
            self.txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/VG/GT_box_id'
            self.selected_txt = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/VG/selected_query.txt'
            number_count = 0
            img_dir = '{}/{}'.format(self.image_base_dir,'query')
            # for temp_count,txt in enumerate(tqdm(os.listdir(self.txt_dir))):
            #     if temp_count > 999:
            #         break
            #     image_id = txt.split('.')[0]
            #     with open(join(self.txt_dir, txt)) as f:
            #         lines = f.readlines()
            #     for box_index, line in enumerate(lines):
            #         cate,cate_name, x1, y1, w, h = line.split('|')
            #         box = [int(x1), int(y1), int(x1)+int(w), int(y1)+int(h)]
            #         if (box[0] >= box[2]) or (box[1] >= box[3]):
            #             continue
            #         else:
            #             self.box_points.append(box)
            #             self.img_names.append(image_id)
            #             self.ys.append(int(cate))
            #             self.box_idxs.append(box_index)
            #             self.I.append(number_count)
            #             number_count += 1
            
            with open(self.selected_txt) as f:
                lines = f.readlines()
            for line in lines:
                cat_id, txt_box = line.split(' ')
                txt,box_count = txt_box.split('-')
                image_path = os.path.join(img_dir,'{}-{}.jpg'.format(txt,int(box_count)))
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.ys.append(int(cat_id))
                    self.I.append(number_count)
                    number_count += 1
                else:
                    no_img_list.append(image_path)

                # if os.path.exists(os.path.join(self.txt_dir, '{}.txt'.format(txt))):
                #     with open(os.path.join(self.txt_dir, '{}.txt'.format(txt))) as detailed_txt:
                #         d_lines = detailed_txt.readlines()
                #     select_line = d_lines[int(box_count)-1]
                #     cat_id2,_,x,y,w,h = select_line.split('|')
                #     box = [int(x), int(y), int(x)+int(w), int(y)+int(h)]
                #     if (box[0] >= box[2]) or (box[1] >= box[3]):
                #         continue
                #     else:
                #         self.box_points.append(box)
                #         self.img_names.append(txt)
                #         self.ys.append(int(cat_id))
                #         self.box_idxs.append(int(box_count)-1)
                #         self.I.append(number_count)
                #         number_count += 1
                # else:
                #     raise RuntimeError('不存在{}'.format(txt))
            
        else:
            img_dir = '{}/{}'.format(self.image_base_dir, 'pred')
            self.txt_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/VG/{}'.format(iou_thr)
            number_count = 0
            for temp_count,txt in enumerate(tqdm(os.listdir(self.txt_dir))):
                # if temp_count > 9999:
                #     break
                image_id = txt.split('.')[0]
                with open(join(self.txt_dir,txt)) as f:
                    lines = f.readlines()
                for box_index, line in enumerate(lines):
                    img_path = '{}/{}-{}.jpg'.format(img_dir,txt.split('.')[0],box_index)
                    if os.path.exists(img_path):
                        cate,x1,y1,x2,y2 = line.split()
                        box = [int(x1),int(y1),int(x2),int(y2)]
                        if (box[0] >= box[2]) or (box[1] >= box[3]):
                            continue
                        else:
                            self.ys.append(int(cate))
                            self.I.append(number_count)
                            number_count += 1
                            self.image_paths.append(img_path)
                    else:
                        no_img_list.append(img_path)
        print('{}不存在的图片数量为{}\n'.format(mode,len(no_img_list)))


    def img_load(self,index):
        # image = PIL.Image.open('{}/{}.jpg'.format(self.image_dir,self.img_names[index]))
        # x1,y1,x2,y2 = self.box_points[index]
        # # height,width = image.height,image.width
        # # x1,y1,x2,y2 = max(0,x1),max(0,y1),min(x2,width),min(y2,height)
        # box_image = image.crop((x1,y1,x2,y2))
        # # print(box_image.split())
        # if len(list(box_image.split())) == 1 : box_image = box_image.convert('RGB')
        # box_image = self.transform(box_image)
        # return box_image
        im = PIL.Image.open(self.image_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1: im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im


    def __getitem__(self, index):
        # index是数组吗？是Int类型
        im = self.img_load(index)
        target = self.ys[index]
        paths = self.image_paths[index]#'./{}-{}.jpg'.format(self.img_names[index],self.box_idxs[index]+1)

        return im,paths,target,index


    def get_label(self,index):
        return self.ys[index]
    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
    def __len__(self):
        return len(self.ys)

        
