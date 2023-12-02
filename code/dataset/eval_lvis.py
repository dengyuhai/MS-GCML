import os 
from tqdm import tqdm
import numpy as np
import shutil
import cv2
from .base import *

        
class eval_lvis(BaseDataset):
    def __init__(self, root, mode, transform=None, k_fold_eval=False, fold_idx=0):
        super().__init__(root, mode, transform, k_fold_eval, fold_idx)
        self.root = root
        self.mode = mode
        self.transform = transform
        BaseDataset.__init__(self, root, mode, transform, k_fold_eval, fold_idx)
        img_count = 0

        no_img = []
        if self.mode == 'eval':
            pred_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/LVIS/0.3/pred'
            # pred_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/LVIS/val/pred'
            print(mode)
            for i, txt in enumerate(os.listdir(os.path.join(pred_path,'dataset'))):
                current_path = os.path.join(pred_path,'dataset')
                with open(os.path.join(current_path,txt),'r') as f:
                    lines = f.readlines()
                box_count = 1
                for line in lines:
                    img_path =  os.path.join(pred_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    img_no_path = os.path.join(pred_path + '/img',
                                               'no' + txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    box_count += 1
                    if os.path.isfile(img_path):
                        self.ys += [int(line.split()[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)
                    elif os.path.isfile(img_no_path):
                        self.ys += [int(line.split()[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_no_path)
                    else:
                        no_img.append(txt.split('.')[0])
        elif self.mode=='query':
            query_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/LVIS/0.3/query'
            # query_path = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/LVIS/val/query'
            print(mode)
            for i, txt in enumerate(os.listdir(os.path.join(query_path, 'dataset'))):
                # if img_count>8000:
                #     break
                current_path = os.path.join(query_path, 'dataset')
                with open(os.path.join(current_path, txt)) as f:
                    lines = f.readlines()
                box_count = 1
                for count,line in enumerate(lines):
                    img_path = os.path.join(query_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    box_count += 1
                    if not os.path.isfile(img_path):
                        no_img.append(txt.split('.')[0])
                    else:
                        self.ys += [int(line.split()[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)
            print('the number of query is {}'.format(img_count))
        # print('不存在的图片数量有  {}'.format(len(no_img)))


         
    
    
    
        
    
