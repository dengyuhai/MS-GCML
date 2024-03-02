from .base import *
import os
import random


class validate_set(BaseDataset):
    def __init__(self, root, mode, transform, k_fold_eval=False, fold_idx=0,IoUthr=0.3):

        self.root = root 
        self.mode = mode
        self.transform = transform
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        img_count=0

        no_img = []
        if self.mode=='gallery':
            gallery_path = root + '/evaluate/gallery/'+str(IoUthr)+'/val'
            print('--------------------------------------')
            print(gallery_path)
            print(mode)
            print('--------------------------------------')
            for i, txt in enumerate(os.listdir(os.path.join(gallery_path, 'dataset'))):
                # if img_count>30000:
                #     break
                current_path = os.path.join(gallery_path, 'dataset')
                with open(os.path.join(current_path, txt)) as f:
                    lines = f.readlines()
                box_count = 1
                for line in lines:
                    img_path = os.path.join(gallery_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    img_no_path = os.path.join(gallery_path + '/img',
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
                        no_img.append(txt.split('.')[0])

        elif self.mode=='query':
            query_path = root + '/evaluate/query/val'
            print(mode)
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
                        no_img.append(txt.split('.')[0])
                    else:
                        self.ys += [int(line.split(' ', 4)[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)

class test_set(BaseDataset):
    def __init__(self, root, mode, transform=None, k_fold_eval=False, fold_idx=0,rand_list=[],rand_img=[],rand_cat_num=8,task=None,IoUthr = 0.3,cat_names=[],start_number=[]):

        self.root = root + '/OWOD/sam/'
        self.mode = mode
        self.transform = transform
        self.cat_names = cat_names
        self.start_number = start_number
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        img_count=0

        no_img = []
        if self.mode=='gallery':
            gallery_path = root + '/evaluate/gallery/'+str(IoUthr)+'/test'
            print(mode)
            for i, txt in enumerate(os.listdir(os.path.join(gallery_path, 'dataset'))):
                # if img_count>10000:
                #     break
                current_path = os.path.join(gallery_path, 'dataset')
                with open(os.path.join(current_path, txt)) as f:
                    lines = f.readlines()
                box_count = 1
                for line in lines:
                    img_path = os.path.join(gallery_path + '/img', txt.split('.')[0] + '-' + str(box_count) + '.jpg')
                    img_no_path = os.path.join(gallery_path + '/img',
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
                        no_img.append(txt.split('.')[0])

        elif self.mode=='query':
            query_path = root + '/evaluate/query/test'
            print(mode)
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
                        no_img.append(txt.split('.')[0])
                    else:
                        self.ys += [int(line.split(' ', 4)[0])]
                        self.I += [img_count]
                        img_count += 1
                        self.im_paths.append(img_path)

# if __name__=='__main__':
#     dir_path='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/coco2017/train/img_unlable'
#     list1=os.listdir(dir_path)
#     sample=random.sample(list1,1000)
#     print(sample)