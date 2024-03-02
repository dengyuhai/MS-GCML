import os
import re
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import random

T1_CLASS_NAMES=['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'boat', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'bottle',
                'chair', 'couch', 'potted plant', 'dining table', 'tv']
T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]
T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]
T4_CLASS_NAMES = [
    "bed", "toi let", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]
VOC_CLASS_NAMES=['person','bicycle','car','motorbike','aeroplane',
                 'bus','train','boat','bird','cat',
                 'dog','horse','sheep','cow','bottle',
                 'chair','sofa','pottedplant','diningtable','tvmonitor','background']
VOC_CLASS_ID=['15','2','7','14','1',
              '6','19','4','3','8',
              '12','13','17','10','5',
              '9','18','16','11','20','0']
T1_id_voc=[1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72,0]
T1_id=[1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
T2_id=[8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 78, 79, 80, 81, 82]
T3_id=[34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
T4_id=[46, 47, 48, 49, 50, 51, 65, 70, 73, 74, 75, 76, 77, 84, 85, 86, 87, 88, 89, 90]
dict_map_voc = dict(zip(VOC_CLASS_NAMES, T1_id_voc))
# dict_map_coco=dict(zip(T1_CLASS_NAMES, T1_id))
# dict_map_coco.update(zip(T2_CLASS_NAMES, T2_id))
# dict_map_coco.update(zip(T3_CLASS_NAMES, T3_id))
# dict_map_coco.update(zip(T4_CLASS_NAMES, T4_id))

def read_txt(txt_path):
    with open(txt_path) as f:
        lines=f.readlines()
    return [line[:-1] for line in lines]

def calc_area(box):
    '''
    Calculate the area of a box, which is represented by the two ends of the main diagonal
    '''
    return ((box[2]-box[0])*(box[3]-box[1]))

def calc_iou(box1,box2):
    '''
    box1,box2 is the box represented by the upper left and lower right corners
    Calculate the intersection ratio between two boxes
    '''
    box1=np.array(box1,dtype=float)
    box2 = np.array(box2, dtype=float)
    h_in=max(0,min(box1[2],box2[2])-max(box1[0],box2[0]))
    w_in=max(0,min(box1[3],box2[3])-max(box1[1],box2[1]))
    union=calc_area(box1)+calc_area(box2)
    iou=(h_in*w_in)/(union-h_in*w_in)
    return iou

def delet_redundancy(x_txt,y_txt,new_txt_save_path):
    '''
    x_txt: indicates the path of the txt file to be deleted
    y_txt: txt file path to be queried
    new_txt_save_path: Save path for deleting intersection files
    '''
    val_list = set(read_txt(y_txt))
    train_list = read_txt(x_txt)
    train_remove_val = []
    for idx, ele in enumerate(tqdm(train_list)):
        if ele not in val_list:
            train_remove_val.append(ele)
    with open(new_txt_save_path,'w+') as f:
        for ele in train_remove_val:
            f.write(ele)
    return train_remove_val



def extract_pred_box_for_train_txt(img_dir,pred_txt_dir,GT_txt_dir,save_dir,train_cat):
    '''
    img_dir: directory of image datasets to read
    pred_txt_dir: txt directory of pred_box
    GT_txt_dir: txt directory of GT_box
    save_dir: txt directory for training box
    train_cat: Training data category of the task
    Function: Extract training sets of T1, T2, T3 and T4
    '''
    os.makedirs(save_dir, exist_ok=True)
    img_lists = os.listdir(img_dir)
    img_lists=[line.split('.')[0] for line in img_lists]
    for idx, img in enumerate(tqdm(img_lists)):
        pred_txt = os.path.join(pred_txt_dir, img + '.txt')
        GT_txt = os.path.join(GT_txt_dir, img + '.txt')
        if os.path.isfile(GT_txt):
            with open(GT_txt) as f:
                GT_boxes = f.readlines()
        else:
            continue
        GT_train_boxes = []
        GT_train_cats = []
        for line in GT_boxes:
            line=line[:-1]
            *cat, xmin, ymin, xmax, ymax = line.split(' ')
            cat_str = ''
            for str1 in cat:
                cat_str += ' ' + str1
            cat = cat_str[1:]
            if int(cat) in train_cat:
                GT_train_boxes.append([xmin, ymin, xmax, ymax])
                GT_train_cats.append(int(cat))
        if os.path.isfile(pred_txt):
            with open(pred_txt) as f:
                pred_boxes = f.readlines()
        train_boxes = []
        train_cats = []
        for line in pred_boxes:
            line=line[:-1]
            xmin, ymin, xmax, ymax = line.split(' ')
            box = [xmin, ymin, xmax, ymax]
            max_iou = 0
            fitest_cat = -1
            for gt_box,gt_cat in zip(GT_train_boxes,GT_train_cats):
                iou = calc_iou(box,gt_box)
                if iou > 0 and iou > max_iou:
                    max_iou = iou
                    fitest_cat = gt_cat
            if max_iou != 0:
                train_boxes.append(box)
                train_cats.append(fitest_cat)
        with open(os.path.join(save_dir, img + '.txt'), 'w+') as f:
            for pred_box,fit_cat in zip(train_boxes,train_cats):
                f.write(f"{fit_cat} {pred_box[0]} {pred_box[1]} {pred_box[2]} {pred_box[3]}\n")

    return save_dir

def uni_cls_id():
    dataset_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/voc/VOCdevkit/VOC2007/dataset'
    save_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/voc/VOCdevkit/VOC2007/dataset_id'
    os.makedirs(save_dir, exist_ok=True)

    for idx, txt in enumerate(tqdm(os.listdir(dataset_dir))):
        GT_txt = os.path.join(dataset_dir, txt)
        if os.path.isfile(GT_txt):
            with open(GT_txt) as f:
                GT_boxes = f.readlines()
        else:
            continue
        GT_train_boxes = []
        for line in GT_boxes:
            line = line[:-1]
            *cat_names, xmin, ymin, xmax, ymax = line.split(' ')
            cat_str = ''
            for str1 in cat_names:
                cat_str += ' ' + str1
            cat_names = cat_str[1:]
            cat_id = dict_map_voc[cat_names]
            if cat_id == None:
                raise RuntimeError('category error')
            GT_train_boxes.append([cat_id, xmin, ymin, xmax, ymax])
        with open(os.path.join(save_dir, txt), 'w+') as f:
            for box in GT_train_boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

def process_pred_box(GT_box_dir,pred_box_dir,processed_save_dir,thr=0.5):
    os.makedirs(processed_save_dir, exist_ok=True)
    for idx, txt in enumerate(tqdm(os.listdir(pred_box_dir))):
        processed_boxes = []
        pred_boxes = read_txt(os.path.join(pred_box_dir, txt))
        if not os.path.isfile(os.path.join(GT_box_dir, txt)):
            raise RuntimeError('GT does not exist{}'.format(txt))
        GT_boxes = read_txt(os.path.join(GT_box_dir, txt))
        for pred in pred_boxes:
            _, *pred_box = pred.split(' ')
            # pred_box = pred.split(' ')
            best_iou_cls=255
            best_iou=0
            temp_box=np.array(pred_box,dtype=int)
            if temp_box[0]>=temp_box[2] or temp_box[1]>=temp_box[3]:
                continue
            for GT_ele in GT_boxes:
                cls, *GT_box = GT_ele.split(' ')
                iou=calc_iou(pred_box, GT_box)
                if iou>best_iou:
                    best_iou=iou
                    if iou >= thr:
                        best_iou_cls=int(cls)
                    else:
                        best_iou_cls=int(cls)+100
            processed_boxes.append([best_iou_cls, pred_box[0], pred_box[1], pred_box[2], pred_box[3]])

        with open(os.path.join(processed_save_dir, txt), 'w+') as f:
            for ele in processed_boxes:
                f.write(f"{ele[0]} {ele[1]} {ele[2]} {ele[3]} {ele[4]}\n")

def get_img_train(processed_pred_dir,test_img_dir,save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_count = 0
    for idx, txt in enumerate(tqdm(os.listdir(processed_pred_dir))):
        img_path = os.path.join(test_img_dir, txt.split('.')[0] + '.jpg')
        if not os.path.isfile(img_path):
            continue
        train_count +=1
        img = cv2.imread(img_path)
        h,w,_=np.shape(img)
        preds = read_txt(os.path.join(processed_pred_dir, txt))
        number = 1
        for pred in preds:
            box = pred.split(' ')
            box = np.array(box, dtype=int)
            box = np.clip(box, 0, 10000)

            # raise RuntimeError('error')
            x2 = min(w, box[2])
            y2 = min(h, box[3])
            if box[1] >= y2 or box[0]  >= x2:
                continue
            crop_img = img[box[1]:y2, box[0]:x2]
            save_img_name = txt.split('.')[0] + '-' + str(number) + '.jpg'
            number += 1
            cv2.imwrite(os.path.join(save_dir, save_img_name), crop_img)
    print('There are {} training pictures in total'.format(train_count))


def get_img_test(processed_pred_dir,test_img_dir,save_dir):
    '''
    Get a picture of the test set box (prediction box img of val test)
    processed_pred_dir: txt directory of the prediction box that has been processed
    test_img_dir: The path of the image to which the box belongs
    save_dir: Save path of the extracted prediction box image
    '''
    os.makedirs(save_dir, exist_ok=True)
    for idx, txt in enumerate(tqdm(os.listdir(processed_pred_dir))):
        img_path = os.path.join(test_img_dir, txt.split('.')[0] + '.jpg')
        if not os.path.isfile(img_path):
            raise RuntimeError('No image {}'.format(img_path))
        img = cv2.imread(img_path)
        h,w,_=np.shape(img)
        preds = read_txt(os.path.join(processed_pred_dir, txt))
        number = 1
        for pred in preds:
            cls, *box = pred.split(' ')
            box = np.array(box, dtype=int)
            box = np.clip(box, 0, 10000)

            cls = int(cls)
            # if box[0]>=box[2] or box[1]>=box[3]:
            #     number+=1
            #     continue
            if cls < 100:
                x2=min(w-1,box[2])
                y2=min(h-1,box[3])
                crop_img = img[box[1]:y2, box[0]:x2]
                save_img_name = txt.split('.')[0] + '-' + str(number) + '.jpg'
                number += 1
                # print(txt)
                cv2.imwrite(os.path.join(save_dir, save_img_name), crop_img)
            elif cls > 200:
                x2 = min(w - 1, box[2])
                y2 = min(h - 1, box[3])
                crop_img = img[box[1]:y2, box[0]:x2]
                save_img_name = 'no' + txt.split('.')[0] + '-' + str(number) + '.jpg'
                number += 1
                cv2.imwrite(os.path.join(save_dir, save_img_name), crop_img)
            else:
                # raise RuntimeError('error')
                x2 = min(w - 1, box[2])
                y2 = min(h - 1, box[3])
                crop_img = img[box[1]:y2, box[0]:x2]
                save_img_name = txt.split('.')[0] + '-' + str(number) + '.jpg'
                number += 1
                cv2.imwrite(os.path.join(save_dir, save_img_name), crop_img)

def convert_annotation(image_index,xml_dir=None,txt_save_dir=None):
    """
    Convert the xml file of the image image_id to a label file (txt) for object detection
    It contains the category of the object, the coordinates of the top-left point of the bbox, and the width and height of the bbox
    And the four physical quantities are normalized
    """
    f = open(os.path.join(xml_dir,image_index))
    image_name = image_index.split('.')[0]
    os.makedirs(txt_save_dir,exist_ok=True)
    out_file = open(txt_save_dir+'/%s.txt' % (image_name), 'w')
    tree = ET.parse(f)
    root = tree.getroot()
    size = root.find('size')

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes: #or int(difficult) == 1:
        #     print(cls)
        #     continue
        cls_id = dict_map_voc[cls]
        xmlbox = obj.find('bndbox')
        # points = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                  float(xmlbox.find('ymax').text))
        out_file.write(str(cls_id) + ' ' + ' '.join([str(int(a)) for a in points]) + '\n')
        # out_file.write(cls + ' ' + ' '.join([str(int(a)) for a in points]) + '\n')
    out_file.close()

def calc_redundant():
    train_img = os.listdir('/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/train_img_all')
    for i in range(len(train_img)):
        train_img[i] = train_img[i].split('.')[0]
    val_dataset = os.listdir(
        '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/split_dataset_process/val/GT_dataset')
    for i in range(len(val_dataset)):
        val_dataset[i] = val_dataset[i].split('.')[0]
    test_dataset = os.listdir(
        '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/split_dataset_process/eval/img_all')
    for i in range(len(test_dataset)):
        test_dataset[i] = test_dataset[i].split('.')[0]
    val_dataset = set(val_dataset)
    test_dataset = set(test_dataset)
    train_dataset = []
    val_num = 0
    test_num = 0
    for img_name in tqdm(train_img):
        if img_name in test_dataset:
            test_num += 1
        elif img_name in val_dataset:
            val_num += 1
        else:
            train_dataset.append(img_name)
    print('test {}'.format(test_num))
    print('val {}'.format(val_num))
    print('train {}'.format(len(train_dataset)))



if __name__=='__main__':
    task_lists = ['t1','t2','t3','t4']
    train_cats = [T1_id,T2_id,T3_id,T4_id]
    pred_txt_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/sam_txt_all'
    GT_txt_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/coco_voc_train_GT_txt'
    for ii in range(4):
        task_name = task_lists[ii]
        train_cat = train_cats[ii]
        img_txt = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/split_dataset_process/train/{}_dataset'.format(task_name)
        save_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/{}/dataset_stastic'.format(task_name)
        os.makedirs(save_dir, exist_ok=True)
        extract_pred_box_for_train_txt(img_dir=img_txt, pred_txt_dir=pred_txt_dir, GT_txt_dir=GT_txt_dir, save_dir=save_dir,
                                   train_cat=train_cat)
        
    # Steps of data processing:
    # train processing: Correct sam's box -&gt; extract_pred_box_for_train_txt gets the training set of each task -&gt; get_img goes to crop picture
    
    # # print(len(os.listdir('/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/t1/img')))
    # img_txt = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/split_dataset_process/train/t1_dataset'
    # pred_txt_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/sam_txt_all'
    # GT_txt_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/coco_voc_train_GT_txt'
    # save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/t1/dataset'
    # train_cat=T1_id
    # extract_pred_box_for_train_txt(img_dir=img_txt,pred_txt_dir=pred_txt_dir,GT_txt_dir=GT_txt_dir,save_dir=save_dir,train_cat=train_cat)
    #
    # get_img_train(processed_pred_dir=save_dir,
    #         test_img_dir='/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/train_img_all' ,
    #         save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/t1/img')#'/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/t1/img')


    # for i in range(len(train_dataset)):
    #     train_dataset[i]+='.jpg'
    # train_img_dir = '/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/train_img'
    # print(len(os.listdir(train_img_dir)))
    #
    # get_img_train(processed_pred_dir='/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/sam_txt_all',
    #               test_img_dir=train_img_dir,
    #               save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/train/all/img')

    # uni_cls_id()



    # test处理： 将coco图片名id加上1000000-> process_pred_box 预处理 -> get_img 去crop图片
    # sam_pred_box='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test/pred/dataset'
    # GT_box_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test/query/dataset'
    # save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test/pred/0.3/dataset'
    # # # #
    # # process_pred_box(GT_box_dir,sam_pred_box,save_dir,thr=0.3)
    # #
    # get_img_test(processed_pred_dir=save_dir,
    #              test_img_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test_img/rename_all',
    #              save_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test/pred/0.3/img')
    #
    # print(len(os.listdir('/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/test/pred/0.3/img')))


    # all_sasm_txt_dir='/mnt/953da527-d456-4a74-b00d-27844a759cf1/SAM_extract/sam_txt_all'
    # val_dataset='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/split_dataset_process/val/img_all'
    # sam_val_dataset='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/val/pred/dataset'
    # os.makedirs(sam_val_dataset,exist_ok=True)
    # for img_name in os.listdir(val_dataset):
    #     txt_name=img_name.split('.')[0]+'.txt'
    #     if os.path.isfile(os.path.join(all_sasm_txt_dir,txt_name)):
    #         shutil.copy(os.path.join(all_sasm_txt_dir,txt_name),sam_val_dataset)
    # print(len(os.listdir(sam_val_dataset)))

    # test_dir='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/MM/test/query/dataset'
    # txt_id=[]
    # for txt in os.listdir(test_dir):
    #     txt_id.append(int(txt.split('.')[0]))
    # print(len(list(set(txt_id))))
    # dst_dir = '/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/val/pred/dataset'#'/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/MM/test/query/correct_dataset'
    # for txt in os.listdir(dst_dir):
    #     if len(txt.split('.')[0])>=12:
    #         img_id,other = txt.split('.')
    #         if img_id[:2]!='no':
    #             txt_id = str(int(img_id) + 1000000)
    #             str_id = '000000000000'
    #             str_id = str_id[:12 - len(txt_id)] + txt_id
    #             os.rename(os.path.join(dst_dir, txt), os.path.join(dst_dir, str_id +'.'+ other))
    #         else:
    #             txt_id=str(int(img_id[2:]) + 1000000)
    #             str_id = '000000000000'
    #             str_id = str_id[:12 - len(txt_id)] + txt_id
    #             os.rename(os.path.join(dst_dir, txt), os.path.join(dst_dir, 'no'+str_id +'.'+ other))
    #     elif len(txt.split('.')[0])==11:
    #         img_id, other = txt.split('.')
    #         year,id=img_id.split('_')
    #         str_id=year+str(int(id))
    #         os.rename(os.path.join(dst_dir, txt), os.path.join(dst_dir, str_id + '.' + other))
    # print(len(os.listdir(dst_dir)))


