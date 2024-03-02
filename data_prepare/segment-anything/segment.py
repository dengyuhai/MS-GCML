import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import os.path

import torch
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
from tqdm import tqdm
from logger import setup_logger
import argparse

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


from segment_anything.utils.transforms import ResizeLongestSide

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def predict_example():
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image1 = cv2.imread('./notebooks/images/truck.jpg')  # truck.jpg from above
    image1_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device=sam.device)

    image2 = cv2.imread('./notebooks/images/groceries.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2_boxes = torch.tensor([
        [450, 170, 520, 350],
        [350, 190, 450, 350],
        [500, 170, 580, 350],
        [580, 170, 640, 350],
    ], device=sam.device)

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    batched_input = [
        {
            'image': prepare_image(image1, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
            'original_size': image1.shape[:2]
        },
        {
            'image': prepare_image(image2, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
            'original_size': image2.shape[:2]
        }
    ]

    batched_output = sam(batched_input, multimask_output=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(image1)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in image1_boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[0].axis('off')

    ax[1].imshow(image2)
    for mask in batched_output[1]['masks']:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    for box in image2_boxes:
        show_box(box.cpu().numpy(), ax[1])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def plot_box_respectivly(img_path, box_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        for box_txt_name in os.listdir(box_path):
            if box_txt_name.split('.', 1)[0] == img_name.split(".", 1)[0]:
                print(box_txt_name)
                with open(os.path.join(box_path, box_txt_name)) as f:
                    lines = f.readlines()
                boxes = [np.array(line.split(' ', 4), dtype=int) for line in lines]

                sub_dir = os.path.join(save_path, img_name.split(".", 1)[0])
                os.makedirs(sub_dir, exist_ok=True)
                if len(boxes) == 0:
                    break
                count = 0
                for box in boxes:
                    count += 1
                    pic = cv2.imread(os.path.join(img_path, img_name))
                    temp_pic=pic[box[3]:box[1],box[0]:box[2]]
                    cv2.rectangle(pic, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
                    # cv2.rectangle(temp_pic, (box[0], box[3]), (box[2], box[1]), (0, 0, 255), 5)
                    cv2.imwrite(os.path.join(sub_dir, str(count) + '.png'), pic)
                    cv2.imwrite(os.path.join(sub_dir, str(count) + '.jpg'),temp_pic)
                break

def correct_box(ori_dir,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for idx , txt in enumerate(tqdm(os.listdir(ori_dir))):
        with open(os.path.join(ori_dir,txt)) as f:
            lines=f.readlines()
        lines=[line[:-1] for line in lines]
        with open(os.path.join(save_dir,txt),'w+') as f:
            for box in lines:
                x1,y1,x2,y2=box.split(' ')
                f.write(f"{x1} {y1} {x2} {y2}\n")




if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                                     'segment images using SAM')
    parser.add_argument('--image_dir', default='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/collect',type=str,
                        help='Directory of the image file to be segmented')
    parser.add_argument('--save_dir', default='/mnt/6c707c34-d721-41ee-94dc-af88c3c9a6ad/STMLdata/OWOD/sam/collect',type=str,
                        help='Save path of the segmented picture Box')
    parser.add_argument('--gpu_id', default=0,type=int,
                        help='ID of GPU that is used')
    args = parser.parse_args()
    log = setup_logger('SAM_voc_train','/home/msi/PycharmProjects/segment-anything-main/log',0)
    gpu_id = args.gpu_id
    img_path=args.image_dir
    save_txt_path=args.save_dir
    os.makedirs(save_txt_path,exist_ok=True)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    sam.cuda(gpu_id)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for idx,img_name in enumerate(tqdm(os.listdir(img_path))):
        # if img_name.split('.')[0] in prev_set:
        #     continue
        image = cv2.imread(os.path.join(img_path,img_name))
        log.info('image name  {}'.format(img_name))
        if len(np.shape(image)) == 0:
            log.error('{}  is None'.format(img_name))
            print(img_name)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)

        boxes = []
        for ele in masks:
            box = np.array(ele['bbox'], dtype=int)
            boxes.append(np.array([box[0],
                                   box[1] ,
                                   box[0] + box[2],
                                   box[1]+ box[3]], dtype=int))
        with open(os.path.join(save_txt_path, img_name.split('.')[0] + '.txt'), 'w+') as f:
            for pred_box in boxes:
                f.write(f"{pred_box[0]} {pred_box[1]} {pred_box[2]} {pred_box[3]}\n")


