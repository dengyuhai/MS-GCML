# Installation
This code is tested with pytorch 1.12.0 and CUDA 11.6. Follow the below steps for installation.
We have trained and tested our model on Ubuntu 18.04, CUDA 11.6, and PyTorch 1.12.0. You can follow the below steps for installation.
```
conda create -n msgcml python=3.8
conda activate msgcml
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

# Backbone 
We use [GoogLeNet](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and [MiT-B2](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze_connect_hku_hk%2FDocuments%2Fsegformer%2Fpretrained_models&ga=1) backbone pretrained on the ImageNet-1K dataset. In addition, we also use ViT-B/16 backbone trained by the [MoCo-V3](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar) and [CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).

# Dataset

<table align="center">
    <tr>
        <th> </th>
        <th align="center" colspan=1>Images</th>
        <th align="center" colspan=1>Objects</th>
    </tr>
    <tr>
        <td align="left">training dataset</td>
        <td align="center">121298</td>
        <td align="center">8035395</td>
    </tr>
    <tr>
        <td align="left">COCO-VOC test query</td>
        <td align="center">9904</td>
        <td align="center">51757</td>
    </tr>
    <tr>
        <td align="left">COCO-VOC test gallery</td>
        <td align="center">9904</td>
        <td align="center">862861</td>
    </tr>
    <tr>
        <td align="left">BelgaLogos test query</td>
        <td align="center">55</td>
        <td align="center">55</td>
    </tr>
    <tr>
        <td align="left">BelgaLogos test gallery</td>
        <td align="center">10000</td>
        <td align="center">939361</td>
    </tr>
    <tr>
        <td align="left">LVIS test query</td>
        <td align="center">4726</td>
        <td align="center">50537</td>
    </tr>
    <tr>
        <td align="left">LVIS test gallery</td>
        <td align="center">4726</td>
        <td align="center">433671</td>
    </tr>
    <tr>
        <td align="left">Visual Genome test query</td>
        <td align="center">108077</td>
        <td align="center">79921</td>
    </tr>
    <tr>
        <td align="left">Visual Genome test gallery</td>
        <td align="center">108077</td>
        <td align="center">9803549</td>
    </tr>
</table>

## Training set
The training set includes training set of [COCO2017](https://cocodataset.org/#detection-2017) and train-val set of [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html), totalling 121298 images. We utilize [SAM](https://github.com/facebookresearch/segment-anything) to extract 8035395 objects for training. 
## Test set
The test set consists of 862,861 gallery objects extracted using SAM from the validation set of COCO 2017 and the test set of VOC 2007, totaling 9,904 images. The query set contains 51,757 objects extracted based on labeled bounding boxes. 
 

We also conduct evaluation experiments on [BelgaLogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html), [LVIS](https://www.lvisdataset.org/dataset) and [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html). In the BelgaLogos dataset, the provided 55 query logo images are utilized as the query set, while the gallery set comprises 939,361 objects extracted using SAM.For the LVIS dataset, the query set contains 4,726 images and 50,537 labeled objects, while the gallery set includes 433,671 objects extracted using SAM from the images. The Visual Genome dataset encompasses a total of 108,077 images, with 2,516,939 labeled bounding boxes. We define a set of 79,921 objects as the query set, while the gallery set contains 9,803,549 objects extracted using SAM.



# Training
```
bash train.sh
```

# Evaluation
```
bash eval.sh
```