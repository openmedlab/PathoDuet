# [Model/Code] PathoDuet: Foundation Model for Pathological Slide Analysis of H&E and IHC Stains
<!-- select Model and/or Data and/or Code as needed>
### Welcome to OpenMEDLab! 👋

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


<!-- Insert the project banner here -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/PathoDuet/blob/main/banner.png"></a>
</div>

---

<!-- Select some of the point info, feel free to delete -->
Updated on 2023.12.15. We have revolutionized PathoDuet! Now, the p2/p3 models are named HE/IHC models, and a more detailed figure about our work is updated! ~~The paper is also done, and will be available on arXiv soon.~~ The paper is available now.

~~Updated on 2023.08.04. Sorry for the late release. Now the p3 model is available! The paper link will be available in next update soon.~~



## Key Features

This repository provides the official implementation of PathoDuet: Foundation Models for Pathological Slide Analysis of H&E and IHC Stains.

Key feature bulletin points here
- Foundation models for histopathological image analysis.
- The models cover both H&E and IHC stained images.
- The H&E model achieves outstanding performance on both patch classification (following either a typical linear evaluation scheme, or a usual full fine-tuning scheme) and weakly-supervised WSl classification (using CLAM-SB).
- The IHC model is evaluated with some in-house tasks, and shows great performance.

## Links

- [Model](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM)
- [Paper](https://arxiv.org/abs/2312.09894)
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

Our model is based on a new self-supervised learning (SSL) framework. This framework aims at exploiting characteristics of histopathological images by introducing a pretext token and following task raiser during the training. The pretext token is only a small piece of image, but contains special knowledge. 

In task 1, cross-scale positioning, the pretext token is a small patch contained in a large region. The special relation inspires us to position this patch in the region and use the features of the region to generate the feature of the patch in a global view. The patch is also sent to the encoder solely to obtain a local-view feature. The two features are pulled together to strengthen the H&E model. 

In task 2, cross-stain transferring, the pretext token is a small patch cropped from an image of one stain (H&E). The main input is the image of the other stain (IHC). These two images are roughly registered, so it is possible to style transfer one of them (H&E) to mimic the features of the other (IHC). The pseudo and real features are pulled together to obtain an IHC model on the basis of existing H&E model.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/PathoDuet/blob/main/overall.png"></a>
</div>

## Dataset Links

- [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/) for SSL.
- [NCT-CRC-HE](https://zenodo.org/record/1214456#.YVrmANpBwRk), also known as the Kather datasets, for patch classification.
- [Camelyon 16](https://camelyon16.grand-challenge.org/) for weakly-supervised WSI classification.
- [HyReCo](https://ieee-dataport.org/open-access/hyreco-hybrid-re-stained-and-consecutive-histological-serial-sections) for training in task 2.
- [BCI Dataset](https://bci.grand-challenge.org/) for training in task 2 and evaluation on classification.

## Get Started

**Main Requirements**  
> torch==1.12.1
> 
> torchvision==0.13.1
> 
> timm==0.6.7
> 
> tensorboard
> 
> pandas

**Installation**
```bash
git clone https://github.com/openmedlab/PathoDuet
cd PathoDuet
```

**Download Model**

If you just require a pretrain model for your own task, you can find our pretrained model weights [here](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM). We now provide you two versions of models.

- A model pretrained with cross-scale position task (HE model). This model further strengthens its representation of H&E images.
- A model fine-tuned towards IHC images with cross-stain transferring task (IHC model). This model transfers the strong H&E model to an interpreter of IHC images.

You can try our model by the following codes.

```python
from vits import VisionTransformerMoCo
# init the model
model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
# init the fc layer
model.head = nn.Linear(768, args.num_classes)
# load checkpoint
checkpoint = torch.load(your_checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
# Your own tasks
```

Please note that considering the gap between pathological images and natural images, we do not use a normalize function in data augmentation.

**Prepare Dataset**

If you want to go through the whole process, you need to first prepare the training dataset. The H&E training dataset is cropped from TCGA, and should be arranged as
```bash
TCGA
├── TCGA-ACC
│   ├── patch
│   │   ├── 0_0_1.png
│   │   ├── 0_0_2.png
│   │   └── ...
│   └── region
│       ├── 0_0.png
│       ├── 0_1.png
│       └── ...
├── TCGA-BRCA
│   ├── patch
│   │   └── ...
│   └── region
│       └── ...
└── ...
```
To apply our data generating code, we recommend to install
> openslide

The dataset in task 2 should be arranged like
```bash
root
├── Dataset1
│   ├── HE
│   │   ├── 001.png
│   │   ├── a.png
│   │   └── ...
│   ├── IHC1
│   │   ├── 001.png
│   │   ├── a.png
│   │   └── ...
│   └── IHC2
│       ├── 001.png
│       ├── a.png
│       └── ...
├── Dataset2
│   ├── HE
│   │   └── ...
│   └── IHC
│       └── ...
└── ...
```

**Training**

The code is modified from [MoCo v3](https://github.com/facebookresearch/moco-v3).

For basic MoCo v3 training, 
```python
python main_moco.py \
  --tcga ./used_TCGA.csv \
  -a vit_base -b 2048 --workers 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=100 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dist-backend nccl \
  --dist-url 'tcp://localhost:10001' \
  [your dataset folders]
```

For a further patch positioning pretext task, 
```python
python main_bridge.py \
  --tcga ./used_TCGA.csv \
  -a vit_base -b 2048 --workers 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=20 --warmup-epochs=10 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 --bridge-t=0.5 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dist-url 'tcp://localhost:10001' \
  --ckp ./phase2 \
  --firstphase ./checkpoint_0099.pth.tar \
  [your dataset folders]
```

For a further multi-stain reconstruction task, 
```python
python main_cross.py \
  -a vit_base -b 2048 --workers 128 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=500 --warmup-epochs=100 \
  --stop-grad-conv1 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --dist-url 'tcp://localhost:10001' \
  --ckp ./phase3 \
  --firstphase .phase2/checkpoint_0099.pth.tar \
  [your dataset folders]
```

## Performance on Downstream Tasks

We provide performance evaluation on some downstream tasks, and compare our models with ImageNet-pretrained models (using weights of MoCo v3) and [CTransPath](https://github.com/Xiyue-Wang/TransPath/tree/main). ImageNet has shown its great generalization ability in many pretrained models, so we choose MoCo v3's model as a baseline. CTransPath is also a pretrained model in pathology, which is based on 15 million patches from TCGA and PAIP. CTransPath has shown state-of-the-art performance on many pathological tasks of different diseases and sites. 

**Linear Evaluation**

We use NCT-CRC-HE to evaluate the basic understanding of H&E images. We first follow the typical linear evaluation protocol used in [SimCLR](http://proceedings.mlr.press/v119/chen20j.html), which freezes all layers in the pretrained model and trains a newly-added linear layer from scratch. The result of CTransPath is copied from the original paper, and we also provide a reproduced one marked with a *.
| Methods   |      Backbone      |  ACC |   F1 |
|----------|:-------------:|:------:|:-----:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.935 | 0.908 |
| CTransPath |    Modified Swin Transformer   |   **0.965**  | _0.948_ |
| CTransPath* |    Modified Swin Transformer   |   0.956  | 0.932 |
| Ours-HE | ViT-B/16  |    _0.964_   | **0.950** |

**Full Fine-tuning**

In practice, pretrained models are not freezed. Therefore, we also unfreeze the pretrained encoder and finetune all parameters. It is noted that the performance of CTransPath is based on their open model.
| Methods   |      Backbone      |  ACC |   F1 |
|----------|:-------------:|:------:|:-----:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.958 | 0.945 |
| CTransPath |    Modified Swin Transformer   |   _0.969_ | _0.960_  |
| Ours-HE | ViT-B/16  |    **0.973** | **0.964**  |


**WSI Classification**

For WSI classification, we reproduce the performance of CLAM-SB. Meanwhile, CTransPath filtered out some WSI in TCGA-NSCLC and TCGA-RCC due to some image quality consideration, so the performance of CTransPath is a reproduced one using their open model on the whole dataset, marked as CTP (Repro).

| Methods   |   CAMELYON16: ACC |  CAMELYON16: AUC |  TCGA-NSCLC: ACC |  TCGA-NSCLC: AUC |  TCGA-RCC: ACC |  TCGA-RCC: AUC |
|----------|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| CLAM-SB | _0.884_ | _0.940_ | 0.894 | 0.951 | _0.929_ | 0.986|
| CLAM-SB + CTP (Repro) | 0.868 | _0.940_ | _0.904_ | _0.956_ | 0.928 | _0.987_ |
| CLAM-SB + Ours-HE | **0.930** | **0.956** | **0.908** | **0.963** | **0.954** | **0.993** |


**PD-L1 Expression Level Assessment (IHC images)**

Assessing IHC markers' expression levels is one of the primary tasks for pathologists to evaluate an IHC slide. We formulate this task as a 4-class classification task, with carefully selected thresholds. We compare our IHC model's performance with ImageNet-MoCo v3 and CTransPath as well. The metrics include accuracy (ACC), balanced accuracy (bACC) and weighted F1 score (wF1). Here, we give the performance with a limited amount of training data. 

| Methods   |      Backbone      |  ACC |   bACC | wF1 |
|----------|:----------:|:------:|:-----:|:------:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.686 | 0.698 | 0.695 |
| CTransPath |    Modified SwinT   |  _0.700_   | _0.709_ | _0.703_ |
| Ours-IHC | ViT-B/16  |   **0.726**    | **0.721** | **0.732** | 


**Cross-Site Tumor Identification (IHC images)**

Tumor identification is also of great importance. We formulate this task as a 2-class classification task, with/without tumor cells in the given patch. The metrics include accuracy (ACC) and F1 score (F1). Here, we give the performance in the case of 1) an in-site setting, and 2) an out-of-distribution setting. In the first setting, we use a small group of data from site 1 to train the models in a linear protocol, and evaluate on another group of data from site 1. In the second setting, we train the models with more data in site 1, and evaluate on data from an unseen site 2.

| Methods   |      Backbone      |  ACC |   F1 | ACC (OOD) | F1 (OOD) |
|----------|:----------:|:------:|:-----:|:------:|:------:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.864 | 0.862 | 0.504 | 0.503 |
| CTransPath |    Modified SwinT   |  _0.872_   | _0.870_ | _0.677_ | _0.657_ |
| Ours-IHC | ViT-B/16  |   **0.900**    | **0.900** | **0.826** | **0.769** |



**Comparison with Giant Pathological Models**

We also compare our model to some giant models pretrained with ultra-large amounts of pathological slides, namely UNI and Virchow. We use the NCT-CRC-HE and NCT-CRC-HE-NONORM (marked with a *), and copy the results from Virchow. To note, the result of CTransPath is also a copy from Virchow, so it is slightly different from previous results reproduced by us, but the gap is as small as 0.001 or 0.002, which is acceptable as randomness. The training parameters are similar to Virchow's. but we change the batch size to 512 and a rescaled learning rate as 0.001/8=0.000125, and we use typical augmentations like random crop and scale, random flip and random rotation. 

| Methods   |      Backbone      |      #WSIs      |  ACC |   bACC | wF1 |  ACC* |   bACC* | wF1* |
|----------|:----------:|:------:|:-----:|:------:|:------:|:------:|:------:|:------:|
| Ours-H&E |  ViT-B | ~11K | _0.964_ | _0.952_ | _0.964_ | _0.888_ | _0.875_ | _0.894_ |
| CTransPath |    Modified SwinT   |  ~32K | 0.958   | 0.931 | 0.955 | 0.879 | 0.852 | 0.883 |
| UNI | ViT-L  |  ~100K |  -    | - | - |- | 0.874 | 0.875 |
| Virchow | ViT-H  |  ~1.5M |  **0.968**    | **0.956** | **0.968** | **0.948** | **0.938** | **0.950** |

**More Results can be found in our later released paper!**


## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

- Shanghai AI Laboratory.
- Qingyuan Research Institute, Shanghai Jiao Tong University.

## 📝 Citation

If you find this repository useful, please consider citing our arXiv paper.
```
@misc{hua2023pathoduet,
      title={PathoDuet: Foundation Models for Pathological Slide Analysis of H&E and IHC Stains}, 
      author={Shengyi Hua and Fang Yan and Tianle Shen and Xiaofan Zhang},
      year={2023},
      eprint={2312.09894},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
