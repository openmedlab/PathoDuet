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

Updated on 2023.06.29



## Key Features

This repository provides the official implementation of PathoDuet: Foundation Model for Pathological Slide Analysis of H&E and IHC Stains.

Key feature bulletin points here
- A foundation model for histopathological image analysis.
- The model covers both H&E and IHC stained images.
- The model achieves outstanding performance on both linear evaluation and full finetuning of patch classification.

## Links

- [Paper](https://) 
- [Model](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM)
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->
## Details

Our model is based on a new self-supervised learning (SSL) framework. This framework aims at exploiting characteristics of histopathological images by introducing a pretext token during the training. The pretext token is only a small piece of image, but contains special knowledge. 

In task 1, patch positioning, the pretext token is a small patch contained in a large region. The special relation inspires us to position this patch in the region and use the features of the region to generate the feature of the patch in a global view. The patch is also sent to the encoder solely to obtain a local-view feature. The two features are pulled together to strengthen the model. 

In task 2, multi-stain reconstruction, the pretext token is a small patch cropped from an image of one stain (IHC). The main input is the image of the other stain (H&E), and gets masked before the encoder. These two images are roughly registered, so it is possible to recover the image of the first stain giving only a small part of itself (stain information) and a whole image of the second stain (structure information). The two parts of features are concatenated and sent into the decoder, and finally reconstruct the image of the first stain.

<!-- Insert a pipeline of your algorithm here if got one -->
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/PathoDuet/blob/main/overall.png"></a>
</div>

## Dataset Links

- [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/) for SSL.
- [UniToPatho](https://ieee-dataport.org/open-access/unitopatho) for patch retrieval.
- [NCT-CRC-HE](https://zenodo.org/record/1214456#.YVrmANpBwRk) and [CRC](https://zenodo.org/record/53169#.YRfeKYgzbmE), also known as the Kather datasets, for patch classification.
- [Camelyon 16](https://camelyon16.grand-challenge.org/) for weakly-supervised WSI classification.

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

If you just require a pretrain model for your own task, you can find our pretrained model weights [here](https://drive.google.com/drive/folders/1aQHGabQzopSy9oxstmM9cPeF7QziIUxM). We now provide you two versions of models, a purely MoCo v3 pretrained model using our H&E dataset (later referred to as p1), and a model pretrained with patch positioning (later referred to as p2). The model pretrained with multi-stain transferring will be released soon.

**Prepare Dataset**

If you want to go through the whole process, you need to first prepare the training dataset. The training dataset is cropped from TCGA, and should be arranged as
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

**Training**

The code is modified from [MoCo v3](https://github.com/facebookresearch/moco-v3).

For basic MoCo v3 training, 
```bash
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
```bash
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

## Performance on Downstream Tasks

**Linear Evaluation**

We first follow the typical linear evaluation protocol used in [SimCLR](http://proceedings.mlr.press/v119/chen20j.html), which freezes all layers in the pretrained model and trains a newly-added linear layer from scratch. 
| Methods   |      Backbone      |  ACC |   F1 |
|----------|:-------------:|:------:|:-----:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.9347 | 0.9083 |
| CTransPath |    Modified Swin Transformer   |   0.9556  | 0.9317 |
| Ours-p1 | ViT-B/16  |    0.9561   | 0.9437 |
| Ours-p2 | ViT-B/16  |    **0.9639**   | **0.9496** |

**Full Fine-tuning**

In practice, pretrained models are not freezed. Therefore, we also unfreeze the pretrained encoder and finetune all parameters.
| Methods   |      Backbone      |  ACC |   F1 |
|----------|:-------------:|:------:|:-----:|
| ImageNet-MoCo v3 |  ViT-B/16 | 0.9582 | 0.9450 |
| CTransPath |    Modified Swin Transformer   |   0.9691 | 0.9601  |
| Ours-p1 | ViT-B/16  |    0.9726 | 0.9602  |
| Ours-p2 | ViT-B/16  |    **0.9731** | **0.9636**  |


**WSI Classification**

For WSI classification
| Methods   |   CAMELYON16: ACC |  CAMELYON16: AUC |  TCGA-NSCLC: ACC |  TCGA-NSCLC: AUC |  TCGA-RCC: ACC |  TCGA-RCC: AUC |
|----------|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| CLAM-SB | 0.8837 | 0.9395 | 0.8943 | 0.9511 | 0.9285 | 0.9864|
| CLAM-SB + CTransPath | 0.8682 | 0.9403 | 0.9040 | 0.9559 | 0.9275 | 0.9875|
| CLAM-SB + Ours-p1 | 0.9147 | **0.9589** | 0.8898 | 0.9368 | 0.9478 | 0.9916|
| CLAM-SB + Ours-p2 | **0.9302** | 0.9561 | **0.9075** | **0.9631** | **0.9542** | **0.9929** |



## 🛡️ License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

- Shanghai AI Laboratory.
- Qingyuan Research Institute, Shanghai Jiao Tong University.

## 📝 Citation

If you find this repository useful, please consider citing our later released paper.

