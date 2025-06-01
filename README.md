# [CVPR 2025 Highlight] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mamba-as-a-bridge-where-vision-foundation/domain-generalization-on-gta-to-avg)](https://paperswithcode.com/sota/domain-generalization-on-gta-to-avg?p=mamba-as-a-bridge-where-vision-foundation)
### [**Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation**](https://arxiv.org/abs/2504.03193)
>[Xin Zhang](https://devinxzhang.github.io/)\, [Robby T. Tan](https://tanrobby.github.io/)\
>National University of Singapore\
>CVPR 2025



#### [[`Project Page`](https://devinxzhang.github.io/MFuser_ProjPage/)] [[`Paper`](https://arxiv.org/abs/2504.03193)]

## Environment
### Requirements
- The requirements can be installed with:
  
  ```bash
  conda create -n mfuser python=3.9 numpy=1.26.4
  conda activate mfuser
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install xformers==0.0.20
  pip install mmcv-full==1.5.1 
  pip install mamba_ssm==2.2.2
  pip install causal_conv1d==1.4.0
  ```

## Pre-trained VFM & VLM Models
- Please download the pre-trained VFM and VLM models and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | DINOv2 | `dinov2_vitl14_pretrain.pth` |[download link](https://drive.google.com/file/d/1Rrl0RfU51eU8orbNVWHtNr1L3k5xhnld/view?usp=sharing)|
  | CLIP | `ViT-L-14-336px.pt` |[download link](https://drive.google.com/file/d/1s00ofvxn0NCVVgnycXd2wUx4Gs53O6mj/view?usp=sharing)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B.pt` |[download link](https://drive.google.com/file/d/1mQJ1zc_YLt7qAbaAET4-2EGNtIp2I6eB/view?usp=sharing)|
  | SIGLIP | `siglip_vitl16_384.pth` |[download link](https://drive.google.com/file/d/1PezEbwpqlasSYH2KPtU3aUD4hCzk9uE-/view?usp=sharing)|

## Checkpoints
- You can download **MFuser** model checkpoints and save them in `./work_dirs_d` folder. By default, all experiments below use DINOv2-L as the VFM. 

  | Model | Pretrained | Trained on | Config | Link |
  |-----|-----|-----|-----|:-----:|
  | `mfuser-clip-vit-l-city` | CLIP | Cityscapes | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_clip_vit-l_1e-4_20k-c2m-512.py) |[download link](https://drive.google.com/drive/folders/1M0AVa0f81ifm-Bi1gbIW6KDa5CN86S67?usp=sharing)|
  | `mfuser-clip-vit-l-gta` | CLIP | GTA5 | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_clip_vit-l_1e-4_20k-g2c-512.py) |[download link](https://drive.google.com/drive/folders/1eVVkFQqYf16vlDOdRHr6GBd7y8MfKM9D?usp=sharing)|
  | `mfuser-eva02-clip-vit-l-city` | EVA02-CLIP | Cityscapes | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_eva_vit-l_1e-4_20k-c2m-512.py) |[download link](https://drive.google.com/drive/folders/1pHzAY6RnAY37g7YQ2EywUWUz5DES-9aB?usp=sharing)|  
  | `mfuser-eva02-clip-vit-l-gta` | EVA02-CLIP | GTA5 | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_eva_vit-l_1e-4_20k-g2c-512.py) |[download link](https://drive.google.com/drive/folders/1D16a4cldw6iD1NV4a0rPZbEpKS51QFgs?usp=sharing)| 
  | `mfuser-siglip-vit-l-city` | SIGLIP | Cityscapes | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_siglip_vit-l_1e-4_20k-c2m-512.py) |[download link](https://drive.google.com/drive/folders/1Jgra4vENT0fIurXlCFvBvAxJnQFga6JJ?usp=sharing)| 
  | `mfuser-siglip-vit-l-gta` | SIGLIP | GTA5 | [config](https://github.com/devinxzhang/MFuser/blob/main/configs/mfuser/mfuser_siglip_vit-l_1e-4_20k-g2c-512.py) |[download link](https://drive.google.com/drive/folders/1Z2xCzSzmlp1QdM8ebENOR0UxHrgzNFIB?usp=sharing)| 

## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files](https://github.com/devinxzhang/MFuser/tree/main/configs/_base_/datasets) following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```

- The final folder structure should look like this:

```
MFuser
├── ...
├── pretrained
│   ├── dinov2_vitl14_pretrain.pth
│   ├── EVA02_CLIP_L_336_psz14_s6B.pt
│   ├── siglip_vitl16_384.pth
│   ├── ViT-L-14-336px.pt
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   |   ├── 10k
│   │   │   |    ├── train
│   │   │   |    ├── val
│   │   ├── labels
│   │   |   ├── sem_seg
│   │   |   |    ├── masks
│   │   │   |    |    ├── train
│   │   │   |    |    ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```

## Training
```
python train.py configs/[TRAIN_CONFIG]
```

## Evaluation
  Run the evaluation:
  ```
  python test.py configs/[TEST_CONFIG] work_dirs_d/[MODEL] --eval mIoU
  ```

## Citation
If you find our code helpful, please cite our paper:
```bibtex
@article{zhang2025mamba,
  title     = {Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation},
  author    = {Zhang, Xin and Robby T., Tan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
}
```

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [TLDR](https://github.com/ssssshwan/TLDR)
- [tqdm](https://github.com/ByeongHyunPak/tqdm)
- [MambaVision](https://github.com/NVlabs/MambaVision)
