# [CVPR 2025 Highlight] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation

### [**Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation**](https://arxiv.org/abs/2504.03193)
>[Xin Zhang](https://scholar.google.com/citations?user=nSqxFpAAAAAJ&hl=zh-CN)\, [Robby T. Tan](https://tanrobby.github.io/)\
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

## Pre-trained VFM&VLM Models
- Please download the pre-trained VFM and VLM models and save them in `./pretrained` folder.

  | Model | Type | Link |
  |-----|-----|:-----:|
  | DINOv2 | `dinov2_vitl14_pretrain.pth` |[download link](https://drive.google.com/file/d/1Rrl0RfU51eU8orbNVWHtNr1L3k5xhnld/view?usp=sharing)|
  | CLIP | `ViT-L-14-336px.pt` |[download link](https://drive.google.com/file/d/1s00ofvxn0NCVVgnycXd2wUx4Gs53O6mj/view?usp=sharing)|
  | EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B.pt` |[download link](https://drive.google.com/file/d/1mQJ1zc_YLt7qAbaAET4-2EGNtIp2I6eB/view?usp=sharing)|
  | SIGLIP | `siglip_vitl16_384.pth` |[download link](https://drive.google.com/file/d/1PezEbwpqlasSYH2KPtU3aUD4hCzk9uE-/view?usp=sharing)|

## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files](https://github.com/ByeongHyunPak/tqdm/tree/main/configs/_base_/datasets) following your environment.
  
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
  python test.py configs/[TEST_CONFIG] work_dirs/[MODEL] --eval mIoU
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
