# [CVPR 2025 Highlight] Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation



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



## Datasets
- To set up datasets, please follow [the official **TLDR** repo](https://github.com/ssssshwan/TLDR/tree/main?tab=readme-ov-file#setup-datasets).
- After downloading the datasets, edit the data folder root in [the dataset config files](https://github.com/ByeongHyunPak/tqdm/tree/main/configs/_base_/datasets) following your environment.
  
  ```python
  src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
  ```


## Citation
If you find our code helpful, please cite our paper:
```bibtex
@article{zhang2025mamba,
  title     = {Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation},
  author    = {Zhang, Xin and Robby T., Tan},
  journal   = {CVPR},
  year      = {2025}
}
```

## Acknowledgements
This project is based on the following open-source projects.
We thank the authors for sharing their codes.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DAFormer](https://github.com/lhoyer/DAFormer)
- [TLDR](https://github.com/ssssshwan/TLDR)
- [tqdm](https://github.com/ByeongHyunPak/tqdm)
- [MambaVision](https://github.com/NVlabs/MambaVision)