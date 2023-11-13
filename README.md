## ClassWise-SAM-Adapter: Parameter Efficient Fine-tuning Adapts Segment Anything to SAR Domain for Semantic Segmentation

Xinyang Pu, Hecheng Jia, Linghao Zheng, Feng Wang, Feng Xu

The Key Laboratory of Information Science of Electromagnetic Waves, Fudan University, Shanghai, China


## Environment
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Quick Start
1. Prepare the dataset.
2. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
3. Training:
```bash

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py --config [CONFIG_PATH]

```


4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py --config [CONFIG_PATH]

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 train.py   --master_port='29600' --config [CONFIG_PATH]

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py --config configs_git/fusar-sar-map2-sam-vit-b-10cls-ce-trainval_1024_lr2e4_CE_e200.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 train.py --config configs_git/fusar-sar-map-sam-vit-b-5cls-ce-trainval_1024_lr2e4_CEv2_e200_ignore_bg.yaml

```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]

export CUDA_VISIBLE_DEVICES=2 
python test.py --config configs/fusar-sar-map-sam-vit-b-5cls-ce-trainval_1024_lr2e4_CEv2_e200_ignore_bg.yaml --model ./save/fusar-sar-map-sam-vit-b-5cls-ce-trainval_1024_lr2e4_CEv2_e200_ignore_bg/model_epoch_best.pth


```
## Citation

If you find our work useful in your research, please consider citing:


## Acknowledgements
The part of the code is derived from SAM-adapter: <a href='https://www.kokoni3d.com/'> KOKONI, Moxin Technology (Huzhou) Co., LTD </a>, Zhejiang University, Singapore University of Technology and Design, Huzhou University, Beihang University. <a href='https://tianrun-chen.github.io/SAM-Adaptor/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 