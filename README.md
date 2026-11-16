# LF-DIDL

This repository is the official implementation of the paper "Diffusion-based Implicit Degradation Learning for Light Field Image Blind Super-Resolution" IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2026.

## Files

```text
LF-DIDL/
├── model/
│   ├── LF_DIDL.py              # LFnet and LF_DIDL wrapper
│   ├── encoder_SAE.py           # degradation representation encoders and denoiser
│   ├── common.py
│   └── diffusion/diffusion.py   # conditional diffusion
├── utils/                      # HDF5 loader, degradation and LF metrics
├── weights/
│   └── LF-DIDL_4xSR_epoch_100.tar
├── train.py
├── test.py
├── utility.py                  # optimizer and scheduler helpers
└── requirements.txt
```

## Pretrained weights

The pretrained weights for the 5 × 5 angular, 4× spatial SR main model are included in this folder at [`./weights/LF-DIDL_4xSR_epoch_100.tar`](./weights/LF-DIDL_4xSR_epoch_100.tar). This is the epoch-100 checkpoint and contains `epoch`, `state_dict`, `optimizer`, and `scheduler` entries. 

Files shared via Baidu Netdisk: **LF-DIDL_Pretrained Models**

Link: https://pan.baidu.com/s/1vTnTH0DpnArJqJtO9n0K6Q?pwd=jvmj

Extraction Code: jvmj
```



## Environment

Python, PyTorch with CUDA, and the packages in `requirements.txt` are required. Run commands from this directory. The model's original implementation creates its components with `.cuda()` and therefore requires a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Data

The data loader expects HDF5 files with `data` (low resolution) and `label` (high resolution) datasets. The arrays must represent 5 × 5 RGB light fields. Training files go directly in the training directory. The test directory contains one subdirectory per dataset, with HDF5 scenes inside:

```text
train_root/
  scene_001.h5
test_root/
  Dataset_A/scene_001.h5
  Dataset_B/scene_001.h5
```

The loader transposes each HDF5 array into `[U, V, C, H, W]`. Use the same data preparation and normalization as the original project to reproduce reported metrics.

## Evaluate

```bash
python test.py --testset_dir /path/to/test_root --device cuda:0
```

The test script loads the included weight with `strict=True` and reports PSNR/SSIM for each dataset and the mean across datasets. It splits each light field into overlapping patches before inference. High resolution labels are used only to compute metrics; the model generates the degradation condition from the low resolution input.

## Train

```bash
python train.py \
  --trainset_dir /path/to/train_root \
  --testset_dir /path/to/test_root \
  --device cuda:0
```

The training script preserves the main experiment's two stages: 50 epochs training the degradation encoder and SR network, followed by conditional diffusion training through epoch 100. It uses Adam, encoder learning rate `1e-4`, SR learning rate `2e-4`, and halves the learning rate every 15 epochs within each stage. New checkpoints are written under `runs/checkpoints/`.

To resume from a checkpoint, supply `--load_pretrain --model_path /path/to/checkpoint.tar --n_epochs N`, where `N` is larger than the checkpoint epoch. The included epoch-100 checkpoint is primarily for evaluation.

## Notes


Please contact us at yongkailing@163.com for questions.
