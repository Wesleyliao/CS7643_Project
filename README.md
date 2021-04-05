# Project

## Description

TBD


## Setup

Install Python 3.7

```
conda create -n DLProject python=3.7
conda activate DLProject
```

Install dependencies

```
pip install -r requirements.txt
```

If needed, install pytorch with CUDA
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

Install pre-commit hooks
```
pip install pre-commit
pre-commit install
pre-commit run
```

## Download data

### FFHQ Dataset
1. Download photos from [Google Drive](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
1. Save the downloaded archives in `/data/ffhq`
1. Run the extraction script:
```
./data/extract_ffhq.sh
```

### Danbooru 2019 Portraits
Run download script:
```
./data/download_danbooru2019portraits.sh
```