## Description
Binary (normal/pneumonia) image classification of chest X-Ray with ResNet-18. 

## Installation
1. Clone the repository:
```bash
git clone https://github.com/aaparcedo/cap5516_spring25.git
cd cap5516_spring25
```

2. Create and activate a virtual environment:
```bash
conda create -n cap5516 python=3.10 -y
conda acitvate cap5516
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create logs folder
```bash
mkdir logs
```

## Dataset Assignment 1 (classification_pneumonia_xray)
Paths are set to default. Check paths are correct if an error occur.
1. Download dataset Chest X-Ray Images (Pneumonia) [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia]
```bash
python download.py
```
2. Augment dataset and split into 70/20/10 train/val/test split.
```bash
python balance_dataset.py
```

## Dataset Assignment 2 (segmentation_tumor_mri)
```bash
cd segmentation_tumor_mri
gdown --id 1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU
tar -xvf Task01_BrainTumour.tar
```

## Experiments 
Run from the root directory.

If running on HPC with Slurm (update the paths in slurm.sh):
```bash
sbatch classification_pneumonia_xray/slurm.sh
```
OR
```bash
sbatch segmentation_tumor_mri/slurm.sh
```

Otherwise (test results will print to stdout):
```bash
python classification_pneumonia_xray/main.py
```
OR
```bash
python segmentation_tumor_mri/main.py
```
