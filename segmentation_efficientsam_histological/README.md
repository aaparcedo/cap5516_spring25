## Description
Parameter efficient finetuning for nuclei instance segmentation using LoRA and EfficientSAM.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/aaparcedo/cap5516_spring25.git
cd cap5516_spring25/segmentation_efficientsam_histological
git clone https://github.com/yformer/EfficientSAM.git
```

2. Create and activate a virtual environment:
```bash
conda create -n cap5516 python=3.10 -y
conda activate cap5516
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
NuInsSeg dataset from the paper NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Histological Images [https://arxiv.org/pdf/2308.01760].


```
mkdir NuInsSeg
cd NuInsSeg
wget https://zenodo.org/records/10518968/files/NuInsSeg.zip
unzip NuInsSeg
rm NuInsSeg.zip
```

## Experiments 
If running on HPC with Slurm (update the paths in slurm.sh):
```bash
sbatch slurm.sh
```

Otherwise (test results will print to stdout):
```bash
python main.py
```
