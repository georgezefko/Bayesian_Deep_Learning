#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J position_encoding_1
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 10:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

module load python3/3.7.11
module load cuda/11.3
module load cudnn/v8.2.0.53-prod-cuda-11.3
module load ffmpeg/4.2.2
module load pandas/1.3.1-python-3.7.11
module load matplotlib/3.4.2-numpy-1.21.1-python-3.7.11

echo "Running script..."

pip3 install --user torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install --user scikit-image[optional] tqdm

python3 src/classification_encode.py Myanmar
echo "Completed."