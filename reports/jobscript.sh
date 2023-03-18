#!/bin/sh
#BSUB -q gpuv100
#BSUB -n 2
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -J MapiliarySTN
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

module load python3/3.8.11
#module load cuda/11.3
#module load cudnn/v8.2.0.53-prod-cuda-11.3
#module load ffmpeg/4.2.2
#module load pandas

echo "Running script..."

#pip3 install --user torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip3 install --user scikit-image[optional] tqdm
pip3 install -r requirements.txt
#python3 notebooks/notebooks/Laplace_subnetwork.py -m True -pr True -lo True -sa False
for i {1..5}
do
    python3 src/models/execute_model.py
done
echo "Completed."
