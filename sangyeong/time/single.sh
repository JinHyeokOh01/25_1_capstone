#!/bin/bash
#SBATCH -J ddp_worker0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/tkddud386/logs/basic_gru.out

source /data/opt/anaconda3/etc/profile.d/conda.sh # conda 명령어 사용 가능하도록 init
conda activate pytorch1.12.1_p38

python time/basic_gru.py

exit 0
