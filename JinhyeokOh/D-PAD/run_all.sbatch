#!/bin/bash
#SBATCH --job-name=DPAD_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -w moana-r4
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

bash ./scripts/download_datasets.sh
bash ./scripts/run_ECL.sh
bash ./scripts/run_ETTh1.sh
bash ./scripts/run_ETTh2.sh
bash ./scripts/run_ETTm1.sh
bash ./scripts/run_ETTm2.sh
bash ./scripts/run_traffic.sh
bash ./scripts/run_weather.sh

exit 0
