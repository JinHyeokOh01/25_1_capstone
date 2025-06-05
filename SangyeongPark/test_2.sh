#!/bin/bash
#SBATCH -J single_gpu_gru
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -o /data/tkddud386/logs/single_gpu_gru.out

# 호스트 정보 출력
echo "Running on host: $(hostname)"
echo "Host IP: $(hostname -i)"

# NCCL 환경 변수 제거 (단일 GPU에서는 필요 없음)
# export NCCL_SOCKET_IFNAME=enp34s0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_TIMEOUT=300
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Conda 환경 활성화
source /data/opt/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.12.1_p38

# CUDA 상태 출력
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# 단일 GPU로 GRU 모델 학습 실행
python train_single_ddp.py

exit 0