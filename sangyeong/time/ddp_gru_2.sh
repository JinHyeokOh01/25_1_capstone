#!/bin/bash
#SBATCH -J ddp_gru_2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH -o /data/lionsroar/logs/ddp_gru_2.out

# 호스트 정보 출력
echo "Running on host: $(hostname)"
echo "Host IP: $(hostname -i)"

export RANK=2
export WORLD_SIZE=4
export MASTER_ADDR=moana-y1
export MASTER_PORT=29500
export LOCAL_RANK=0

export NCCL_SOCKET_IFNAME=enp34s0   # 해당 노드에서 실제로 사용하는 네트워크 인터페이스
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_TIMEOUT=300  # 더 긴 타임아웃
export NCCL_P2P_DISABLE=1      # P2P 비활성화 (문제 해결용)
export NCCL_IB_DISABLE=1       # InfiniBand 비활성화

source /data/opt/anaconda3/etc/profile.d/conda.sh # conda 명령어 사용 가능하도록 init
conda activate pytorch1.12.1_p38


nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# 마스터 노드 준비 시간
echo "Master node waiting to initialize..."
sleep 20

python ca_ddp_gru.py

exit 0
