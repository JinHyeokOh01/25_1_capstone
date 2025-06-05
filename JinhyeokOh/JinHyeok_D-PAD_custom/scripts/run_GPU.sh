#!/bin/bash
#SBATCH --job-name=DPAD_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0
#SBATCH -w moana-r4
#SBATCH -o logs/slurm-%j.out

cd /data/dhwlsgur795/repos/25_1_capstone

# 결과 로그 디렉토리 생성
mkdir -p logs/results/
mkdir -p exp/ETT_results/

echo "시작 시간: $(date)"
echo "실행 노드: $(hostname)"
echo "GPU 정보: $(nvidia-smi -L)"

for seq_len in 336
do
    for pred_len in 96 192 336
    do
        echo "=========================================================="
        echo "실행 설정: gpu_${seq_len}-${pred_len}"
        echo "시작 시간: $(date)"
        echo "=========================================================="
        
        python singlecard.py \
            --data gpu \
            --data_path gpu_1hour.csv \
            --root_path datasets/ \
            --features M \
            --target gpu_milli \
            --cols gpu_milli cpu_milli memory_mib num_gpu \
            --seq_len ${seq_len} \
            --pred_len ${pred_len} \
            --enc_hidden 256 \
            --dec_hidden 256 \
            --levels 2 \
            --lr 1e-4 \
            --dropout 0.1 \
            --batch_size 32 \
            --RIN 1 \
            --save True \
            --model_name DPAD_GCN_gpu_I${seq_len}_o${pred_len} \
            --model DPAD_GCN \
            --checkpoints exp/run_GPU_${seq_len}_${pred_len} \
            --itr 1
        
        # 결과 요약을 별도 파일에 저장
        echo "gpu_${seq_len}-${pred_len} 결과 $(date)" >> logs/results/summary.txt
        tail -n 10 logs/slurm-$SLURM_JOB_ID.out >> logs/results/summary.txt
        echo "" >> logs/results/summary.txt
        
        echo "완료 시간: $(date)"
        echo ""
    done
done

echo "모든 실험 완료: $(date)"
exit 0