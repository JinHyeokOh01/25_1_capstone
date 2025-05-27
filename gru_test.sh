#!/usr/bin/bash

#SBATCH -J GRU_30sec_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -t 1-0
#SBATCH -o /data/seois0408/logs/gru_30sec_test.out

source /data/opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/opt/anaconda3

pip install --no-cache-dir --force-reinstall numpy==1.26.4

echo "Python Path: $(which python)"

echo "[🔎 Python 버전]"
python --version

python -c "import numpy; print('[📦 NumPy 경로]', numpy.__file__)"

echo "[🔎 NumPy 버전]"
python -c "import numpy as np; print(np.__version__)"

echo "[🔎 TensorFlow 버전]"
python -c "import tensorflow as tf; print(tf.__version__)"

echo "[🔎 Pandas 버전]"
python -c "import pandas as pd; print(pd.__version__)"



python GRU_torch.py

exit 0
