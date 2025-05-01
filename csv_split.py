import pandas as pd
import numpy as np

# 1. 원본 CSV 불러오기
df = pd.read_csv('gpu_1hour.csv')

# 2. DataFrame을 4개로 균등 분할
chunks = np.array_split(df, 4)

# 3. 각 청크를 파일로 저장
for idx, chunk in enumerate(chunks, start=1):
    filename = f'gpu_dataset_part{idx}.csv'
    chunk.to_csv(filename, index=False)
    print(f"✅ 저장 완료: {filename}")
