import pandas as pd
import numpy as np

# 1. 데이터 불러오기 (creation_time, deletion_time은 초 단위 상대값)
df = pd.read_csv('openb_pod_list_default.csv')

# 2. second 설정 (3600초 = 1시간)
step_sec = 30

# 3. 전체 시간 범위 계산 (시작은 내림, 끝은 올림)
start_time = (df['creation_time'].min() // step_sec) * step_sec
end_time   = int(np.ceil(df['deletion_time'].max() / step_sec)) * step_sec

# 4. 1시간 단위 타임스탬프 배열 생성
time_bins = np.arange(start_time, end_time + step_sec, step_sec)

# 5. 각 시간 구간 별 자원 사용량 집계
records = []
for t_start in time_bins[:-1]:
    t_end = t_start + step_sec
    active = df[(df['creation_time'] < t_end) & (df['deletion_time'] > t_start)]
    records.append({
        'time_sec'   : t_start,
        'gpu_milli'  : active['gpu_milli'].sum(),
        'cpu_milli'  : active['cpu_milli'].sum(),
        'memory_mib' : active['memory_mib'].sum(),
        'num_gpu'    : active['num_gpu'].sum()
    })

# 6. DataFrame 생성 및 저장
usage_df = pd.DataFrame.from_records(records)
usage_df.to_csv('gpu_1sec.csv', index=False)

print("✅ 30초 단위(초 기준) 시계열 데이터 생성 완료!")
print(usage_df.head())
