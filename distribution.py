# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# # Load the original CSV
# file_path = "openb_pod_list_default.csv"
# df = pd.read_csv(file_path)

# # 1시간 단위로 집계 (같은 방식)
# step_sec = 3600
# start_time = (df['creation_time'].min() // step_sec) * step_sec
# end_time = int((df['deletion_time'].max() // step_sec + 1) * step_sec)
# time_bins = list(range(start_time, end_time + step_sec, step_sec))

# records = []
# for t_start in time_bins[:-1]:
#     t_end = t_start + step_sec
#     active = df[(df['creation_time'] < t_end) & (df['deletion_time'] > t_start)]
#     records.append({
#         'time_sec': t_start,
#         'gpu_milli': active['gpu_milli'].sum(),
#         'cpu_milli': active['cpu_milli'].sum(),
#         'memory_mib': active['memory_mib'].sum(),
#         'num_gpu': active['num_gpu'].sum()
#     })

# df_hourly = pd.DataFrame.from_records(records)

# # 데이터 4등분
# split_dfs = np.array_split(df_hourly, 4)

# # 저장
# for i, part in enumerate(split_dfs, start=1):
#     part.to_csv(f"gpu_dataset_part{i}.csv", index=False)

# # 확인용 분포 시각화

# plt.figure(figsize=(14, 6))
# for i, part in enumerate(split_dfs, start=1):
#     plt.plot(part['gpu_milli'].values, label=f'Part {i}')

# plt.title("분할된 4개 시계열 (gpu_milli)")
# plt.xlabel("시간 인덱스")
# plt.ylabel("GPU 사용량 (milli)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ----- 위는 데이터 분할 및 시각화 코드 -----
# ----- 아래는 전체 데이터셋을 시각화하는 코드 -----

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 원본 데이터 로드
file_path = "openb_pod_list_default.csv"
df = pd.read_csv(file_path)

# 1시간(3600초) 단위로 gpu_milli 합계 시계열 생성
step_sec = 3600
start_time = (df['creation_time'].min() // step_sec) * step_sec
end_time = int((df['deletion_time'].max() // step_sec + 1) * step_sec)
time_bins = list(range(start_time, end_time + step_sec, step_sec))

time_series = []

for t_start in time_bins[:-1]:
    t_end = t_start + step_sec
    active = df[(df['creation_time'] < t_end) & (df['deletion_time'] > t_start)]
    gpu_sum = active['gpu_milli'].sum()
    time_series.append((t_start, gpu_sum))

df_usage = pd.DataFrame(time_series, columns=['time_sec', 'gpu_milli'])

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(df_usage['time_sec'], df_usage['gpu_milli'], label='gpu_milli')
plt.title("GPU 사용량 (gpu_milli)의 시계열 분포")
plt.xlabel("Time (seconds)")
plt.ylabel("GPU Usage (milli)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
