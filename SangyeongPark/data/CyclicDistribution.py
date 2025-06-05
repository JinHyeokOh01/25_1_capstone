import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("gpu_1min.csv")  # 위 데이터를 CSV로 저장했다면 파일명

# 주기 수
num_cycles = 4

# 라운드로빈 인덱스 추가
df['cycle'] = df.index % num_cycles

# 각 사이클로 나누기
cycles = [df[df['cycle'] == i].reset_index(drop=True) for i in range(num_cycles)]

# 저장
# for i, c in enumerate(cycles):
#     c.to_csv(f"cycle_{i}.csv", index=False)


# # 시각화 설정
metrics = ['gpu_milli', 'num_gpu']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# fig, axes = plt.subplots(2, 4, figsize=(16, 4), sharex=True)

# for row_idx, metric in enumerate(metrics):
#     for c in range(num_cycles):
#         ax = axes[row_idx, c]
#         ax.plot(
#             cycles[c]['time_sec'],
#             cycles[c][metric],
#             label=f'Cycle {c}',
#             color=colors[c],
#             marker='o',
#             linewidth=1,
#             markersize=0.1
#         )
#         ax.set_title(f"{metric} - Cycle {c}")
#         ax.grid(True)
#         if row_idx == 1:
#             ax.set_xlabel("Time (sec)")
#         ax.set_ylabel(metric)

# plt.suptitle("GPU 관련 지표의 사이클별 시계열 그래프", fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# --- 아래부터 상관계수 행렬 계산 및 출력 ---

for metric in metrics:
    # 각 사이클 데이터 가져오기
    data = pd.concat([cycles[i][metric].reset_index(drop=True) for i in range(num_cycles)], axis=1)
    data.columns = [f"Cycle_{i}" for i in range(num_cycles)]
    
    corr_matrix = data.corr()
    
    print(f"\n=== {metric} 사이클 간 상관계수 행렬 ===")
    print(corr_matrix)