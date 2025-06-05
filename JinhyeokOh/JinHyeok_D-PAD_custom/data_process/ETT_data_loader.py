import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='gpu_1hour.csv', 
                 target='gpu_milli', scale=True, inverse=False, cols=None):

        # 기본 시퀀스 길이 설정
        if size == None:
            self.seq_len = 336  # run_GPU.sh에서 사용하는 기본값
            self.label_len = 0   # singlecard.py에서 설정된 기본값
            self.pred_len = 96   # 예측 길이 기본값
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # 데이터셋 타입 설정
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        # GPU 데이터 기본 설정 수정
        if cols is None and data_path == 'gpu_1hour.csv':
            # GPU 데이터의 모든 feature 컬럼 사용 (time_sec 제외)
            self.cols = ['gpu_milli', 'cpu_milli', 'memory_mib', 'num_gpu']
        else:
            self.cols = cols
            
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        print(f"원본 데이터 컬럼: {df_raw.columns.tolist()}")
        print(f"데이터 형태: {df_raw.shape}")
        print(f"사용할 컬럼: {self.cols}")
        print(f"타겟 컬럼: {self.target}")
        
        # 데이터 분할 비율 계산
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        print(f"데이터 분할: train={num_train}, val={num_vali}, test={num_test}")
        
        # 경계 설정
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 특징 선택
        if self.features == 'M' or self.features == 'MS':
            # 멀티변량 처리
            if self.cols:
                # 지정된 컬럼들만 사용
                available_cols = [col for col in self.cols if col in df_raw.columns]
                if len(available_cols) != len(self.cols):
                    missing_cols = [col for col in self.cols if col not in df_raw.columns]
                    print(f"경고: 다음 컬럼들이 데이터에 없습니다: {missing_cols}")
                df_data = df_raw[available_cols]
                print(f"실제 사용 컬럼: {available_cols}")
            else:
                # time_sec 컬럼 제외한 모든 컬럼 사용
                cols_data = [col for col in df_raw.columns if col != 'time_sec']
                df_data = df_raw[cols_data]
                print(f"자동 선택된 컬럼: {cols_data}")
        elif self.features == 'S':
            # 단변량 처리 - 대상 컬럼만 사용
            if self.target not in df_raw.columns:
                raise ValueError(f"타겟 컬럼 '{self.target}'이 데이터에 없습니다.")
            df_data = df_raw[[self.target]]
            print(f"단변량 모드: {self.target} 컬럼만 사용")

        # 타겟 컬럼의 인덱스 저장 (나중에 _process_one_batch_DPAD에서 사용)
        if self.target in df_data.columns:
            self.target_idx = df_data.columns.get_loc(self.target)
            print(f"타겟 컬럼 '{self.target}'의 인덱스: {self.target_idx}")
        else:
            raise ValueError(f"타겟 컬럼 '{self.target}'이 선택된 feature 컬럼에 없습니다.")

        # 스케일링 적용
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            print(f"스케일링 적용됨. 훈련 데이터 범위: {border1s[0]}:{border2s[0]}")
        else:
            data = df_data.values
            print("스케일링 미적용")

        # 데이터 분할
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
            
        print(f"최종 데이터 형태 - X: {self.data_x.shape}, Y: {self.data_y.shape}")
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# 이전 버전과의 호환성 유지를 위한 별칭
Dataset_ETT_hour = Dataset_Custom
Dataset_ETT_minute = Dataset_Custom


# GPU 데이터셋에 최적화된 클래스
class Dataset_GPU(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None, 
                 features='M', data_path='gpu_1hour.csv', 
                 target='gpu_milli', scale=True, inverse=False, cols=None):
        
        # GPU 데이터셋 기본 설정 - 모든 feature 컬럼 사용
        if cols is None:
            cols = ['gpu_milli', 'cpu_milli', 'memory_mib', 'num_gpu']
        
        super().__init__(
            root_path=root_path,
            flag=flag,
            size=size,
            features=features,
            data_path=data_path,
            target=target,
            scale=scale,
            inverse=inverse,
            cols=cols
        )