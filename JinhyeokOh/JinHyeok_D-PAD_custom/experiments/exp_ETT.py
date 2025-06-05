import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings

from model.D_PAD_GAT import DPAD_GAT
from model.D_PAD_ATT import DPAD_ATT
from model.D_PAD_SEBlock import DPAD_SE
from model.D_PAD_adpGCN import DPAD_GCN

from utils.ETTh_metrics import metric, metric_
from utils.tools import EarlyStopping, adjust_learning_rate, load_model, save_model, visual
warnings.filterwarnings('ignore')
from data_process.ETT_data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from experiments.exp_basic import Exp_Basic


class Exp_ETT(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETT, self).__init__(args)
        # 디바이스 설정
        if isinstance(args.rank, str):
            self.device = torch.device(args.rank)
        else:
            self.device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() and args.rank >= 0 else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # 데이터 로더 초기화 시 target_idx 정보 저장
        temp_loader = self._get_data(flag='test')
        self.test_loader = temp_loader

    def _build_model(self):
        if self.args.features == 'S':
            self.input_dim = 1
        elif self.args.features == 'M':
            if self.args.data == 'gpu':
                self.input_dim = len(self.args.cols)
                print(f"GPU 데이터 - 사용 컬럼: {self.args.cols}")
                print(f"GPU 데이터 - 입력 차원: {self.input_dim}")
            elif "ETT" in self.args.data:
                self.input_dim = 7
            elif self.args.data == 'ECL' or self.args.data == 'electricity':
                self.input_dim = 321
            elif self.args.data == 'solar_AL':
                self.input_dim = 137
            elif self.args.data == 'exchange':
                self.input_dim = 8
            elif self.args.data == 'traffic':
                self.input_dim = 862
            elif self.args.data == 'weather':
                self.input_dim = 21
            elif self.args.data == 'illness':
                self.input_dim = 7
        else:
            raise ValueError("Invalid feature setting")

        print(f"Model input dimension: {self.input_dim}")

        if 'DPAD_GAT' in self.args.model_name:
            model = DPAD_GAT(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden=self.args.enc_hidden,
                dec_hidden=self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )
        elif 'DPAD_ATT' in self.args.model_name:
            model = DPAD_ATT(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden=self.args.enc_hidden,
                dec_hidden=self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN,
                num_heads=self.args.num_heads
            )
        elif 'DPAD_SE' in self.args.model_name:
            model = DPAD_SE(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden=self.args.enc_hidden,
                dec_hidden=self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )
        elif 'DPAD_GCN' in self.args.model_name:
            model = DPAD_GCN(
                output_len=self.args.pred_len,
                input_len=self.args.seq_len,
                input_dim=self.input_dim,
                enc_hidden=self.args.enc_hidden,
                dec_hidden=self.args.dec_hidden,
                num_levels=self.args.levels,
                dropout=self.args.dropout,
                K_IMP=self.args.K_IMP,
                RIN=self.args.RIN
            )
        else:
            raise ValueError("Unknown model name")

        # 모델을 적절한 디바이스로 이동
        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'electricity': Dataset_Custom,
            'Solar': Dataset_Custom,
            'traffic': Dataset_Custom,
            'exchange': Dataset_Custom,
            'illness': Dataset_Custom,
            'gpu': Dataset_Custom
        }
        Data = data_dict[self.args.data]

        shuffle_flag = flag != 'test'
        drop_last = True
        batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            cols=args.cols
        )
        if len(data_set) <= 0:
            raise ValueError(f"Dataset for '{flag}' split is empty. Check slicing or path.")

        print(f"{flag} dataset size: {len(data_set)}")
        
        # GPU 데이터의 경우 target 인덱스 저장
        if self.args.data == 'gpu' and hasattr(data_set, 'target_idx'):
            self.target_idx = data_set.target_idx
            print(f"Target column '{self.args.target}' index: {self.target_idx}")
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _select_criterion(self, losstype):
        if losstype == "mse":
            return nn.MSELoss()
        elif losstype == "mae":
            return nn.L1Loss()
        else:
            return nn.L1Loss()

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        valid_loader = self._get_data(flag='val')
        self.test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.0002)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count, train_loss = 0, []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            valid_loss, *_ = self.valid(valid_loader, criterion, flag="valid")
            test_loss, *_ = self.valid(self.test_loader, criterion, flag="test")

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        save_model(epoch, self.args.lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        return os.path.join(path, 'checkpoint.pth')

    def valid(self, valid_loader, criterion, flag):
        self.model.eval()
        total_loss, preds, trues = [], [], []

        with torch.no_grad():  # validation 시 gradient 계산 방지
            for batch_x, batch_y in valid_loader:
                pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                loss = criterion(pred, true)  # 이미 같은 디바이스에 있으므로 cpu() 불필요
                total_loss.append(loss.item())

        preds, trues = np.array(preds), np.array(trues)
        if preds.ndim == 4:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        elif preds.ndim == 3:
            preds = preds
            trues = trues
        else:
            preds = preds.reshape(preds.shape[0], -1, 1)
            trues = trues.reshape(trues.shape[0], -1, 1)

        mae, mse, rmse, mape, mspe, corr = metric_(preds, trues)
        print(f"-----------{flag} Results-----------")
        print(f"|  Normed  | mse:{mse:.4f} | mae:{mae:.4f} | rmse:{rmse:.4f} | mape:{mape:.4f} |")
        return np.average(total_loss), mae, mse, rmse, mape

    def test(self, setting, evaluate=0):
        self.model.eval()
        preds, trues = [], []

        if evaluate:
            model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        with torch.no_grad():  # test 시 gradient 계산 방지
            for batch_x, batch_y in self.test_loader:
                pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds, trues = np.array(preds), np.array(trues)
        if preds.ndim == 4:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        elif preds.ndim == 3:
            preds = preds
            trues = trues
        else:
            preds = preds.reshape(preds.shape[0], -1, 1)
            trues = trues.reshape(trues.shape[0], -1, 1)

        result_path = os.path.join('exp/ETT_results', setting)
        os.makedirs(result_path, exist_ok=True)

        mae, mse, rmse, mape, mspe, corr = metric_(preds, trues)
        print("Test Results:")
        print(f"|  Normed  | mse:{mse:.4f} | mae:{mae:.4f} | rmse:{rmse:.4f} | mape:{mape:.4f} | mspe:{mspe:.4f} | corr:{corr:.4f} |")

        if self.args.save:
            np.save(os.path.join(result_path, 'pred.npy'), preds)
            np.save(os.path.join(result_path, 'true.npy'), trues)
            if trues.ndim == 3:
                visual(trues[:, :, 0], preds[:, :, 0], os.path.join(result_path, 'result.pdf'))
            with open(os.path.join(result_path, 'metrics.txt'), 'w') as f:
                f.write(f'MSE: {mse:.4f}\n')
                f.write(f'MAE: {mae:.4f}\n')
                f.write(f'RMSE: {rmse:.4f}\n')
                f.write(f'MAPE: {mape:.4f}\n')
                f.write(f'MSPE: {mspe:.4f}\n')
                f.write(f'CORR: {corr:.4f}\n')

        return mse, mae, rmse, mape

    def _process_one_batch_DPAD(self, batch_x, batch_y):
        # 디버깅 정보 출력 (첫 번째 배치에서만)
        if not hasattr(self, '_debug_printed'):
            print("=== 디버깅 정보 ===")
            print(f"batch_x shape: {batch_x.shape}")  # [batch_size, seq_len, feature_dim]
            print(f"batch_y shape: {batch_y.shape}")  # [batch_size, label_len+pred_len, feature_dim]
            print(f"args.data: {self.args.data}")
            print(f"args.target: {self.args.target}")
            print(f"args.features: {self.args.features}")
            print(f"args.pred_len: {self.args.pred_len}")
            if hasattr(self, 'target_idx'):
                print(f"target_idx: {self.target_idx}")
            print("==================")
            self._debug_printed = True
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        
        # GPU 데이터의 경우 정확한 target 인덱스 사용
        if self.args.data == 'gpu':
            if hasattr(self, 'target_idx'):
                f_dim = self.target_idx
            else:
                # 기본값: gpu_milli는 데이터에서 첫 번째 컬럼 (컬럼 순서 확인 필요)
                f_dim = 0
                print(f"Warning: target_idx not found, using default index {f_dim}")
            
            # target 컬럼만 선택
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim+1].to(self.device)
            
            # 첫 번째 배치에서 타겟 값 확인
            if not hasattr(self, '_target_checked'):
                print(f"선택된 타겟 데이터 통계:")
                print(f"  shape: {batch_y.shape}")
                print(f"  min: {batch_y.min():.4f}, max: {batch_y.max():.4f}, mean: {batch_y.mean():.4f}")
                self._target_checked = True
        else:
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        
        return outputs, batch_y