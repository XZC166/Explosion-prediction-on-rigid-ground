import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# 1. 重新读取数据并按之前完全相同的方式划分（复刻测试集）
case_info_df = pd.read_csv('data/case_info.csv')[['id', 'blast', 'bili_height', 'height']]
data_folder = 'data/collect_pressure_peak'
all_X, all_y = [], []

for filename in os.listdir(data_folder):
    case_path = os.path.join(data_folder, filename)
    case_row = case_info_df[case_info_df['id'] == filename]
    if case_row.empty: continue
    blast, b_height, height = case_row.iloc[0]['blast'], case_row.iloc[0]['bili_height'], case_row.iloc[0]['height']
    try: points_data = np.loadtxt(case_path)
    except: continue
        
    for row in points_data:
        x, y, z, p = row[0], row[1], row[2], row[3]
        R = np.sqrt(x**2 + y**2 + z**2)
        Z = R / (blast**(1/3)) if blast > 0 else R
        all_X.append([x, y, z, blast, b_height, height, R, Z])
        all_y.append(p)

all_X = np.array(all_X)
all_y = np.array(all_y).reshape(-1, 1)

# 加载归一化器处理 X
scaler_X = joblib.load('scaler_X.pkl')
X_norm = scaler_X.transform(all_X)

# 确保这里的 random_state=42 和 train_mape.py 保持绝对一致，以得到原汁原味的测试集
X_train, X_test, y_train, y_test = train_test_split(X_norm, all_y, test_size=0.15, random_state=42)

# 2. 构造对应的模型并加载参数
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    def forward(self, x): return self.net(x)

model = ImprovedMLP(input_dim=8)
model.load_state_dict(torch.load('best_mlp_mape.pth'))
model.eval()

Y_SCALE_FACTOR = 1000000.0

# 3. 预测并计算 R方
with torch.no_grad():
    # 测试集
    preds_test_scaled = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    preds_test = preds_test_scaled * Y_SCALE_FACTOR  # 乘回 100 万回到真实物理值
    
    # 训练集
    preds_train_scaled = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
    preds_train = preds_train_scaled * Y_SCALE_FACTOR

r2_test = r2_score(y_test, preds_test)
r2_train = r2_score(y_train, preds_train)

print(f"--- R² (决定系数) 评估结果 ---")
print(f"模型在 [测试集] 上的 R² 得分为: {r2_test:.4f}")
print(f"模型在 [训练集] 上的 R² 得分为: {r2_train:.4f}")
