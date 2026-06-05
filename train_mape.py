import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. 回到 v2 版本的特征工程 ---
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

# 对特征依旧使用 StandardScaler
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(all_X)

# ★ 修复：只要数据变化重新训练，就立刻更新并保存最新的 scaler_X.pkl
joblib.dump(scaler_X, 'scaler_X.pkl')

# --- 核心改动：不再求对数平滑，而是只除以1e6缩放到0~10区间 —— 杜绝指数还原丢失峰值的问题 ---
Y_SCALE_FACTOR = 1000000.0  # 定义100万为一个单位，防止数值过大导致梯度爆炸
y_scaled = all_y / Y_SCALE_FACTOR

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_scaled, test_size=0.15, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=256, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=256, shuffle=False)

# 模型依旧使用深宽适中的 v2 结构
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus() # 保证输出压力绝对是正数，绝不会预测负压
        )
    def forward(self, x): return self.net(x)

model = ImprovedMLP(input_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 核心改动2：引入 MAPE Loss (平均绝对百分比误差) ---
# 既然我们看重的是“远点几十万预测百分比，近点数百万也看重百分比”，那就直接迫使模型优化相对百分比误差！
def mape_loss(pred, true):
    # 防止除0或极其微小的值
    return torch.mean(torch.abs((pred - true) / (true + 1e-5)))

epochs = 400
best_val_loss = float('inf')

print("开始采用 MAPE 损失和线性比例直接训练...")
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = mape_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 40 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                val_loss += mape_loss(model(batch_X), batch_y).item()
        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1:3d} | Val MAPE 百分比误差: {val_loss*100:5.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mlp_mape.pth')

print("\n--- 直接测试人工给定的那近/远组测试点 ---")
model.load_state_dict(torch.load('best_mlp_mape.pth'))
model.eval()

test_points = [
    {"x":0.0, "y":0.0, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 1886690.00},
    {"x":70.7107, "y":70.7107, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 144310.00},
    {"x":141.421, "y":141.421, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 114141.00},
    {"x":212.132, "y":212.132, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 106602.00},
]

for p in test_points:
    x, y, z, blast, b_height, height = p['x'], p['y'], p['z'], p['blast'], p['b_height'], p['height']
    R = np.sqrt(x**2 + y**2 + z**2)
    Z = R / (blast**(1/3)) if blast > 0 else R
    
    new_input = np.array([[x, y, z, blast, b_height, height, R, Z]])
    norm_new_input = scaler_X.transform(new_input)
    
    with torch.no_grad():
        tensor_new_input = torch.tensor(norm_new_input, dtype=torch.float32)
        # 模型输出的是 scaling 放平的值，这里乘回 1,000,000 直接得真值
        new_pred_real = (model(tensor_new_input).numpy() * Y_SCALE_FACTOR)[0][0]
        
    error = abs(new_pred_real - p['true_val']) / p['true_val'] * 100
    print(f"坐标: [{x:>6}, {y:>6}, {z:>6}], 真值: {p['true_val']:>10.2f}, 预测: {new_pred_real:>10.2f}, 误差: {error:>5.2f}%")


