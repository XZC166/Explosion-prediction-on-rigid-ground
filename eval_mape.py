import numpy as np
import torch
from torch import nn
import joblib

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

scaler_X = joblib.load('scaler_X.pkl')  # MAPE版本复用了最早的scaler_X
Y_SCALE_FACTOR = 1000000.0

model = ImprovedMLP(input_dim=8)
model.load_state_dict(torch.load('best_mlp_mape.pth'))
model.eval()

test_points = [
    {"x":0.0, "y":0.0, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 1886690.00},
    {"x":70.7107, "y":70.7107, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 144310.00},
    {"x":141.421, "y":141.421, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 114141.00},
    {"x":212.132, "y":212.132, "z":0.0, "blast":0.1, "b_height":30.0, "height":13.92, "true_val": 106602.00},
]

print("--- MAPE模型 最新测试效果 ---")
for p in test_points:
    x, y, z, blast, b_height, height = p['x'], p['y'], p['z'], p['blast'], p['b_height'], p['height']
    R = np.sqrt(x**2 + y**2 + z**2)
    Z = R / (blast**(1/3)) if blast > 0 else R
    
    new_input = np.array([[x, y, z, blast, b_height, height, R, Z]])
    norm_new_input = scaler_X.transform(new_input)
    
    with torch.no_grad():
        tensor_new_input = torch.tensor(norm_new_input, dtype=torch.float32)
        new_pred_real = (model(tensor_new_input).numpy() * Y_SCALE_FACTOR)[0][0]
        
    error = abs(new_pred_real - p['true_val']) / p['true_val'] * 100
    print(f"坐标: [{x:>6}, {y:>6}, {z:>6}], 真值: {p['true_val']:>10.2f}, 预测: {new_pred_real:>10.2f}, 误差: {error:>5.2f}%")
