import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

NUM_RUNS = 5                       
Y_SCALE_FACTOR = 1000000.0         
EPOCHS = 400                       
BATCH_SIZE = 64                    
LR = 0.001                         
DATA_FOLDER = 'grounddata_new/collect_pressure_peak'
OUTPUT_FOLDER = 'predictions'      

case_info_df = pd.read_csv('grounddata_new/case_info.csv')[['id', 'blast', 'bili_height', 'height']]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class ImprovedMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

def mape_loss(pred, true):
    return torch.mean(torch.abs((pred - true) / (true + 1e-5)))

for run_id in range(1, NUM_RUNS + 1):
    print(f"\n{'='*40}")
    print(f"          开始第 {run_id} 次独立实验          ")
    print(f"{'='*40}")
    
    SEED = 42 + run_id
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    all_case_data = [] 
    
    for filename in os.listdir(DATA_FOLDER):
        case_path = os.path.join(DATA_FOLDER, filename)
        case_row = case_info_df[case_info_df['id'] == filename]
        if case_row.empty: 
            continue
        
        blast = case_row.iloc[0]['blast']
        b_height = case_row.iloc[0]['bili_height']
        height = case_row.iloc[0]['height']
        
        try:
            points_data = np.loadtxt(case_path)
            if len(points_data.shape) == 1:
                points_data = points_data.reshape(1, -1)
        except Exception:
            continue
        
        case_X = []
        case_y = []
        
        for row in points_data:
            x, y, z, p = row[0], row[1], row[2], row[3] - 101325
            R = np.sqrt(x**2 + y**2 + z**2)
            Z = R / (blast**(1/3)) if blast > 0 else R
            log_Z = np.log(Z + 1e-5)
            inv_Z = 1.0 / (Z + 1e-5)
            case_X.append([x, y, z, blast, b_height, height, R, Z, log_Z, inv_Z])
            case_y.append(p)
        
        if len(case_X) > 0:
            all_case_data.append({
                'case_file': filename,
                'X': np.array(case_X),
                'y': np.array(case_y).reshape(-1, 1),
                'raw_points': points_data
            })

    n_total_files = len(all_case_data)
    n_test_files = max(1, int(n_total_files * 0.2))  
    
    shuffled_indices = np.random.permutation(n_total_files)
    test_indices = shuffled_indices[:n_test_files]
    train_indices = shuffled_indices[n_test_files:]
    
    train_cases = [all_case_data[i] for i in train_indices]
    test_cases = [all_case_data[i] for i in test_indices]
    for case in train_cases: case['split'] = 'train'
    for case in test_cases: case['split'] = 'test' 
    
    X_train = np.vstack([case['X'] for case in train_cases])
    y_train = np.vstack([case['y'] for case in train_cases])
    X_test = np.vstack([case['X'] for case in test_cases])
    y_test = np.vstack([case['y'] for case in test_cases])
    
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    joblib.dump(scaler_X, f'scaler_X_{run_id}.pkl')
    
    y_train_scaled = y_train / Y_SCALE_FACTOR
    y_test_scaled = y_test / Y_SCALE_FACTOR
    
    train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ImprovedMLP(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    model_save_path = f'model_{run_id}.pth'
    
    print(f"|-- 数据准备完毕：训练集算例 {len(train_cases)}个，总体测点 {len(X_train)}个 --|")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = mape_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    val_loss += mape_loss(model(batch_X), batch_y).item()
            val_loss /= len(test_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                
    print(f"-> 第 {run_id} 次训练结束，已保存权重文件: {model_save_path}")
    
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    run_output_dir = os.path.join(OUTPUT_FOLDER, f'run_{run_id}')
    train_dir = os.path.join(run_output_dir, 'train')
    test_dir = os.path.join(run_output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    with torch.no_grad():
        for case in all_case_data:
            case_name = case['case_file']
            case_X = case['X']
            raw_points = case['raw_points']  
            
            case_X_norm = scaler_X.transform(case_X)
            preds_scaled = model(torch.tensor(case_X_norm, dtype=torch.float32)).numpy()
            
            preds = (preds_scaled * Y_SCALE_FACTOR) + 101325
            
            target_dir = train_dir if case.get('split') == 'train' else test_dir
            output_filepath = os.path.join(target_dir, case_name)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for i in range(len(raw_points)):
                    x = raw_points[i, 0]
                    y = raw_points[i, 1]
                    z = raw_points[i, 2]
                    p_pred = preds[i, 0]
                    f.write(f"{x} {y} {z} {p_pred:.4f}\n")
                    
    print(f"-> 第 {run_id} 次实验彻底闭环！\n")
