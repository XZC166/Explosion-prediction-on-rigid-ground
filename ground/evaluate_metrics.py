import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

TRUE_DIR = 'grounddata_new/collect_pressure_peak_merged'
PRED_BASE_DIR = 'predictions'

def evaluate_directory(true_dir, pred_dir):
    if not os.path.exists(pred_dir):
        return None
    
    y_true_abs_all = []
    y_pred_abs_all = []
    y_true_over_all = []
    y_pred_over_all = []
    
    for filename in os.listdir(pred_dir):
        true_path = os.path.join(true_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(true_path):
            continue
            
        try:
            # Load data
            true_data = np.loadtxt(true_path)
            pred_data = np.loadtxt(pred_path)
            
            # Ensure 2D (in case of single point case)
            if len(true_data.shape) == 1: true_data = true_data.reshape(1, -1)
            if len(pred_data.shape) == 1: pred_data = pred_data.reshape(1, -1)
            
            # Extract pressure column (4th column, index 3)
            true_p = true_data[:, 3]
            pred_p = pred_data[:, 3]
            
            y_true_abs_all.extend(true_p)
            y_pred_abs_all.extend(pred_p)
            
            # Overpressure (subtract environment pressure 101325)
            y_true_over_all.extend(true_p - 101325)
            y_pred_over_all.extend(pred_p - 101325)
            
        except Exception as e:
            pass
            
    if len(y_true_abs_all) == 0:
        return None
        
    y_true_abs_all = np.array(y_true_abs_all)
    y_pred_abs_all = np.array(y_pred_abs_all)
    y_true_over_all = np.array(y_true_over_all)
    y_pred_over_all = np.array(y_pred_over_all)
    
    # Calculate Overpressure MAPE (Closest to model training loss concept)
    mape_over = np.mean(np.abs((y_true_over_all - y_pred_over_all) / (np.abs(y_true_over_all) + 1e-5))) * 100
    
    # Calculate Absolute Pressure MAPE
    mape_abs = np.mean(np.abs((y_true_abs_all - y_pred_abs_all) / (y_true_abs_all + 1e-5))) * 100
    
    rmse_over = np.sqrt(mean_squared_error(y_true_over_all, y_pred_over_all))
    r2 = r2_score(y_true_over_all, y_pred_over_all)
    
    return {
        'mape_over': mape_over,
        'mape_abs': mape_abs,
        'rmse': rmse_over,
        'r2': r2,
        'samples': len(y_true_abs_all)
    }

print(f"{'='*85}")
print(f"{'Run':^5} | {'Split':^6} | {'Samples':^8} | {'超压MAPE(%)':^14} | {'绝对压MAPE(%)':^14} | {'R2 Score':^10} | {'RMSE (Pa)':^10}")
print(f"{'-'*85}")

run_train_mapes = []
run_test_mapes = []
run_test_r2s = []

for run_id in range(1, 6):
    run_dir = os.path.join(PRED_BASE_DIR, f'run_{run_id}')
    train_dir = os.path.join(run_dir, 'train')
    test_dir = os.path.join(run_dir, 'test')
    
    train_res = evaluate_directory(TRUE_DIR, train_dir)
    test_res = evaluate_directory(TRUE_DIR, test_dir)
    
    if train_res:
        print(f"{run_id:^5} | {'Train':^6} | {train_res['samples']:^8} | {train_res['mape_over']:^14.2f} | {train_res['mape_abs']:^14.2f} | {train_res['r2']:^10.4f} | {train_res['rmse']:^10.1f}")
        run_train_mapes.append(train_res['mape_over'])
    if test_res:
        print(f"{run_id:^5} | {'Test':^6} | {test_res['samples']:^8} | {test_res['mape_over']:^14.2f} | {test_res['mape_abs']:^14.2f} | {test_res['r2']:^10.4f} | {test_res['rmse']:^10.1f}")
        run_test_mapes.append(test_res['mape_over'])
        run_test_r2s.append(test_res['r2'])

print(f"{'-'*85}")
train_avg = np.mean(run_train_mapes) if run_train_mapes else 0
test_avg = np.mean(run_test_mapes) if run_test_mapes else 0
r2_avg = np.mean(run_test_r2s) if run_test_r2s else 0
print(f"5次交叉验证平均 => 训练集超压MAPE: {train_avg:.2f}% | 测试集超压MAPE: {test_avg:.2f}% | 测试集R2: {r2_avg:.4f}")
print(f"{'='*85}")
