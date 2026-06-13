import os
import numpy as np
import pandas as pd

def calculate_mape_metrics(pred_dir, true_dir):
    """
    读取保存的预测结果和真实结果，计算整体和单算例的 MAPE
    """
    all_case_results = []
    
    total_true_abs = []
    total_pred_abs = []
    total_true_over = []
    total_pred_over = []
    
    # 遍历预测文件夹
    for filename in sorted(os.listdir(pred_dir)):
        if filename.startswith('.'):
            continue
            
        pred_path = os.path.join(pred_dir, filename)
        true_path = os.path.join(true_dir, filename)
        
        if not os.path.exists(true_path):
            continue
            
        # 1. 读取数据
        # 预测文件格式: x, y, z, p_mean(绝对压强), p_var
        pred_data = np.loadtxt(pred_path)
        if len(pred_data.shape) == 1:
            pred_data = pred_data.reshape(1, -1)
            
        # 真实文件格式: x, y, z, p_true(绝对压强)
        true_data = np.loadtxt(true_path)
        if len(true_data.shape) == 1:
            true_data = true_data.reshape(1, -1)
            
        # 2. 提取绝对压强 (Absolute Pressure)
        y_true_abs = true_data[:, 3]
        y_pred_abs = pred_data[:, 3] # 保存文件中的第4列是多次MC采样的均值
        
        # 3. 计算超压峰值 (Overpressure)
        y_true_over = y_true_abs
        y_pred_over = y_pred_abs
        
        # 4. 计算当前算例单文件的 MAPE
        # 绝对压强 MAPE
        case_mape_abs = np.mean(np.abs((y_pred_abs - y_true_abs) /( y_true_abs+ 1e-5))) * 100
        # 超压峰值 MAPE (加入 1e-5 稳定项防止远处低压测点分母溢出)
        case_mape_over = np.mean(np.abs((y_pred_over - y_true_over) / (y_true_over + 1e-5))) * 100
        
        all_case_results.append({
            '算例名称': filename,
            '测点数量': len(y_true_abs),
            '绝对压强 MAPE (%)': f"{case_mape_abs:.3f}%",
            '超压峰值 MAPE (%)': f"{case_mape_over:.3f}%"
        })
        
        # 收集用于计算全局总 MAPE 的数据
        total_true_abs.extend(y_true_abs)
        total_pred_abs.extend(y_pred_abs)
        total_true_over.extend(y_true_over)
        total_pred_over.extend(y_pred_over)
        
    # 5. 计算全测试集总体的全局 MAPE
    total_true_abs = np.array(total_true_abs)
    total_pred_abs = np.array(total_pred_abs)
    total_true_over = np.array(total_true_over)
    total_pred_over = np.array(total_pred_over)
    
    global_mape_abs = np.mean(np.abs((total_pred_abs - total_true_abs) / (total_true_abs+ 1e-5))) * 100
    global_mape_over = np.mean(np.abs((total_pred_over - total_true_over) / (total_true_over + 1e-5))) * 100
    
    # 6. 打印精美报告
    print("\n" + "="*50)
    print("          模型确定性预测精度评估 (MAPE)          ")
    print("="*50)
    print(f"测试集总计评估点数 : {len(total_true_abs)}")
    print(f"★ 全局绝对压强总 MAPE : {global_mape_abs:.3f}%")
    print(f"★ 全局超压峰值总 MAPE : {global_mape_over:.3f}%")
    print("="*50)
    
    # 转换为 DataFrame 方便展示
    df_report = pd.DataFrame(all_case_results)
    print("\n[各个算例详细误差清单]:")
    print(df_report.to_string(index=False))
    
    return df_report

if __name__ == '__main__':
    # 填入你的文件夹路径
    PRED_DIR = 'predictions_MC_arrival_time/run_2/test'  
    TRUE_DIR = 'grounddata_new/arrival_time'
    
    df_res = calculate_mape_metrics(PRED_DIR, TRUE_DIR)