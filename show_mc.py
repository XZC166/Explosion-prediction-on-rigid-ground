import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题（可选，如果标签用英文可以注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(pred_dir, true_dir):
    """
    加载指定预测文件夹和真实文件夹中的对应数据
    """
    all_data = []
    
    # 遍历预测文件夹中的所有算例
    for filename in os.listdir(pred_dir):
        pred_path = os.path.join(pred_dir, filename)
        true_path = os.path.join(true_dir, filename)
        
        if not os.path.exists(true_path):
            print(f"警告: 找不到对应的真实数据文件 {filename}，已跳过。")
            continue
            
        # 读取预测数据 (x, y, z, mean, var)
        pred_data = np.loadtxt(pred_path)
        if len(pred_data.shape) == 1:
            pred_data = pred_data.reshape(1, -1)
            
        # 读取真实数据 (x, y, z, p_true) 
        # 顺便兼容你之前代码中减去大气压或保留原样的情况，假设最后一列为总压强 p
        true_data = np.loadtxt(true_path)
        if len(true_data.shape) == 1:
            true_data = true_data.reshape(1, -1)
            
        # 计算测点到爆源(0,0,0)的距离R
        x, y, z = pred_data[:, 0], pred_data[:, 1], pred_data[:, 2]
        R = np.sqrt(x**2 + y**2 + z**2)
        
        p_mean = pred_data[:, 3]
        p_var = pred_data[:, 4]
        p_std = np.sqrt(p_var)
        
        # 95% 置信区间上下界
        lower_bound = p_mean - 1.96 * p_std
        upper_bound = p_mean + 1.96 * p_std
        
        p_true = true_data[:, 3] # 假设真实数据的第四列是压强P
        
        case_df = pd.DataFrame({
            'R': R,
            'p_true': p_true,
            'p_mean': p_mean,
            'lower': lower_bound,
            'upper': upper_bound,
            'case_name': filename
        })
        all_data.append(case_df)
        
    return all_data

def calculate_metrics(df_all):
    """
    计算 PICP 和 MPIW / NMPIW
    """
    y_true = df_all['p_true'].values
    y_mean = df_all['p_mean'].values
    lower = df_all['lower'].values
    upper = df_all['upper'].values
    
    # 1. PICP: 真实值落入置信区间的比例
    in_interval = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(in_interval)
    
    # 2. MPIW: 平均区间宽度
    widths = upper - lower
    mpiw = np.mean(widths)
    
    # 3. NMPIW: 归一化区间宽度（用真实值的极差归一化，更具可比性）
    y_range = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / (y_range + 1e-5)
    
    print("\n" + "="*40)
    print("          不确定性量化指标评估          ")
    print("="*40)
    print(f"总评估测点数 : {len(df_all)}")
    print(f"PICP (95% 置信区间覆盖率) : {picp * 100:.2f}%")
    print(f"MPIW (平均预测区间宽度)   : {mpiw:.2f} s")
    print(f"NMPIW (归一化区间宽度)    : {nmpiw * 100:.2f}%")
    print("="*40 + "\n")

def plot_by_cases(all_cases_list, save_dir):
    """
    维度一：对各个算例单独绘图（横坐标为到爆源距离R）
    """
    case_plot_dir = os.path.join(save_dir, "individual_cases")
    os.makedirs(case_plot_dir, exist_ok=True)
    
    for df in all_cases_list:
        case_name = df['case_name'].iloc[0]
        # 按距离排序便于线图绘制
        df_sorted = df.sort_values(by='R')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['R'], df_sorted['p_true'], 'k_--', label='True', alpha=0.7)
        plt.plot(df_sorted['R'], df_sorted['p_mean'], 'r-', label='Pred', linewidth=2)
        plt.fill_between(df_sorted['R'], df_sorted['lower'], df_sorted['upper'], 
                         color='red', alpha=0.15, label='95% confidence interval')
        
#         plt.title(f"算例不确定性量化分析 - {case_name}")
        plt.xlabel('Distance (m)')
        plt.ylabel("Arrival time(s)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 替换掉文件名中的非法字符
        safe_filename = case_name.replace('.', '_') + "_uq.png"
        plt.savefig(os.path.join(case_plot_dir, safe_filename), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"-> 所有单算例可视化图片已保存至: {case_plot_dir}")

def plot_all_combined(df_all, save_dir):
    """
    维度二：对所有算例总体绘图（横坐标按真实超压幅值从小到大排序）
    """
    # 按照真实值从小到大排序
    df_sorted = df_all.sort_values(by='p_true').reset_index(drop=True)
    
    plt.figure(figsize=(15, 7))
    
    # 测点较多时，使用 scatter 和 fill_between 结合展现
    x_axis = np.arange(len(df_sorted))
    
    plt.fill_between(x_axis, df_sorted['lower'], df_sorted['upper'], 
                     color='orange', alpha=0.25, label='95% confidence interval')
    plt.plot(x_axis, df_sorted['p_mean'], 'g-', label='Pred', linewidth=1.5)
    plt.scatter(x_axis, df_sorted['p_true'], color='black', s=2, label='True', alpha=0.6)
    
#     plt.title("全体测点不确定性量化分析 (按超压幅值排序)")
    plt.xlabel("POI")
    plt.ylabel("Arrival time(s)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    save_path = os.path.join(save_dir, "all_cases_combined_uq.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> 全体算例聚合可视化图片已保存至: {save_path}")

if __name__ == '__main__':
    # ------------------ 在这里直接手动配置你的路径 ------------------
    PRED_DIR = 'predictions_MC_impulse/run_1/test'
    TRUE_DIR = 'grounddata_new/impulse'
    SAVE_DIR = 'uq_results_visual_impulse'
    # -------------------------------------------------------------
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. 加载数据
    print("正在读取并匹配预测与真实数据...")
    cases_list = load_data(PRED_DIR, TRUE_DIR)
    
    if not cases_list:
        print("未找到有效的数据对，请检查路径是否正确。")
    else:
        df_all_combined = pd.concat(cases_list, ignore_index=True)
        
        # 3. 计算指标
        calculate_metrics(df_all_combined)
        
        # 4. 绘图与保存
        print("正在绘制各个算例的距离分布不确定性图...")
        plot_by_cases(cases_list, SAVE_DIR)
        
        print("正在绘制全测点幅值排序不确定性图...")
        plot_all_combined(df_all_combined, SAVE_DIR)
        
        print("\n所有评估与可视化任务已顺利完成！")