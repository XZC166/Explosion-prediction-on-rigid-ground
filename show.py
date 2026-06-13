import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')
# 设置全局字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式也使用类似字体
plt.rcParams['font.size'] = 16  # 默认字体大小
def calculate_metrics(y_true, y_pred):
    """
    计算回归指标
    """
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'MAE': np.nan, 'MAPE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
    
    # MAE
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # MAPE (避免除零)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10))) * 100
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # R2
    if len(y_true_clean) >= 2:
        r2 = r2_score(y_true_clean, y_pred_clean)
    else:
        r2 = np.nan
    
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}
def plot_predictions_scatter(y_true, y_pred, save_path=None, title=None, figsize=(8, 8)):
    """
    绘制预测值与真实值的散点图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径（如果为None则显示图像）
    title: 图表标题
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]/1000
    y_pred_clean = y_pred[mask]/1000
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    
    # 确定坐标轴范围
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    min_val = min(y_true_clean.min(), y_pred_clean.min())
    # 添加一点边距
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    plt.scatter(y_true_clean, y_pred_clean, alpha=0.5, s=20, c='blue', edgecolors='none')
    
    # 绘制理想线 (y=x)
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r', linewidth=2, label='Perfect Prediction (y=x)')
    # 绘制±30%误差线
    # +30%: y = 1.3 * x
    plt.plot([plot_min, plot_max], [plot_min * 1.3, plot_max * 1.3], 'r--', linewidth=1.5, alpha=0.8, label='+30% Error')
    # -30%: y = 0.7 * x
    plt.plot([plot_min, plot_max], [plot_min * 0.7, plot_max * 0.7], 'r--', linewidth=1.5, alpha=0.8, label='-30% Error')
    
    # 可选：填充误差区域
    plt.fill_between([plot_min, plot_max], 
                     [plot_min * 0.7, plot_max * 0.7], 
                     [plot_min * 1.3, plot_max * 1.3], 
                     alpha=0.1, color='red', label='±30% Error Band')
        
    # 设置坐标轴
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel('True(kPa)', fontsize=16)
    plt.ylabel('Pred(kPa)', fontsize=16)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=16)
    
    # 添加对角线上的点密度指示
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
def plot_mape_distribution_by_magnitude(y_true, y_pred, save_path=None, figsize=(12, 6)):
    """
    绘制不同压力值范围内MAPE分布占比的分组柱状图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径（如果为None则显示图像）
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 计算MAPE
    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    
    # 定义压力值范围（Pa）
    pressure_bins = [0, 1000, 10000, 100000, 1000000, np.inf]
    pressure_labels = ['0-1 kPa', '1-10 kPa', '10-100 kPa', '100-1000 kPa', '>1000 kPa']
    
    # 定义MAPE范围（横轴）
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
    
    # 为每个压力范围统计MAPE分布
    distribution_data = []
    pressure_counts = []
    
    for i in range(len(pressure_bins) - 1):
        # 获取当前压力范围内的样本
        pressure_mask = (y_true_clean >= pressure_bins[i]) & (y_true_clean < pressure_bins[i+1])
        mape_in_range = mape[pressure_mask]
        count_in_range = np.sum(pressure_mask)
        pressure_counts.append(count_in_range)
        
        if count_in_range > 0:
            # 统计MAPE分布
            hist, _ = np.histogram(mape_in_range, bins=mape_bins)
            percentages = hist / count_in_range * 100
            distribution_data.append(percentages)
        else:
            distribution_data.append(np.zeros(len(mape_labels)))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图位置
    x = np.arange(len(mape_labels))
    width = 0.15  # 柱子宽度
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    # 绘制分组柱状图
    bars_list = []
    for i, (pressure_label, color) in enumerate(zip(pressure_labels, colors)):
        percentages = distribution_data[i]
        bars = ax.bar(x + i * width, percentages, width, 
                     label=pressure_label, color=color, 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        bars_list.append(bars)
        
        # 在柱子上方添加百分比标签（只添加大于5%的）
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            if pct > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 设置标签和标题
    ax.set_xlabel('MAPE Range', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=16)
    ax.set_title('MAPE Distribution by Pressure Magnitude', fontsize=16)
    ax.set_xticks(x + width * (len(pressure_labels) - 1) / 2)
    ax.set_xticklabels(mape_labels)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=16, title='Pressure Range')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAPE分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'pressure_ranges': pressure_labels,
        'pressure_counts': pressure_counts,
        'mape_ranges': mape_labels,
        'distribution': distribution_data
    }

def plot_mape_distribution_by_magnitude2(y_true, y_pred, save_path=None, figsize=(12, 6)):
    """
    绘制不同压力值范围内MAPE分布占比的堆叠柱状图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径（如果为None则显示图像）
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 计算MAPE
    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    
    # 定义压力值范围（Pa）
    pressure_bins = [0, 1000, 10000, 100000, 1000000,np.inf]
    pressure_labels = ['0-1000 Pa', '1000-10 kPa', '10-100 kPa', '100-1000 kPa', '>1000 kPa']
    
    # 定义MAPE范围
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
    
    # 为每个压力范围统计MAPE分布
    distribution_data = []
    pressure_counts = []
    
    for i in range(len(pressure_bins) - 1):
        # 获取当前压力范围内的样本
        pressure_mask = (y_true_clean >= pressure_bins[i]) & (y_true_clean < pressure_bins[i+1])
        mape_in_range = mape[pressure_mask]
        count_in_range = np.sum(pressure_mask)
        pressure_counts.append(count_in_range)
        
        if count_in_range > 0:
            # 统计MAPE分布
            hist, _ = np.histogram(mape_in_range, bins=mape_bins)
            percentages = hist / count_in_range * 100
            distribution_data.append(percentages)
        else:
            distribution_data.append(np.zeros(len(mape_labels)))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制堆叠柱状图
    bottom = np.zeros(len(pressure_labels))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    for i, (label, color) in enumerate(zip(mape_labels, colors)):
        percentages = [data[i] for data in distribution_data]
        bars = ax.bar(pressure_labels, percentages, bottom=bottom, 
                     label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += np.array(percentages)
    
    # 在每个柱子上添加样本数标签
    for i, (count, total_height) in enumerate(zip(pressure_counts, bottom)):
        if count > 0:
            ax.text(i, total_height + 1, f'n={count}', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('True Pressure Range', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=16)
    ax.set_title('MAPE Distribution by Pressure Magnitude', fontsize=16)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=16, title='MAPE Range')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAPE分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'pressure_ranges': pressure_labels,
        'pressure_counts': pressure_counts,
        'mape_ranges': mape_labels,
        'distribution': distribution_data
    }

def plot_prediction_magnitude_comparison(y_true, y_pred, save_path=None, figsize=(12, 6)):
    """
    绘制真实值和预测值在5个数量级范围的对比柱状图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask_true = ~(np.isnan(y_true))
    mask_pred = ~(np.isnan(y_pred))
    y_true_clean = y_true[mask_true]
    y_pred_clean = y_pred[mask_pred]
    
    if len(y_true_clean) == 0 or len(y_pred_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 定义数量级范围（Pa）
    bins = [0, 100, 1000, 10000, 100000, 1000000, np.inf]
    labels = ['0-100 Pa', '100-1000 Pa', '1000-10 kPa', '10-100 kPa', '100-1000 kPa', '>1000 kPa']
    
    # 统计真实值和预测值的分布
    counts_true, _ = np.histogram(y_true_clean, bins=bins)
    counts_pred, _ = np.histogram(y_pred_clean, bins=bins)
    
    percentages_true = counts_true / len(y_true_clean) * 100
    percentages_pred = counts_pred / len(y_pred_clean) * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置柱状图位置
    x = np.arange(len(labels))
    width = 0.35
    
    # 绘制分组柱状图
    bars1 = ax.bar(x - width/2, percentages_true, width, label='True Values', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, percentages_pred, width, label='Predicted Values',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加标签
    ax.set_xlabel('OverPressure Range', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left',fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 在柱子上添加数值标签
    for bars, percentages in zip([bars1, bars2], [percentages_true, percentages_pred]):
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比柱状图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'ranges': labels,
        'true_counts': counts_true,
        'pred_counts': counts_pred,
        'true_percentages': percentages_true,
        'pred_percentages': percentages_pred
    }
def plot_mape_distribution(y_true, y_pred, save_path=None, figsize=(12, 6)):
    """
    绘制MAPE误差分布占比的饼图/柱状图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径（如果为None则显示图像）
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 计算MAPE
    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    
    # 定义MAPE范围（误差区间）
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
    
    # 统计MAPE分布
    hist, _ = np.histogram(mape, bins=mape_bins)
    percentages = hist / len(mape) * 100
    
    # 创建图形 - 柱状图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置颜色
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    # 绘制柱状图
    bars = ax.bar(mape_labels, percentages, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # 在柱子上方添加百分比标签
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('MAPE Range', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=16)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAPE分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'mape_ranges': mape_labels,
        'counts': hist,
        'percentages': percentages,
        'total_samples': len(mape)
    }


def plot_mape_distribution_pie(y_true, y_pred, save_path=None, figsize=(10, 8)):
    """
    绘制MAPE误差分布占比的饼图
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径（如果为None则显示图像）
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 计算MAPE
    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    
    # 定义MAPE范围（误差区间）
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
    
    # 统计MAPE分布
    hist, _ = np.histogram(mape, bins=mape_bins)
    percentages = hist / len(mape) * 100
    
    # 过滤掉占比为0的类别
    filtered_labels = []
    filtered_percentages = []
    filtered_colors = []
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    for label, pct, color in zip(mape_labels, percentages, colors):
        if pct > 0:
            filtered_labels.append(label)
            filtered_percentages.append(pct)
            filtered_colors.append(color)
    
    # 创建饼图
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(filtered_percentages, 
                                        labels=filtered_labels,
                                        colors=filtered_colors,
                                        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                                        startangle=90,
                                        explode=[0.02] * len(filtered_labels),
                                        shadow=True,
                                        textprops={'fontsize': 12})
    
    # 设置百分比文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # 设置标题
    ax.set_title(f'MAPE Error Distribution\n(Total samples: {len(mape)})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(wedges, filtered_labels, title="MAPE Range",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAPE饼图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'mape_ranges': filtered_labels,
        'counts': hist[hist > 0],
        'percentages': filtered_percentages,
        'total_samples': len(mape)
    }


def plot_error_statistics(y_true, y_pred, save_path=None, figsize=(12, 6)):
    """
    绘制误差统计汇总图（包含误差分布和统计指标）
    
    参数:
    y_true: 真实值数组
    y_pred: 预测值数组
    save_path: 保存路径
    figsize: 图表大小
    """
    
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("错误: 没有有效数据点")
        return
    
    # 计算MAPE
    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    
    # 计算统计指标
    mean_mape = np.mean(mape)
    median_mape = np.median(mape)
    std_mape = np.std(mape)
    max_mape = np.max(mape)
    min_mape = np.min(mape)
    
    # 定义MAPE范围
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']
    
    # 统计MAPE分布
    hist, _ = np.histogram(mape, bins=mape_bins)
    percentages = hist / len(mape) * 100
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 左图：柱状图
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    bars = ax1.bar(mape_labels, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加百分比标签
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('MAPE Range', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('MAPE Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 右图：统计指标文本框
    ax2.axis('off')
    
    # 准备统计文本
    stats_text = f"""
    Error Statistics Summary
    {'='*30}
    
    Total Samples: {len(mape)}
    
    Mean MAPE:     {mean_mape:.2f}%
    Median MAPE:   {median_mape:.2f}%
    Std MAPE:      {std_mape:.2f}%
    
    Min MAPE:      {min_mape:.2f}%
    Max MAPE:      {max_mape:.2f}%
    
    {'='*30}
    Performance Assessment:
    
    {'✓ Excellent' if mean_mape < 10 else '○ Good' if mean_mape < 20 else '✗ Poor'}
    """
    
    # 显示文本框
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
             fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontfamily='monospace')
    
    plt.suptitle('MAPE Error Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差统计图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'mape_ranges': mape_labels,
        'counts': hist,
        'percentages': percentages,
        'total_samples': len(mape),
        'mean_mape': mean_mape,
        'median_mape': median_mape,
        'std_mape': std_mape,
        'min_mape': min_mape,
        'max_mape': max_mape
    }
def plot_single_case_distribution_from_data(true_data, pred_data, save_path=None, 
                                             subtract_atm=True, atm_pressure=101325, figsize=(20, 5)):
    """
    绘制单个算例的真实值、预测值和误差在x,y平面上的分布图
    
    参数:
    true_data: 真实数据数组，格式为 [x, y, z, p] 或 [x, y, p]
    pred_data: 预测数据数组，格式为 [x, y, z, p] 或 [x, y, p]
    save_path: 保存路径（如果为None则显示图像）
    subtract_atm: 是否减去大气压（默认True）
    atm_pressure: 大气压值（默认101325 Pa）
    case_name: 算例名称（用于标题，如果为None则使用默认名称）
    figsize: 图表大小
    """
    
    try:
        # 确保数据是二维数组
        if true_data.ndim == 1:
            true_data = true_data.reshape(1, -1)
        if pred_data.ndim == 1:
            pred_data = pred_data.reshape(1, -1)
        
        # 提取坐标（前两列或前三列中的x,y）
        if true_data.shape[1] >= 3:
            x_coords = true_data[:, 0]
            y_coords = true_data[:, 1]
        else:
            print(f"错误: 数据格式不正确，列数={true_data.shape[1]}，需要至少2列坐标")
            return None
        
        # 提取压力列（最后一列）
        if true_data.shape[1] >= 4:
            if subtract_atm:
                y_true = true_data[:, 3] - atm_pressure
            else:
                y_true = true_data[:, 3]
        elif true_data.shape[1] >= 3:
            if subtract_atm:
                y_true = true_data[:, 2] - atm_pressure
            else:
                y_true = true_data[:, 2]
        else:
            print(f"错误: 数据格式不正确，列数={true_data.shape[1]}，需要至少3列")
            return None
        
        if pred_data.shape[1] >= 4:
            if subtract_atm:
                y_pred = pred_data[:, 3] - atm_pressure
            else:
                y_pred = pred_data[:, 3]
        elif pred_data.shape[1] >= 3:
            if subtract_atm:
                y_pred = pred_data[:, 2] - atm_pressure
            else:
                y_pred = pred_data[:, 2]
        else:
            print(f"错误: 预测数据格式不正确，列数={pred_data.shape[1]}，需要至少3列")
            return None
        
        # 确保样本数一致
        n_samples = min(len(y_true), len(y_pred))
        if len(y_true) != len(y_pred):
            print(f"警告: 样本数不一致 - 真实:{len(y_true)}, 预测:{len(y_pred)}，将截断到 {n_samples}")
            y_true = y_true[:n_samples]/1000
            y_pred = y_pred[:n_samples]/1000
            x_coords = x_coords[:n_samples]
            y_coords = y_coords[:n_samples]
        
        # 计算误差
        y_error = abs(y_pred - y_true)/(y_true+ 1e-10)*100
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 确定压力值的共同范围
        p_min = min(y_true.min(), y_pred.min())
        p_max = max(y_true.max(), y_pred.max())
        error_abs_max = max(abs(y_error.min()), abs(y_error.max()))
        
        # 1. 真实值子图
        im1 = axes[0].scatter(x_coords, y_coords, c=y_true, cmap='jet', 
                              norm=Normalize(vmin=p_min, vmax=p_max), 
                              s=100, alpha=0.8, linewidth=0.5)
        axes[0].set_title('True', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Coordinate (m)', fontsize=14)
        axes[0].set_ylabel('Y Coordinate (m)', fontsize=14)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_aspect('equal')
        axes[0].tick_params(labelsize=14)
        
        # 2. 预测值子图
        im2 = axes[1].scatter(x_coords, y_coords, c=y_pred, cmap='jet', 
                              norm=Normalize(vmin=p_min, vmax=p_max), 
                              s=100, alpha=0.8, linewidth=0.5)
        axes[1].set_title('Pred', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X Coordinate (m)', fontsize=14)
        # axes[1].set_ylabel('Y Coordinate (m)', fontsize=14)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_aspect('equal')
        axes[1].tick_params(labelsize=14)
        
        # 3. 误差子图
        im3 = axes[2].scatter(x_coords, y_coords, c=y_error, cmap='RdBu_r', 
                              norm=Normalize(vmin=-error_abs_max, vmax=error_abs_max), 
                              s=100, alpha=0.8, linewidth=0.5)
        axes[2].set_title('Error', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('X Coordinate (m)', fontsize=14)
        # axes[2].set_ylabel('Y Coordinate (m)', fontsize=14)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_aspect('equal')
        axes[2].tick_params(labelsize=14)
        
        # 添加颜色条 - 修改部分：增加刻度数使颜色过渡更平缓
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('OverPressure (kPa)', fontsize=14)
        cbar1.ax.tick_params(labelsize=14)
        cbar1.ax.locator_params(nbins=14)  # 增加颜色条刻度数
        
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('OverPressure (kPa)', fontsize=14)
        cbar2.ax.tick_params(labelsize=14)
        cbar2.ax.locator_params(nbins=10)  # 增加颜色条刻度数
        
        cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Error (kPa)', fontsize=14)
        cbar3.ax.tick_params(labelsize=14)
        cbar3.ax.locator_params(nbins=10)  # 增加颜色条刻度数
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return {
            'x': x_coords,
            'y': y_coords,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_error': y_error
        }
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None

def plot_metrics_vs_distance(true_data, pred_data, save_path=None, 
                             subtract_atm=True, atm_pressure=101325, 
                             figsize=(30, 10)):
    """
    绘制评估指标随距离变化的曲线图
    
    参数:
    true_data: 真实数据数组，格式为 [x, y, z, p] 或 [x, y, p]
    pred_data: 预测数据数组，格式为 [x, y, z, p] 或 [x, y, p]
    save_path: 保存路径（如果为None则显示图像）
    subtract_atm: 是否减去大气压（默认True）
    atm_pressure: 大气压值（默认101325 Pa）
    figsize: 图表大小
    """
    
    try:
        # 确保数据是二维数组
        if true_data.ndim == 1:
            true_data = true_data.reshape(1, -1)
        if pred_data.ndim == 1:
            pred_data = pred_data.reshape(1, -1)
        
        # 提取坐标
        if true_data.shape[1] >= 3:
            x_coords = true_data[:, 0]
            y_coords = true_data[:, 1]
            z_coords = true_data[:, 2] if true_data.shape[1] >= 3 else np.zeros_like(x_coords)
        else:
            print(f"错误: 数据格式不正确，列数={true_data.shape[1]}，需要至少2列坐标")
            return None
        
        # 提取压力列
        if true_data.shape[1] >= 4:
            if subtract_atm:
                y_true = true_data[:, 3] - atm_pressure
            else:
                y_true = true_data[:, 3]
        elif true_data.shape[1] >= 3:
            if subtract_atm:
                y_true = true_data[:, 2] - atm_pressure
            else:
                y_true = true_data[:, 2]
        else:
            print(f"错误: 数据格式不正确，列数={true_data.shape[1]}，需要至少3列")
            return None
        
        if pred_data.shape[1] >= 4:
            if subtract_atm:
                y_pred = pred_data[:, 3] - atm_pressure
            else:
                y_pred = pred_data[:, 3]
        elif pred_data.shape[1] >= 3:
            if subtract_atm:
                y_pred = pred_data[:, 2] - atm_pressure
            else:
                y_pred = pred_data[:, 2]
        else:
            print(f"错误: 预测数据格式不正确，列数={pred_data.shape[1]}，需要至少3列")
            return None
        
        # 确保样本数一致
        n_samples = min(len(y_true), len(y_pred))
        if len(y_true) != len(y_pred):
            print(f"警告: 样本数不一致 - 真实:{len(y_true)}, 预测:{len(y_pred)}，将截断到 {n_samples}")
            y_true = y_true[:n_samples]/1000
            y_pred = y_pred[:n_samples]/1000
            x_coords = x_coords[:n_samples]
            y_coords = y_coords[:n_samples]
            z_coords = z_coords[:n_samples]
        
        # 计算距离 (x^2 + y^2)^(1/2)
        distance = np.sqrt(x_coords**2 + y_coords**2)
        
        # 按距离排序
        sort_idx = np.argsort(distance)
        distance_sorted = distance[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # 计算误差
        absolute_error = np.abs(y_pred_sorted - y_true_sorted)
        relative_error = np.abs((y_pred_sorted - y_true_sorted) / (y_true_sorted + 1e-10)) * 100
        squared_error = (y_pred_sorted - y_true_sorted) ** 2
        
        # 创建距离区间（用于平滑显示）
        n_bins = 20
        distance_bins = np.linspace(distance_sorted.min(), distance_sorted.max(), n_bins + 1)
        distance_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        # 计算每个区间内的统计指标
        bin_mae = []
        bin_rmse = []
        bin_mape = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (distance_sorted >= distance_bins[i]) & (distance_sorted < distance_bins[i+1])
            if np.sum(mask) > 0:
                bin_errors = absolute_error[mask]
                bin_squared_errors = squared_error[mask]
                bin_relative_errors = relative_error[mask]
                
                bin_mae.append(np.mean(bin_errors))
                bin_rmse.append(np.sqrt(np.mean(bin_squared_errors)))
                bin_mape.append(np.mean(bin_relative_errors))
                bin_counts.append(np.sum(mask))
            else:
                bin_mae.append(np.nan)
                bin_rmse.append(np.nan)
                bin_mape.append(np.nan)
                bin_counts.append(0)
        
        # 创建图形（1行3列）
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. MAE vs Distance
        ax1 = axes[0]
        ax1.plot(distance_centers, bin_mae, 'b-o', linewidth=2, markersize=4, label='MAE')
        ax1.set_xlabel('Distance from Origin (m)', fontsize=16)
        ax1.set_ylabel('MAE (kPa)', fontsize=16)
        ax1.set_title('MAE vs Distance', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=16)
        
        # 2. RMSE vs Distance
        ax2 = axes[1]
        ax2.plot(distance_centers, bin_rmse, 'r-s', linewidth=2, markersize=4, label='RMSE')
        ax2.set_xlabel('Distance from Origin (m)', fontsize=16)
        ax2.set_ylabel('RMSE (kPa)', fontsize=16)
        ax2.set_title('RMSE vs Distance', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=16)
        
        # 3. MAPE vs Distance
        ax3 = axes[2]
        ax3.plot(distance_centers, bin_mape, 'g-^', linewidth=2, markersize=4, label='MAPE')
        ax3.set_xlabel('Distance from Origin (m)', fontsize=16)
        ax3.set_ylabel('MAPE (%)', fontsize=16)
        ax3.set_title('MAPE vs Distance', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=16)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return {
            'distance': distance_sorted,
            'true_values': y_true_sorted,
            'pred_values': y_pred_sorted,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'bin_centers': distance_centers,
            'bin_mae': bin_mae,
            'bin_rmse': bin_rmse,
            'bin_mape': bin_mape,
            'bin_counts': bin_counts
        }
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None

def evaluate_predictions(pred_folder, true_folder, photopath, save_results=True):
    """
    评估预测结果
    
    参数:
    pred_folder: 预测结果文件夹路径
    true_folder: 真实结果文件夹路径
    save_results: 是否保存结果到CSV文件
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(pred_folder):
        print(f"错误: 预测文件夹不存在: {pred_folder}")
        return None
    if not os.path.exists(true_folder):
        print(f"错误: 真实文件夹不存在: {true_folder}")
        return None
    
    # 获取所有预测文件
    pred_files = [f for f in os.listdir(pred_folder) if os.path.isfile(os.path.join(pred_folder, f))]
    
    if len(pred_files) == 0:
        print(f"警告: 预测文件夹中没有找到文件: {pred_folder}")
        return None
    
    print(f"\n{'='*80}")
    print(f"开始评估预测结果")
    print(f"{'='*80}")
    print(f"预测文件夹: {pred_folder}")
    print(f"真实文件夹: {true_folder}")
    print(f"找到 {len(pred_files)} 个预测文件\n")
    
    # 存储每个算例的指标
    results_list = []
    all_y_true = []
    all_y_pred = []
    
    # 表头
    print(f"{'文件名':<30s} {'样本数':<8s} {'MAE':<12s} {'MAPE(%)':<12s} {'RMSE':<12s} {'R²':<12s}")
    print("-" * 90)
    
    for pred_file in pred_files:
        # 构建真实文件路径（相同文件名）
        true_file_path = os.path.join(true_folder, pred_file)
        
        # 检查真实文件是否存在
        if not os.path.exists(true_file_path):
            print(f"警告: {pred_file} 在真实文件夹中不存在，跳过")
            continue
        
        try:
            # 读取预测结果
            pred_data = np.loadtxt(os.path.join(pred_folder, pred_file))
            if pred_data.ndim == 1:
                pred_data = pred_data.reshape(1, -1)
            
            # 读取真实结果
            true_data = np.loadtxt(true_file_path)
            if true_data.ndim == 1:
                true_data = true_data.reshape(1, -1)
            
            # 提取压力列（假设格式为 x,y,z,p）
            # 如果数据有4列，最后一列是压力
            if pred_data.shape[1] >= 4:
                y_pred = pred_data[:, 3]-101325  # 第4列是压力
            else:
                print(f"警告: {pred_file} 预测数据格式不正确，列数={pred_data.shape[1]}，需要至少4列")
                continue
            
            if true_data.shape[1] >= 4:
                y_true = true_data[:, 3]-101325   # 第4列是压力
            else:
                print(f"警告: {pred_file} 真实数据格式不正确，列数={true_data.shape[1]}，需要至少4列")
                continue
            
            # 确保样本数一致
            n_samples = min(len(y_true), len(y_pred))
            if len(y_true) != len(y_pred):
                print(f"警告: {pred_file} 样本数不一致 - 真实:{len(y_true)}, 预测:{len(y_pred)}，将截断到 {n_samples}")
                y_true = y_true[:n_samples]
                y_pred = y_pred[:n_samples]
            
            # 计算指标
            metrics = calculate_metrics(y_true, y_pred)
            
            photopath = os.path.join(resultpath, f'{pred_file}.png')
            plot_single_case_distribution_from_data(true_data, pred_data, save_path=photopath)
            photopath = os.path.join(resultpath, f'{pred_file}metrics_vs_distance.png')
            plot_metrics_vs_distance(true_data, pred_data, save_path=photopath)
            
            # 存储结果
            results_list.append({
                'case_file': pred_file,
                'n_samples': n_samples,
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })
            
            # 收集所有数据用于总体指标
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # 打印结果
            print(f"{pred_file:<30s} {n_samples:<8d} {metrics['MAE']:<12.2f} {metrics['MAPE']:<12.2f} {metrics['RMSE']:<12.2f} {metrics['R2']:<12.4f}")
            
        except Exception as e:
            print(f"处理文件 {pred_file} 时出错: {e}")
    
    if len(results_list) == 0:
        print("没有成功处理任何文件")
        return None
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 计算总体指标
    print("\n" + "="*80)
    print("总体指标（基于所有样本点）:")
    print("="*80)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    overall_metrics = calculate_metrics(all_y_true, all_y_pred)
    
    #绘制散点图
    photopath = os.path.join(resultpath, 'scatter.png')
    plot_predictions_scatter(all_y_true, all_y_pred, save_path=photopath)
    photopath0 = os.path.join(resultpath, 'true_vs_pred_magnitude.png')
    plot_prediction_magnitude_comparison(all_y_true, all_y_pred,save_path=photopath0)
    photopath1 = os.path.join(resultpath,'mape_distribution_by_magnitude.png')
    plot_mape_distribution_by_magnitude(
        all_y_true, all_y_pred,
        save_path=photopath1
    )
    photopath2 = os.path.join(resultpath,'mape_distribution_by_magnitude2.png')
    plot_mape_distribution_by_magnitude2(
        all_y_true, all_y_pred,
        save_path=photopath2
    )
    photopath3 = os.path.join(resultpath,'mape_distribution.png')
    plot_mape_distribution(
        all_y_true, all_y_pred,
        save_path=photopath3
    )
    print(f"总样本数: {len(all_y_true)}")
    print(f"MAE: {overall_metrics['MAE']:.2f} Pa")
    print(f"MAPE: {overall_metrics['MAPE']:.2f}%")
    print(f"RMSE: {overall_metrics['RMSE']:.2f} Pa")
    print(f"R²: {overall_metrics['R2']:.4f}")
    
    # 计算平均指标（按算例平均）
    print("\n" + "="*80)
    print("平均指标（按算例平均）:")
    print("="*80)
    
    # 排除NaN值计算平均值
    valid_mae = results_df['MAE'].dropna()
    valid_mape = results_df['MAPE'].dropna()
    valid_rmse = results_df['RMSE'].dropna()
    valid_r2 = results_df['R2'].dropna()
    
    avg_mae = valid_mae.mean()
    avg_mape = valid_mape.mean()
    avg_rmse = valid_rmse.mean()
    avg_r2 = valid_r2.mean()
    
    print(f"平均MAE: {avg_mae:.2f} Pa (±{valid_mae.std():.2f})")
    print(f"平均MAPE: {avg_mape:.2f}% (±{valid_mape.std():.2f})")
    print(f"平均RMSE: {avg_rmse:.2f} Pa (±{valid_rmse.std():.2f})")
    print(f"平均R²: {avg_r2:.4f} (±{valid_r2.std():.4f})")
    
    # 保存结果
    if save_results:
        output_file = os.path.join(resultpath, 'evaluation_results.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        # 保存汇总结果
        summary_df = pd.DataFrame({
            'Metric': ['MAE', 'MAPE', 'RMSE', 'R2'],
            'Overall': [overall_metrics['MAE'], overall_metrics['MAPE'], overall_metrics['RMSE'], overall_metrics['R2']],
            'Mean': [avg_mae, avg_mape, avg_rmse, avg_r2],
            'Std': [valid_mae.std(), valid_mape.std(), valid_rmse.std(), valid_r2.std()]
        })
        summary_file = os.path.join(resultpath, 'evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总结果已保存到: {summary_file}")
    
    return results_df, overall_metrics
# 指定文件夹路径
pred_folder = 'predictions_collect_pressure_peak_merged/run_1/test'
true_folder = 'grounddata_new/collect_pressure_peak_merged'
resultpath = 'folds_merged/run1'

# 评估
results = evaluate_predictions(pred_folder, true_folder, resultpath, save_results=True)