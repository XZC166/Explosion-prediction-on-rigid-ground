# MAPE专项优化实验总结与下一轮指导

## 1. 当前评估口径

当前所有实验都围绕爆炸超压预测的五折 case-wise 划分展开，数据目录为 `data/collect_pressure_peak`，每折复用 `predictions/run_i/train` 和 `predictions/run_i/test` 中已有的算例文件列表。

主要指标为测试集平均超压 MAPE，R2 作为辅助指标。统一评估脚本为：

```powershell
python evaluate_metrics.py --pred-base-dir <prediction_dir>
```

需要注意：前三轮快速实验基本沿用“该折测试集参与 checkpoint 或权重选择”的旧口径，因此适合做方案筛选，不应直接当作最终严格泛化结论。下一轮若要形成正式结果，应引入 train/validation/test 口径。

## 2. 基线阶段

基线模型使用 `main_case_loop.py`，核心设置为：

- MLP 主结构不变。
- 输入特征扩展到 10 维：`x y z blast b_height height R Z log_Z inv_Z`。
- 激活函数使用 SiLU。
- 输出用 Softplus 保证压力预测为正。
- 训练 400 epoch。
- 损失函数为 MAPE loss。

基线结果：

| 方法 | Test MAPE | Test R2 |
| --- | ---: | ---: |
| 原始 case-wise MLP 基线 | 17.23% | 0.7838 |

基线说明：当前模型已经能把测试 MAPE 控制到 20% 以下，但不同折和不同算例之间差异明显，远场、低超压区间更容易拉高 MAPE。

## 3. 第一次优化：Dropout p 网格等权集成

第一次集成实验使用 `main_case_loop_ensemble.py`，目标是验证“同一 MLP 主结构，不同 dropout p 成员等权平均”能否稳定降低 MAPE。

实验设置：

- 输出目录：`ensemble_outputs/dropout_p_grid`
- Dropout 网格：`p = [0.0, 0.1, 0.2, 0.3, 0.5]`
- 5 折，每折 5 个成员，共 25 个模型。
- 每个成员 400 epoch。
- 每折成员预测取等权平均。
- 预测文件格式统一为 `x y z p_mean_abs p_var_abs`。

主要结果：

| 方法 | Test MAPE | Test R2 |
| --- | ---: | ---: |
| 基线 | 17.23% | 0.7838 |
| Dropout p 等权集成 | 16.85% | 0.7905 |

逐折表现显示，Run 2、Run 3、Run 4 的 MAPE 有下降，但 Run 1 和 Run 5 反而上升。结论是：简单等权集成有小幅收益，但会把较弱 dropout 成员也纳入平均，提升有限。

关键产物：

- `ensemble_outputs/dropout_p_grid/member_metrics.csv`
- `ensemble_outputs/dropout_p_grid/metrics_summary.csv`
- `ensemble_outputs/dropout_p_grid/predictions`
- `ensemble_outputs/dropout_p_grid/models/run_i/p_*.pth`

## 4. 第二次优化：复用 25 个成员做加权/Top-K 集成

第二次实验不重新训练，使用 `ensemble_weighted_inference.py` 复用第一次训练出的 25 个成员模型，只改变推理阶段的成员选择和权重。

实验策略：

- `equal_all`
- `top1`
- `top2`
- `top3`
- `inv_mape_all`
- `inv_mape_top3`
- `softmax_top3_t2`
- `softmax_top3_t5`

输出目录：`ensemble_outputs/weighted_reuse`

主要结果：

| 策略 | Train MAPE | Test MAPE | Test R2 |
| --- | ---: | ---: | ---: |
| `equal_all` | 15.49% | 16.85% | 0.7905 |
| `top1` | 14.52% | 16.25% | 0.8600 |
| `top2` | 15.15% | 15.88% | 0.8246 |
| `top3` | 14.89% | 16.23% | 0.8304 |
| `inv_mape_all` | 15.37% | 16.72% | 0.7955 |
| `inv_mape_top3` | 14.82% | 16.16% | 0.8323 |
| `softmax_top3_t2` | 14.42% | 15.79% | 0.8446 |
| `softmax_top3_t5` | 14.67% | 16.01% | 0.8370 |

当前全局最佳结果来自 `softmax_top3_t2`：

| 方法 | Test MAPE | Test R2 |
| --- | ---: | ---: |
| Dropout p 等权集成 | 16.85% | 0.7905 |
| 复用成员 `softmax_top3_t2` | 15.79% | 0.8446 |

结论：第二轮证明了“成员筛选和加权”比继续等权平均更有效。弱成员会拖累平均结果，按每折成员 MAPE 做 Top-K/Softmax 加权是目前最有效的一步。

关键产物：

- `ensemble_outputs/weighted_reuse/strategy_metrics.csv`
- `ensemble_outputs/weighted_reuse/best_strategy_summary.csv`
- `ensemble_outputs/weighted_reuse/predictions/softmax_top3_t2`

## 5. 第三次优化：MAPE 专项训练

第三次实验尝试从训练目标本身继续压 MAPE，使用 `mape_focused_training.py` 新训练 4 类配置，再用 `mape_config_ensemble.py` 做配置级集成。

设计动机：

- 前两轮说明集成权重有效，但不一定解决误差来源。
- 诊断显示高 `Z` 远场、低超压区间对 MAPE 影响较大。
- 因此尝试 log 目标、高 `Z` 样本权重、以及二者组合。

训练配置：

- `log_l1`：预测 `log1p(overpressure)`，使用 L1 loss。
- `log_huber`：预测 `log1p(overpressure)`，使用 SmoothL1/Huber loss。
- `z_weighted_mape`：保持线性 overpressure 目标，对高 `Z` 样本提高 MAPE loss 权重。
- `log_l1_z_weighted`：log 目标 + 高 `Z` 样本权重。

输出目录：`ensemble_outputs/mape_focused_training`

单配置完整结果：

| Config | Train MAPE | Test MAPE | Test R2 |
| --- | ---: | ---: | ---: |
| `log_l1` | 13.90% | 18.22% | 0.8834 |
| `log_huber` | 14.14% | 19.12% | 0.8946 |
| `z_weighted_mape` | 15.11% | 16.92% | 0.7964 |
| `log_l1_z_weighted` | 12.13% | 17.95% | 0.9058 |

配置级集成结果：

| Strategy | Train MAPE | Test MAPE | Test R2 |
| --- | ---: | ---: | ---: |
| `best_single` | 15.11% | 16.92% | 0.7964 |
| `equal_top2` | 12.55% | 16.15% | 0.8975 |
| `equal_top3` | 12.36% | 16.35% | 0.9157 |
| `softmax_top3_t2` | 12.72% | 16.14% | 0.9046 |

第三轮结论：

- 单配置最佳是 `z_weighted_mape`，Test MAPE 为 16.92%。
- 配置级最佳是 `softmax_top3_t2`，Test MAPE 为 16.14%。
- 第三轮优于第一次等权集成 16.85%，但没有超过第二轮最佳 15.79%。
- log 目标虽然能提高部分 R2，但对主目标 MAPE 不友好，不建议作为下一轮主方向。
- 高 `Z` 加权有一定价值，但单独使用还不够，需要与更强成员多样性或分段权重结合。

关键产物：

- `mape_focused_training.py`
- `mape_config_ensemble.py`
- `ensemble_outputs/mape_focused_training/config_metrics.csv`
- `ensemble_outputs/mape_focused_training/config_ensemble/strategy_metrics.csv`

## 6. 当前最好结果排序

| 阶段 | 方法 | Test MAPE | Test R2 |
| --- | --- | ---: | ---: |
| 基线 | 原始 case-wise MLP | 17.23% | 0.7838 |
| 第一次 | Dropout p 等权集成 | 16.85% | 0.7905 |
| 第三次 | MAPE 专项配置级 `softmax_top3_t2` | 16.14% | 0.9046 |
| 第二次 | 复用 dropout 成员 `top2` | 15.88% | 0.8246 |
| 第二次 | 复用 dropout 成员 `softmax_top3_t2` | 15.79% | 0.8446 |

目前最应该保留的结果是第二次优化的 `ensemble_outputs/weighted_reuse/predictions/softmax_top3_t2`。

## 7. 下一次优化实验建议

下一轮目标仍建议以测试集平均超压 MAPE 为主，优先尝试低成本但更贴近现有成功路径的方向。

### 7.1 优先方向：扩大强成员多样性，再做加权复用

第二轮说明“同一批成员中选择/加权”有效，第三轮说明“单独改训练目标”收益不足。因此下一轮建议不要把主力放在 log loss，而是回到成员多样性：

- 保留 MAPE loss 和当前 10 维物理特征。
- 新增少量成员配置：
  - `dropout_p = [0.0, 0.1, 0.2, 0.3]`
  - `weight_decay = [0, 1e-6, 1e-5]`
  - `lr = [5e-4, 1e-3]`
  - 可加入 2 到 3 个随机种子。
- 不需要一次全网格爆炸式训练，建议先选 8 到 12 个成员做五折实验。
- 训练后继续使用 `top2/top3/softmax_top3_t2/softmax_top3_t5` 复用推理。

预期：比单独做 log 目标更可能突破 15.79%。

### 7.2 次优方向：按区间做分段权重

现有统一权重对所有测点使用同一个成员组合，而误差集中在远场和低超压区间。下一轮可以在推理阶段做分段加权，不重新训练或少量重训：

- 按 `Z` 分段，例如近场、中场、远场。
- 或按预测超压量级分段，避免依赖真实标签。
- 每个区间单独选择 Top-K 成员或 Softmax 权重。
- 输出仍保持 `x y z p_mean_abs p_var_abs`。

这个方向应作为推理层优化，优先复用第二轮 25 个 dropout 成员，先做只读/快速试算。

### 7.3 必做方向：严格验证口径

当前最佳 15.79% 属于快速筛选口径。下一轮如果要形成可靠论文或报告结论，应增加严格版本：

- 外层仍保留当前五折 test cases。
- 每折 train cases 内部再划分 validation cases。
- checkpoint 选择、成员权重、策略选择只允许使用 validation。
- test cases 最后只评估一次。

建议先用快速口径找到可能低于 15.79% 的方案，再把最优 1 到 2 个方案放到严格口径复核。

## 8. 下一轮推荐执行顺序

1. 先复用第二轮 25 个 dropout 成员，做 `Z` 分段权重的只读试算。
2. 如果分段权重能低于 15.79%，新增正式推理脚本并落盘评估。
3. 如果分段权重收益不足，训练 8 到 12 个 MAPE loss 多样性成员。
4. 对新成员池继续跑 `top2/top3/softmax_top3_t2/softmax_top3_t5`。
5. 若快速口径低于 15.79%，再做严格 train/validation/test 复核。
6. 暂不建议优先继续扩大 log 目标实验，因为第三轮结果没有显示出 MAPE 优势。

## 9. 常用命令

评估当前最佳第二轮结果：

```powershell
python evaluate_metrics.py --pred-base-dir ensemble_outputs/weighted_reuse/predictions/softmax_top3_t2
```

复现第二轮加权复用：

```powershell
python ensemble_weighted_inference.py
```

复现第三轮 MAPE 专项训练：

```powershell
python mape_focused_training.py
python mape_config_ensemble.py
```

单元测试：

```powershell
python -m unittest discover -s tests
```
