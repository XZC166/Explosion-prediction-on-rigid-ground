# 最终 RF signed-log 方案说明

本文档说明当前项目最后保留的正式方案，包括它使用哪些代码、是不是集成方法、如何复现正式五折结果、会产生哪些结果文件，以及后续如需预测新数据应注意什么。

## 1. 方案结论

最终推荐方案是 `rf200_l2_d16_clip5e4`，在正式输入特征分层 case-wise 五折复核中的结果为：

| 方法 | Avg Test MAPE | Std Test MAPE | Avg Test R2 | Std Test R2 |
| --- | ---: | ---: | ---: | ---: |
| 原始 MLP baseline | 25.17% | 8.22 | 0.7793 | 0.1133 |
| RF `rf200_l2_d16_clip5e4` | 13.46% | 1.35 | 0.8964 | 0.0598 |

RF 平均 Test MAPE 低于 15%，但 5 折中第 5 折为 15.79%，因此准确表述应为：

- 正式分层五折平均 MAPE 低于 15%。
- 4 / 5 折低于 15%。
- 不能表述为每一折都低于 15%。

## 2. 它是不是集成方法

它不是此前那种“多个独立模型预测后再加权平均”的集成方案。

最终方案使用的是 `sklearn.ensemble.RandomForestRegressor`。随机森林内部包含多棵决策树，因此从算法类别上属于树模型 ensemble；但在本项目的实验语境里，它是一个单一 RF 模型配置，不做多个 RF/MLP 成员之间的 Top-K、Softmax、加权平均或 stacking。

本轮正式复核中每一折都会用该折训练集重新训练一个 RF 模型，并在该折 test cases 上评估。正式结果不是由 5 个 fold 模型集成得到，而是 5 折交叉复核指标的平均。

## 3. 使用的主要代码

核心脚本：

- `mape_rf_screening.py`

关键逻辑位置：

- `FORMAL_RF_CONFIG`：冻结最终 RF 配置。
- `generate_stratified_case_folds(...)`：生成正式输入特征分层 case-wise 五折 manifest。
- `train_predict_config(...)`：训练 RF signed-log 模型并评估 test MAPE/R2。
- `train_predict_mlp_baseline(...)`：在同一 manifest 上训练原始 MLP baseline，用作成对对照。
- `write_frozen_input_stratification_rule(...)`：写出冻结分层规则。
- `run_formal_fivefold(...)`：正式复核主流程。

相关支撑脚本：

- `main_case_loop_ensemble.py`：提供数据加载、10 维特征构造、MLP baseline、预测文件写出等基础函数。
- `mape_single_fold_screening.py`：提供输入特征分层 bucket 规则 `case_difficulty_bucket(...)`。
- `evaluate_metrics.py`：通用预测目录评估工具。
- `tests/test_mape_rf_screening.py`：RF signed-log、正式五折 manifest、汇总逻辑的单元测试。

## 4. 输入特征

每个测点使用 10 维物理特征：

| 特征 | 含义 |
| --- | --- |
| `x` | 测点 x 坐标 |
| `y` | 测点 y 坐标 |
| `z` | 测点 z 坐标 |
| `blast` | 爆炸当量/装药量 |
| `b_height` | 装药或爆心相关高度字段，对应 `case_info.csv` 中 `bili_height` |
| `height` | case 高度字段 |
| `R` | `sqrt(x^2 + y^2 + z^2)` |
| `Z` | `R / blast^(1/3)` |
| `log_Z` | `log(Z + 1e-5)` |
| `inv_Z` | `1 / (Z + 1e-5)` |

训练标签使用超压：

```text
overpressure = absolute_pressure - 101325
```

## 5. 目标变换和样本权重

RF 不直接拟合原始超压，而是拟合 signed-log 目标：

```text
y_rf = sign(overpressure) * log1p(abs(overpressure))
```

预测后反变换：

```text
pred_overpressure = sign(pred_rf) * expm1(abs(pred_rf))
pred_absolute_pressure = pred_overpressure + 101325
```

训练样本权重用于贴近 MAPE 目标，并避免近零超压点权重无限放大：

```text
weight = clip(1 / abs(overpressure), 0, 1 / 50000)
```

随后对权重做均值归一化，使平均权重为 1。

## 6. RF 配置

最终候选名：

```text
rf200_l2_d16_clip5e4
```

对应配置：

```text
n_estimators = 200
min_samples_leaf = 2
max_depth = 16
random_state = 2002
clip_denominator = 50000
n_jobs = 1
```

`n_jobs=1` 是为了避免当前 Windows/权限环境中 joblib 并行可能出现的问题。

## 7. 正式分层五折规则

正式复核使用 input-only stratified case-wise 五折。分层只使用预测前已知输入信息，不使用真实超压 `y`。

分层输入量：

- `blast`
- case 内 `Z = R / blast^(1/3)` 的中位值
- case 内 `Z` 的最大值

bucket 固定为：

- `low_charge_near`
- `low_charge_far`
- `mid_charge_near`
- `mid_charge_far`
- `high_charge_near`
- `high_charge_far`

同一个 case 内的测点不会被拆到不同 fold。当前正式 manifest 使用 seed：

```text
20260612
```

39 个 case 在 test 中每个出现且只出现一次，五折 test size 为：

```text
9 / 8 / 8 / 7 / 7
```

## 8. 如何复现正式五折结果

运行命令：

```powershell
python mape_rf_screening.py --formal-fivefold --output-folder ensemble_outputs/mape_rf_screening_formal_stratified_fivefold --data-folder data/collect_pressure_peak --case-info-path data/case_info.csv --rf-configs "rf200_l2_d16_clip5e4:200:2:16:2002:50000" --formal-candidate-config rf200_l2_d16_clip5e4 --formal-folds 5 --formal-seed 20260612 --epochs 400 --batch-size 64 --baseline-seed 42 --threshold-percent 15.0
```

注意：脚本默认保护已有正式结果目录。如果要保留现有产物，复跑时请换一个新的 `--output-folder`。只有确认要覆盖同目录时，才添加：

```powershell
--overwrite
```

## 9. 会产生哪些结果文件

正式结果目录：

```text
ensemble_outputs/mape_rf_screening_formal_stratified_fivefold
```

主要文件：

| 路径 | 内容 |
| --- | --- |
| `repro_commands.txt` | 本次正式复核的复现命令 |
| `formal_fivefold/formal_manifest.csv` | 整套五折 manifest，记录每个 case 属于哪个 fold/split |
| `formal_fivefold/frozen_input_stratification_rule.csv` | 冻结的输入特征分层规则 |
| `formal_fivefold/formal_pair_results.csv` | 逐折 MLP baseline 与 RF 候选的成对 MAPE/R2 |
| `formal_fivefold/formal_summary.csv` | 平均值、标准差、15% 阈值判定等正式汇总 |
| `formal_fivefold/manifests/run_i/split_manifest.csv` | 每一折单独的 train/test case 列表 |
| `formal_fivefold/baseline_model/run_i/model.pth` | 每一折训练出的 MLP baseline checkpoint |
| `formal_fivefold/baseline_model/run_i/scaler_X.pkl` | 每一折 MLP baseline 的特征 scaler |
| `formal_fivefold/baseline_predictions/run_i/...` | 每一折 MLP baseline 的 train/test 预测文件 |
| `formal_fivefold/candidate_predictions/rf200_l2_d16_clip5e4/run_i/...` | 每一折 RF 候选的 train/test 预测文件 |

预测文件格式为：

```text
x y z p_pred_abs p_var_abs
```

其中 `p_pred_abs` 是预测绝对压力，`p_var_abs` 对单个 RF 候选固定写为 0。

## 10. 如何查看正式结果

最重要的两个表：

```text
ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/formal_summary.csv
ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/formal_pair_results.csv
```

`formal_pair_results.csv` 中每一行是一折，包含：

- `baseline_mape`
- `baseline_r2`
- `candidate_mape`
- `candidate_r2`
- `delta_mape_candidate_minus_baseline`
- `candidate_below_15_percent`
- `test_cases`

`formal_summary.csv` 中包含：

- `baseline_avg_mape`
- `baseline_std_mape`
- `candidate_avg_mape`
- `candidate_std_mape`
- `candidate_avg_r2`
- `candidate_std_r2`
- `candidate_avg_below_15_percent`
- `candidate_all_folds_below_15_percent`
- `candidate_folds_below_15_percent`

## 11. RF 可视化图说明

RF 结果可使用 `show_rf.py` 生成可视化图。默认读取最终正式五折 RF 候选的 test 预测结果：

```text
ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/candidate_predictions/rf200_l2_d16_clip5e4
```

并与真实数据目录对比：

```text
data/collect_pressure_peak
```

默认输出目录为：

```text
rf_visual_results
```

主要图包括：

| 图名 | 含义 |
| --- | --- |
| `scatter.png` | 真实超压与预测超压散点图，含 `y=x` 和 `±30%` 误差带 |
| `true_vs_pred_magnitude.png` | 真实值与预测值在不同超压数量级区间内的占比分布 |
| `mape_distribution.png` | 所有测点的 MAPE 区间占比分布；柱高是样本占比，不是平均 MAPE |
| `mape_distribution_by_magnitude.png` | 不同真实超压数量级内的 MAPE 区间占比分布 |
| `rf_all_cases_combined.png` | 按真实超压从小到大排序的全体 test 测点 POI 对比图 |
| `rf_distance_comparison.png` | 全体 test 测点按距离分箱后的真实/预测趋势对比图 |
| `individual_cases/value*_rf_distance_comparison.png` | 指定单个 case 的距离-超压真实/预测对比图 |

需要注意：最终 RF 是确定性模型，预测文件中的 `p_var_abs` 对单个 RF 候选固定为 0，因此这些 RF 可视化图中的阴影带不是 MC Dropout 图中的真实 `95% confidence interval`。

`rf_distance_comparison.png` 中的 `Binned variation band` 是按距离分箱后计算的辅助波动带。具体做法是：对每个距离箱计算预测均值、真实均值和残差标准差，其中

```text
residual = pred_kpa - true_kpa
```

图中阴影为：

```text
pred_mean ± residual_std
```

它表示同一距离段内 RF 预测误差的离散程度，只用于辅助观察趋势。

单个 case 图中的 `Case residual band` 是该 case 内所有测点残差标准差形成的辅助波动带。图中阴影为：

```text
pred_kpa ± residual_std(case)
```

它表示该 case 内预测误差的大致波动范围，也不是概率意义上的置信区间。

若只想生成指定 case 的单独距离对比图，可运行：

```powershell
python show_rf.py --individual-cases 14,18,26,31,36,5
```

## 12. 新数据预测时需要注意

当前正式复核保存的是每折交叉验证模型和预测结果。若要部署到新数据，建议另行训练一个“最终部署模型”：

- 使用全部 39 个已有标注 case 训练一个 RF `rf200_l2_d16_clip5e4`。
- 保存为类似 `final_models/rf200_l2_d16_clip5e4.joblib` 的文件。
- 对新 case 使用同样的 10 维特征构造、signed-log 反变换和输出格式。

也就是说，本方案已经证明 RF 配置可靠，但当前仓库还没有单独封装“训练最终部署模型”和“预测新文件”的脚本。若需要给工程使用，建议下一步补充：

- `train_final_rf_model.py`
- `predict_with_final_rf.py`

## 13. 与历史实验的关系

此前尝试过多种 MLP ensemble、Dropout 网格、MAPE 专项训练、成员加权、分段权重和单折筛选。最终选择 RF signed-log 方案的原因是：

- MLP 多成员加权在严格复核中没有稳定复现快速筛选收益。
- 分段权重没有稳定超过统一加权。
- RF signed-log + 裁剪 MAPE 权重在 input-only 分层复核中平均低于 15%。
- 正式五折中，RF 对同 manifest 下的原始 MLP baseline 有明显成对提升。

更完整的实验过程见：

```text
MAPE专项优化实验总结.md
```
