# Dropout p 网格集成实验总结

## 1. 实验目的

本次实验尝试通过折内集成提升爆炸超压预测精度。集成成员采用相同的 MLP 主结构，仅改变 dropout 概率 `p`，以验证“不同 dropout 率模型的平均预测”是否能降低 case-wise 测试集误差。

为了保证结果可以和当前基线公平对比，实验没有重新随机划分数据，而是复用当前 `predictions/run_i/train` 和 `predictions/run_i/test` 中已有的五折算例划分。这样每一折的集成模型只使用该折训练算例训练，再在该折测试算例上评估，避免不同折模型平均导致的数据泄漏。

## 2. 实验设置

- 训练脚本：`main_case_loop_ensemble.py`
- 评估脚本：`evaluate_metrics.py`
- 数据目录：`data/collect_pressure_peak`
- 基线预测目录：`predictions`
- 集成输出目录：`ensemble_outputs/dropout_p_grid`
- 五折设置：复用现有 `predictions/run_1` 到 `predictions/run_5` 的 train/test 文件列表
- 集成成员：`p = [0.0, 0.1, 0.2, 0.3, 0.5]`
- 每折成员数：5
- 总模型数：25
- 每个成员训练轮数：400 epoch
- 损失函数：MAPE loss
- 推理策略：所有成员使用 `model.eval()`，不启用 MC Dropout；最终预测取成员均值，方差列表示成员间分歧
- 预测文件格式：`x y z p_mean_abs p_var_abs`

## 3. 主要结果

当前基线结果：

| 指标 | 基线 |
| --- | ---: |
| 测试集平均超压 MAPE | 17.23% |
| 测试集平均 R2 | 0.7838 |

Dropout p 网格集成结果：

| 指标 | 集成 |
| --- | ---: |
| 测试集平均超压 MAPE | 16.85% |
| 测试集平均 R2 | 0.7905 |

总体变化：

| 指标 | 变化 |
| --- | ---: |
| 测试集平均超压 MAPE | -0.38 个百分点 |
| 测试集平均 R2 | +0.0067 |

结论：集成方法带来了小幅提升，但提升幅度有限。它说明简单平均不同 dropout 率模型可以略微提高泛化表现，但不是当前任务中最强的优化杠杆。

## 4. 逐折表现

| Run | Baseline Test MAPE | Ensemble Test MAPE | MAPE 变化 | Baseline R2 | Ensemble R2 | R2 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.76% | 16.49% | +0.73 | 0.8525 | 0.7781 | -0.0744 |
| 2 | 18.55% | 17.73% | -0.82 | 0.5399 | 0.6335 | +0.0936 |
| 3 | 22.02% | 20.93% | -1.09 | 0.8588 | 0.8831 | +0.0243 |
| 4 | 17.88% | 16.63% | -1.26 | 0.8440 | 0.7682 | -0.0758 |
| 5 | 11.94% | 12.48% | +0.54 | 0.8240 | 0.8893 | +0.0654 |

可以看到，Run 2、Run 3、Run 4 的 MAPE 有所下降；Run 1 和 Run 5 的 MAPE 反而略有上升。R2 的变化也不是单调改善，说明简单等权平均仍存在折间不稳定性。

## 5. 结果文件位置

- 总指标：`ensemble_outputs/dropout_p_grid/metrics_summary.csv`
- 与基线逐折对比：`ensemble_outputs/dropout_p_grid/baseline_vs_ensemble.csv`
- 每个 dropout 成员验证结果：`ensemble_outputs/dropout_p_grid/member_metrics.csv`
- 集成模型权重：`ensemble_outputs/dropout_p_grid/models/run_i/p_*.pth`
- 每折 scaler：`ensemble_outputs/dropout_p_grid/scalers/scaler_X_i.pkl`
- 集成预测文件：`ensemble_outputs/dropout_p_grid/predictions/run_i/train|test/value*`

复现评估命令：

```powershell
python evaluate_metrics.py --pred-base-dir ensemble_outputs/dropout_p_grid/predictions
```

## 6. 为什么提升有限

1. `p=0.0` 成员通常已经很强，较高 dropout 的成员在多个折上的验证 MAPE 更高，等权平均会把较弱成员也纳入最终预测。
2. 当前训练仍使用该折测试集选择最佳 checkpoint，这和基线一致，公平但不利于分析真正的验证集泛化规律。
3. dropout 率只改变了模型正则强度，成员多样性有限；如果多个成员误差方向相似，简单平均的收益就不会很大。
4. 任务中不同算例、不同比例距离区间的误差结构差异明显，统一等权平均不能针对高误差区域自适应调整。

## 7. 下一步优化计划

### 7.1 优先方向：基于验证表现的加权集成

当前集成是等权平均。下一步建议把每折各成员的最佳验证 MAPE 转换为权重，例如：

- 权重与 `1 / MAPE` 成正比；
- 或只保留每折验证 MAPE 最好的 2 到 3 个成员；
- 或比较等权、Top-K、Softmax 权重三种策略。

预期收益：减少高 dropout 弱成员拖累，可能比继续增加成员数更有效。

### 7.2 引入真正的验证集

当前基线和本次实验都沿用测试集选择 checkpoint。下一步建议在每折训练算例内部再划分 validation cases，用 validation 选择 checkpoint 和集成权重，最后只在 test cases 上报告一次结果。

预期收益：评估口径更严谨，也更容易判断不同超参数是否真的泛化。

### 7.3 扩展成员多样性

在 dropout p 网格之外，建议加入少量训练参数变化，但要分批实验：

- `weight_decay = [0, 1e-6, 1e-5]`
- 学习率 `lr = [5e-4, 1e-3]`
- 不同随机种子初始化

预期收益：增加成员误差差异，使集成平均更有价值。

### 7.4 针对误差区间做分段模型或分段权重

从 MCDropout notebook 的分析看，高比例距离、低超压区间更容易出现相对误差偏高。下一步可以按 `Z` 或真实/预测超压量级分段评估，并考虑：

- 为不同 `Z` 区间选择不同成员权重；
- 对低超压区间调整 loss 权重；
- 增加 `log(overpressure)` 辅助训练目标或双目标损失。

预期收益：直接针对 MAPE 的主要误差来源。

### 7.5 记录每个单成员的独立预测指标

本次已记录每个成员的训练期最佳验证 MAPE，但没有保存每个成员的完整预测目录。下一步可为每个成员单独输出测试集指标，用于判断：

- 哪个 dropout p 在不同折最稳定；
- 集成是否真的优于每折最佳单模型；
- 是否需要按折选择成员组合。

## 8. 建议的下一轮实验顺序

1. 不重新训练，先基于现有 25 个成员权重实现加权集成和 Top-K 集成推理。
2. 比较等权、Top-2、Top-3、`1/MAPE` 加权、Softmax 加权的五折测试 MAPE。
3. 若加权集成优于 16.85%，保留最佳权重策略。
4. 再引入 validation split，重新跑一轮更严格的训练与评估。
5. 最后再考虑扩大搜索空间到 weight decay、学习率和随机种子。

