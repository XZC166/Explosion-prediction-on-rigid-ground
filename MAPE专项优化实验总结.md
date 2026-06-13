# MAPE专项优化实验总结与下一轮指导

## 1. 当前评估口径

当前所有实验都围绕爆炸超压预测的五折 case-wise 划分展开，数据目录为 `data/collect_pressure_peak`，每折复用 `predictions/run_i/train` 和 `predictions/run_i/test` 中已有的算例文件列表。

主要指标为测试集平均超压 MAPE，R2 作为辅助指标。统一评估脚本为：

```powershell
python evaluate_metrics.py --pred-base-dir <prediction_dir>
```

需要注意：除非明确标注“严格 train/validation/test 复核已完成”，否则本总结中的优化结果均只能作为试验参考或方案筛选依据。凡是外层 test fold 参与 checkpoint 选择、成员验证 MAPE、Top-K/Softmax 权重或最终策略选择的结果，都不能作为正式泛化结论，也不能写入“当前正式最佳结果”。后续所有优化必须先完成严格 train/validation/test 复核，才能正式登记为主结果。

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

## 6. 第四次优化：复用 dropout 成员做 `Z` 分段只读试算

第四次试算按上一轮建议，先不重新训练，只复用第二轮已有的 25 个 dropout 成员，验证“按 `Z` 分段后每段单独选成员/权重”能否低于现有最优 15.79%。

试算口径：

- 成员来源：`ensemble_outputs/dropout_p_grid/models`、`ensemble_outputs/dropout_p_grid/scalers` 和 `ensemble_outputs/dropout_p_grid/member_metrics.csv`。
- 参考结果：`ensemble_outputs/weighted_reuse/predictions/softmax_top3_t2`。
- 由于 `ensemble_outputs/dropout_p_grid/predictions` 保存的是 5 个 dropout 成员的等权平均，不是成员级单独预测，因此本轮在内存中加载 25 个已保存模型和 scaler 做成员预测。
- 每折仍只在本折的 5 个 dropout 成员内选择权重；`Z` 分段只决定“某个测点使用哪套成员组合”。
- 分段策略用该折 train cases 估计分段阈值和段内成员表现，再在 test cases 上评估。
- 全过程只读试算，未写入新预测目录，未覆盖 `ensemble_outputs/weighted_reuse` 的任何结果。

代表性结果如下：

| 策略 | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Test MAPE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 现有 `softmax_top3_t2` | 14.35% | 16.73% | 20.57% | 15.90% | 11.42% | 15.79% | 0.8446 |
| `Z` 三分位分段，各段 Top2 | 13.76% | 17.07% | 20.35% | 16.49% | 11.78% | 15.89% | 0.8538 |
| `Z` 三分位分段，各段 Softmax Top3 T=5 | 14.30% | 16.71% | 20.17% | 16.91% | 11.58% | 15.93% | 0.8461 |
| `Z` 远场二段，各段 Softmax Top3 T=5 | 14.21% | 16.70% | 21.00% | 17.06% | 11.76% | 16.15% | 0.8432 |

第四次试算结论：

- 简单 `Z` 分段没有低于现有最佳 `softmax_top3_t2` 的 15.79%。
- 三分位分段 Top2 是本轮分段策略里最接近的方案，Test MAPE 为 15.89%，但仍比现有最佳高约 0.10 个百分点。
- 分段策略能改善部分折或局部区间，例如 Run 1 和 Run 3 有下降，但 Run 4、Run 5 被拉高，整体收益不足。
- 因为没有突破 15.79%，本轮不建议落成正式推理脚本或评估产物，避免新增一个没有主指标收益的结果目录。
- 后续主方向应转向“扩大强成员多样性”，而不是继续在同一批 25 个成员上做更复杂的 `Z` 分段搜索。

## 7. 第五次优化：MAPE loss 多样性成员池 + 加权复用

第五次实验按第四次后的建议，回到成员多样性方向：保留 MAPE loss 和当前 10 维物理特征，不再扩大 log 目标实验；新增 8 个轻量多样性成员，改变 dropout、学习率、weight decay 和随机种子，再复用第二轮已验证有效的 Top-K/Softmax 推理层加权。

成员配置：

| Member | Dropout | LR | Weight decay | Seed offset |
| --- | ---: | ---: | ---: | ---: |
| `p0_lr1e-3_wd0_s0` | 0.0 | 1e-3 | 0 | 0 |
| `p01_lr1e-3_wd0_s1` | 0.1 | 1e-3 | 0 | 1 |
| `p02_lr1e-3_wd1e-6_s0` | 0.2 | 1e-3 | 1e-6 | 0 |
| `p03_lr1e-3_wd1e-6_s1` | 0.3 | 1e-3 | 1e-6 | 1 |
| `p0_lr5e-4_wd1e-6_s1` | 0.0 | 5e-4 | 1e-6 | 1 |
| `p01_lr5e-4_wd1e-5_s0` | 0.1 | 5e-4 | 1e-5 | 0 |
| `p02_lr5e-4_wd0_s2` | 0.2 | 5e-4 | 0 | 2 |
| `p03_lr5e-4_wd1e-5_s2` | 0.3 | 5e-4 | 1e-5 | 2 |

训练与复用脚本：

- 新增 `mape_diverse_reuse.py`。
- 新增测试 `tests/test_mape_diverse_reuse.py`。
- 输出目录：`ensemble_outputs/mape_diverse_reuse`。
- 训练过程中曾被中断，因此脚本补充了断点续跑和增量写入 `member_metrics.csv` 的能力；最终 8 个成员、5 折共 40 行成员指标全部齐全。

统一加权复用结果（快速筛选口径，仅作试验参考）：

| Strategy | Train MAPE | Test MAPE | Test R2 |
| --- | ---: | ---: | ---: |
| `equal_all` | 15.21% | 15.87% | 0.8055 |
| `top1` | 15.98% | 15.87% | 0.7877 |
| `top2` | 15.26% | 15.46% | 0.8207 |
| `top3` | 15.06% | 15.35% | 0.8191 |
| `inv_mape_all` | 15.18% | 15.80% | 0.8069 |
| `inv_mape_top3` | 15.05% | 15.33% | 0.8190 |
| `softmax_top3_t2` | 15.01% | 15.26% | 0.8175 |
| `softmax_top3_t5` | 15.04% | 15.31% | 0.8186 |

快速筛选口径下的最佳来自第五次实验的 `softmax_top3_t2`：

| Run | Test MAPE | Test R2 |
| --- | ---: | ---: |
| Run 1 | 13.72% | 0.9055 |
| Run 2 | 16.00% | 0.5509 |
| Run 3 | 20.66% | 0.8236 |
| Run 4 | 14.93% | 0.8537 |
| Run 5 | 10.99% | 0.9538 |
| Average | 15.26% | 0.8175 |

第五次实验后又在新成员池上做了简单 `Z` 分段只读试算，纠正口径后使用特征中的 `Z = R / blast^(1/3)`，而不是原始坐标 `z`。代表性最好分段策略为 `Z` 三分位分段 Softmax Top3 T=5，Test MAPE 为 16.20%，没有超过统一加权的 15.26%，因此不落正式分段产物。

第五次结论：

- 在快速筛选口径下，扩大 MAPE loss 成员多样性看起来能把 Test MAPE 从第二次最佳 15.79% 降到 15.26%，下降约 0.54 个百分点。
- 快速筛选最佳仍来自按每折成员验证 MAPE 做 Softmax Top3 权重，但该验证 MAPE 实际仍依赖外层 test fold，因此只能说明该方向值得复核，不能说明已经取得正式泛化提升。
- 简单 `Z` 分段在新成员池上没有收益，暂不建议继续扩大分段搜索。
- 需要注意：该结果仍沿用快速筛选口径，成员验证 MAPE 和策略选择仍使用了外层 test fold，因此应作为方案筛选结果；若用于论文或报告，应做严格 train/validation/test 复核。

关键产物：

- `mape_diverse_reuse.py`
- `tests/test_mape_diverse_reuse.py`
- `ensemble_outputs/mape_diverse_reuse/member_metrics.csv`
- `ensemble_outputs/mape_diverse_reuse/strategy_metrics.csv`
- `ensemble_outputs/mape_diverse_reuse/best_strategy_summary.csv`
- `ensemble_outputs/mape_diverse_reuse/predictions/softmax_top3_t2`
- `ensemble_outputs/mape_diverse_reuse/metrics/softmax_top3_t2/metrics_summary.csv`

## 8. 第五次方案严格复核：train/validation/test 口径

严格复核已将第五次方案放入独立目录 `ensemble_outputs/mape_diverse_reuse_strict`，不覆盖原快速试验产物。复核口径如下：

- 外层仍保留当前五折 test cases。
- 每折 train cases 内部再按 case 划分 train/validation cases。
- checkpoint 选择、成员验证 MAPE、Top-K/Softmax 权重和最终策略选择只允许使用 validation。
- 外层 test cases 只在 validation 选定最终策略后评估一次。

严格复核结果：

| 结果 | 策略 | 选择依据 | Validation MAPE | Test MAPE | Test R2 |
| --- | --- | --- | ---: | ---: | ---: |
| 第五次严格复核 | `softmax_top3_t2` | validation MAPE | 17.48% | 19.95% | 0.8665 |
| 第五次快速参考 | `softmax_top3_t2` | outer test 快速筛选 | - | 15.26% | 0.8175 |
| 第二次快速参考 | `softmax_top3_t2` | outer test 快速筛选 | - | 15.79% | 0.8446 |

严格复核结论：

- 第五次方案在严格口径下没有复现 15.26% 的提升，最终 Test MAPE 为 19.95%。
- 因此，第五次快速结果 `15.26%` 应降级为试验参考，不应作为正式最佳结果。
- 当前不能据此认定“扩大 MAPE loss 多样性成员池”带来了可靠泛化收益。
- 后续若继续探索新成员池、分段权重或新损失函数，必须先经过相同严格复核，才能写入正式结果表。

关键产物：

- `ensemble_outputs/mape_diverse_reuse_strict/member_metrics.csv`
- `ensemble_outputs/mape_diverse_reuse_strict/strategy_metrics.csv`
- `ensemble_outputs/mape_diverse_reuse_strict/best_strategy_summary.csv`
- `ensemble_outputs/mape_diverse_reuse_strict/comparison_summary.csv`
- `ensemble_outputs/mape_diverse_reuse_strict/repro_commands.txt`

## 9. 当前结果登记

### 9.1 试验参考结果排序

下表保留历史快速筛选结果，便于回溯方案探索，但这些结果未完成严格复核，不能作为正式泛化结论。

| 阶段 | 方法 | Test MAPE | Test R2 | 状态 |
| --- | --- | ---: | ---: | --- |
| 基线 | 原始 case-wise MLP | 17.23% | 0.7838 | 历史参考 |
| 第一次 | Dropout p 等权集成 | 16.85% | 0.7905 | 试验参考 |
| 第三次 | MAPE 专项配置级 `softmax_top3_t2` | 16.14% | 0.9046 | 试验参考 |
| 第四次 | `Z` 三分位分段 Top2 只读试算 | 15.89% | 0.8538 | 试验参考 |
| 第二次 | 复用 dropout 成员 `top2` | 15.88% | 0.8246 | 试验参考 |
| 第二次 | 复用 dropout 成员 `softmax_top3_t2` | 15.79% | 0.8446 | 试验参考 |
| 第五次 | MAPE 多样性成员 `softmax_top3_t2` | 15.26% | 0.8175 | 试验参考，严格复核未通过 |

### 9.2 严格复核结果登记

| 阶段 | 方法 | Validation MAPE | Test MAPE | Test R2 | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| 第五次严格复核 | MAPE 多样性成员 `softmax_top3_t2` | 17.48% | 19.95% | 0.8665 | 正式复核结果，不优于历史参考 |
| 输入特征分层正式五折复核 | RF signed-log `rf200_l2_d16_clip5e4` | 不适用 | 13.46% | 0.8964 | 正式复核结果；平均低于 15%，但并非每折都低于 15% |

当前最应该保留的正式复核产物更新为 `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold`。`ensemble_outputs/mape_diverse_reuse_strict` 仍作为第五次方案严格复核失败记录保留；第五次快速结果 `ensemble_outputs/mape_diverse_reuse/predictions/softmax_top3_t2` 和第二次快速结果 `ensemble_outputs/weighted_reuse/predictions/softmax_top3_t2` 仍建议保留，但只作为试验参考和对照。

## 10. 后续优化实验规范与建议

后续目标仍建议以严格口径下的外层 test 平均超压 MAPE 为主。快速筛选可以继续用于发现候选方向，但结果只能进入“试验参考”；只有经过严格 train/validation/test 复核后，才能进入“正式复核结果登记”。

### 10.1 必须执行的正式结果口径

- 外层 test cases 不参与 checkpoint、成员排序、权重计算或策略选择。
- 每个外层 fold 的 train cases 内部必须再划分 validation cases。
- checkpoint 选择、成员验证 MAPE、Top-K/Softmax 权重、最终策略选择只允许使用 validation。
- test cases 只在最终策略确定后评估一次。
- 文档中凡未满足上述条件的结果，必须标注为“试验参考”。

### 10.2 下一步方向：重新设计严格口径下的候选方案

第五次严格复核退化明显，后续不应直接围绕 15.26% 继续追加复杂优化。下一步应回到严格口径重新设计候选方案，例如小幅扩充成员池或调整 validation 划分稳定性，但正式写入前必须完成严格复核：

- 保留 MAPE loss 和当前 10 维物理特征。
- 继续围绕 `lr = [5e-4, 1e-3]`、`dropout_p = [0.0, 0.1, 0.2, 0.3]` 和少量 weight decay 做补充。
- 优先补 4 到 8 个成员，避免一次性大网格。
- 训练后继续跑 `top2/top3/inv_mape_top3/softmax_top3_t2/softmax_top3_t5`。

### 10.3 暂缓方向：继续扩大分段权重

两轮 `Z` 分段只读试算均没有超过统一加权，说明当前收益不足。除非新成员池或严格复核暴露出明确的分段误差模式，否则不建议优先投入更多复杂分段搜索。

## 11. 下一轮推荐执行顺序

1. 已完成：训练 8 个 MAPE loss 多样性成员并加权复用，快速口径最佳 `softmax_top3_t2` 为 15.26%，但仅作试验参考。
2. 已完成：在新成员池上追加简单 `Z` 分段只读试算；结果未超过 15.26%，不落正式分段产物。
3. 已完成：第五次方案严格 train/validation/test 复核；严格 Test MAPE 为 19.95%，未证明可靠提升。
4. 下一步若继续优化，应先设计严格口径候选方案，再训练和复核。
5. 任何新结果未完成严格复核前，只能写入“试验参考”，不能写入“当前正式最佳”。
6. 暂不建议优先继续扩大 log 目标或复杂分段实验。

## 12. 常用命令

评估当前最佳第五轮结果：

```powershell
python evaluate_metrics.py --pred-base-dir ensemble_outputs/mape_diverse_reuse/predictions/softmax_top3_t2
```

评估第二轮对照结果：

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

复现第五轮 MAPE loss 多样性成员训练与加权复用：

```powershell
python mape_diverse_reuse.py
```

仅复用已训练第五轮成员重新跑策略评估：

```powershell
python mape_diverse_reuse.py --output-folder ensemble_outputs/mape_diverse_reuse --reuse-only
```

复现第五轮严格 train/validation/test 复核：

```powershell
python mape_diverse_reuse.py --output-folder ensemble_outputs/mape_diverse_reuse_strict --num-runs 5 --epochs 400 --batch-size 64 --eval-every 20 --validation-fraction 0.2 --validation-seed 20260609
```

仅复用严格复核成员重新跑 validation 策略选择和最终 test 评估：

```powershell
python mape_diverse_reuse.py --output-folder ensemble_outputs/mape_diverse_reuse_strict --num-runs 5 --epochs 400 --batch-size 64 --eval-every 20 --validation-fraction 0.2 --validation-seed 20260609 --reuse-only
```

单元测试：

```powershell
python -m unittest discover -s tests
```

## 13. 2026-06-10 更新：下一轮严格随机划分优化规划

本次更新在前述严格 train/validation/test 口径基础上，进一步明确后续每一次优化尝试的实验纪律：每个新方案都必须像原始 case-wise 基线一样，重新随机分配训练集和测试集，训练集与测试集互不干涉，避免提前看见测试答案。只有当候选方案在单折随机测试中相对同折原始基线出现明确提升后，才进入最严格的五折交叉验证复核。

### 13.1 新增强制实验纪律

- 每次候选优化实验都要独立生成并保存随机 case-wise split manifest。
- 同一次实验中的原始基线、候选模型、集成策略和区域诊断必须共用同一份 split manifest，保证对比是成对且公平的。
- 外层 test cases 只用于最终评估，不参与 checkpoint 选择、成员排序、权重计算、分段阈值确定或最终策略选择。
- 每个外层 train cases 内部再随机划分 inner-train/validation cases。
- checkpoint 选择、成员验证 MAPE、Top-K/Softmax 权重、难区阈值和策略选择只能使用 inner-train/validation。
- 单折随机测试只作为候选筛选入口；未通过单折测试提升的方案不进入五折严格复核。
- 五折严格复核前必须冻结方案、超参、随机划分规则和策略选择规则；五折过程中不得根据任一折 test 表现再改方案。
- 所有未完成上述流程的结果仍只能标注为“试验参考”，不能登记为“正式最佳结果”。

### 13.2 单折筛选阶段建议

下一轮先建立一个严格单折筛选口径：

1. 用固定随机种子生成 1 折 case-wise train/test manifest，并在 train 内部生成 validation manifest。
2. 在同一 manifest 上训练原始 MAPE MLP 基线，得到同折 baseline Test MAPE 和 Test R2。
3. 在同一 manifest 上训练候选集成成员，所有成员选择和权重只依据 validation。
4. 最终只对 test 评估一次，比较候选方案相对同折 baseline 的 MAPE 差值。
5. 同时输出按 `Z` 分位、真实超压分位、case 级别的只读诊断表，但这些诊断只能用于下一轮方案设计，不能反向修改当前方案。

建议通过标准：

- 候选方案 Test MAPE 明确低于同折 baseline。
- Test R2 不出现明显退化。
- 高 `Z`、低超压难区 MAPE 有改善，且不是以明显牺牲主体区域为代价。

### 13.3 五折严格复核阶段建议

单折通过后再启动五折严格复核：

- 生成 5 折随机 case-wise train/test manifest，每折 test cases 与 train cases 严格隔离。
- 每折 train cases 内部继续划分 validation。
- 每折都训练同一套 baseline 和同一套候选方案，记录成对差值。
- 策略选择以五折 validation 平均 MAPE 为依据，最终 test 只做冻结方案后的评估。
- 正式结果登记必须包含平均 Test MAPE、平均 Test R2、逐折 Test MAPE、逐折差值、标准差，以及高 `Z`/低超压区域 MAPE。

### 13.4 下一轮优先方向

第一优先方向是严格 validation 驱动的全局集成。保留当前 10 维物理特征和 MAPE loss，训练 6 到 8 个轻量成员，只在 dropout、learning rate、weight decay 和 seed 上做小范围多样性。策略候选限定为 `top2`、`top3`、`equal_top3`、`inv_mape_top3`、`softmax_top3_t2`、`softmax_top3_t5`，以及一个受约束的 validation 权重优化方案。

第二优先方向是难区感知集成，但必须保持保守。难区只允许用 train/validation 中确定的 `Z` 分位阈值、坐标/爆距特征和模型预测超压来识别；推理 test 时不能使用真实 test 超压作为分段依据。每个区域最多选 2 到 3 个成员或一套 Softmax 权重，区域样本不足时回退到全局权重。

第三优先方向是难区加权训练。暂不扩大 log 目标实验，只尝试 capped MAPE 权重，例如高 `Z` 或低预测超压区域使用 `1.0`、`1.5`、`2.0` 三档权重。阈值和权重选择仍只允许来自 train/validation。

### 13.5 暂缓方向

- 暂缓继续扩大复杂 `Z` 分段搜索。前两轮分段只读试算均未稳定超过统一加权。
- 暂缓继续扩大 log 目标路线。历史结果显示 log 目标虽然可能提高部分 R2，但对主指标 MAPE 不友好。
- 暂缓一次性大网格搜索。下一轮应优先小成员池、强约束、严格复核，避免产生更多无法正式登记的快速筛选结果。

## 14. 2026-06-11 更新：严格单折筛选与 RF signed-log 候选

本轮按照第 13 节建议，先补充了严格单折筛选入口，再基于诊断转向表格模型候选。新增产物如下：

- `mape_single_fold_screening.py`：严格单折筛选脚本，支持随机/分层 case-wise split、inner validation 策略选择、test 延迟评估。
- `mape_rf_screening.py`：随机森林 signed-log 目标筛选脚本，支持分层随机多 seed 与旧五折 manifest 对照。
- `tests/test_mape_single_fold_screening.py`
- `tests/test_mape_rf_screening.py`
- `docs/superpowers/specs/2026-06-11-single-fold-mape-screening-design.md`
- `docs/superpowers/plans/2026-06-11-single-fold-mape-screening.md`

### 14.1 MLP 严格筛选结果

先按原优先方向测试 validation 驱动的 MLP 多样性成员池，并加入两个修正：

1. 分层 split：按 negative/low/regular case 分层，使 validation 和 test 都包含负超压、低超压和常规 case。
2. signed output：取消 Softplus 正超压约束，允许模型预测负超压。

代表性结果如下：

| 方案 | Split | Baseline Test MAPE | Candidate Strategy | Candidate Test MAPE | 结论 |
| --- | --- | ---: | --- | ---: | --- |
| MLP 正输出 | 随机单折 | 20.99% | `validation_weight_opt` | 28.24% | validation 过拟合，退化 |
| MLP signed 输出 | 随机单折 | 31.08% | `validation_weight_opt` | 25.59% | 修复部分负超压 case，但仍高 |
| MLP signed 输出 | 分层单折 | 20.57% | `validation_weight_opt` | 24.34% | 仍不优于 baseline |

诊断结论：

- 随机单折中 validation 全为正超压 case，而 test 含有 `value3/value8/value9` 等负超压 case，导致 validation 不能代表 test。
- signed 输出能改善部分负超压 case，但 MLP 成员池在常规 case 上容易退化，validation 选出的成员/权重不能稳定外推。
- 因此，本轮不建议继续把 MLP 集成作为优先冲 15% 的方向。

### 14.2 RF signed-log + 裁剪 MAPE 权重候选

随后测试表格模型路线。当前最好候选使用：

- 模型：`RandomForestRegressor`
- 特征：沿用当前 10 维物理特征。
- 目标：`sign(overpressure) * log1p(abs(overpressure))`。
- 反变换：`sign(pred) * expm1(abs(pred))`。
- 样本权重：`clip(1 / abs(overpressure), 0, 1 / 50000)` 后归一化，避免近零超压点无限放大。
- `n_jobs=1`，避免当前环境中 joblib 并行权限问题。

分层随机 5 seed 复核结果：

| Config | Avg Test MAPE | Std | Avg Test R2 | Per-seed MAPE |
| --- | ---: | ---: | ---: | --- |
| `rf200_l1_d16_clip5e4` | 12.56% | 1.55 | 0.7573 | 13.78 / 10.63 / 10.77 / 13.29 / 14.31 |
| `rf100_l3_d16_clip5e4` | 12.63% | 1.63 | 0.7754 | 13.91 / 10.80 / 10.53 / 13.55 / 14.39 |
| `rf200_l2_d16_clip5e4` | 12.65% | 1.59 | 0.7650 | 13.85 / 10.68 / 10.82 / 13.41 / 14.48 |
| `rf150_l3_d16_clip5e4` | 12.69% | 1.65 | 0.7720 | 13.85 / 10.73 / 10.72 / 13.43 / 14.71 |
| `rf150_l2_none_clip5e4` | 12.74% | 1.65 | 0.7636 | 13.88 / 10.74 / 10.90 / 13.32 / 14.88 |

该口径下，5 个 seed 的所有单折 Test MAPE 均低于 15%，已经达到“MAPE 低于 15%”的筛选目标。

旧五折 manifest 对照结果：

| Config | Avg Test MAPE | Std | Avg Test R2 | Per-fold MAPE |
| --- | ---: | ---: | ---: | --- |
| `rf200_l1_d16_clip5e4` | 16.59% | 4.20 | 0.9166 | 22.68 / 17.12 / 15.45 / 18.01 / 9.70 |
| `rf150_l2_none_clip5e4` | 16.60% | 4.16 | 0.9284 | 22.56 / 17.16 / 15.41 / 18.13 / 9.74 |
| `rf150_l3_d16_clip5e4` | 16.63% | 4.15 | 0.9118 | 22.51 / 17.13 / 15.57 / 18.21 / 9.74 |
| `rf200_l2_d16_clip5e4` | 16.63% | 4.17 | 0.9219 | 22.61 / 17.18 / 15.45 / 18.17 / 9.75 |
| `rf100_l3_d16_clip5e4` | 16.65% | 4.05 | 0.9215 | 22.42 / 17.20 / 15.52 / 18.15 / 9.96 |

结论：

- RF signed-log + 裁剪 MAPE 权重是目前最强候选，在分层随机 5 seed 中稳定低于 15%。
- 该结果仍应登记为“强候选/分层随机复核结果”，暂不能直接登记为旧五折正式最佳，因为旧五折平均仍约 16.6%。
- 旧五折 Run 1 和 Run 4 是主要瓶颈，应优先诊断这些 fold 的 test case 组成和低超压/负超压比例。

### 14.3 推荐下一步

1. 冻结 `rf200_l1_d16_clip5e4` 作为下一轮主候选。
2. 建立正式“五折分层 case-wise manifest”，替代历史随机五折 manifest 中难区分布不均的问题。
3. 在正式分层五折上同时跑原 MLP baseline 和 RF signed-log 候选，登记成对差值。
4. 若正式分层五折仍低于 15%，再考虑写入“当前正式最佳”。
5. 若必须沿用旧五折 manifest，则下一步应集中优化 Run 1/Run 4，而不是继续追逐分层随机口径下的更低数值。

复现命令：

```powershell
python mape_rf_screening.py --output-folder ensemble_outputs/mape_rf_screening --legacy-fivefold
```

## 15. 2026-06-11 追加：仅使用预测前已知信息的分层复核

第 14 节中的分层随机结果使用了 case 内真实超压分布给 case 打粗略标签。该做法适合作为探索筛选，但最严格口径下会被质疑使用了 test case 的真实压力信息。因此本次追加复核将分层规则改为只使用预测前已知的输入特征：

- `blast`
- case 内 `Z = R / blast^(1/3)` 的中位值和最大值
- 由上述输入量生成的桶：`low_charge_near`、`low_charge_far`、`mid_charge_near`、`mid_charge_far`、`high_charge_near`、`high_charge_far`

分层仍保持 case-wise，不拆分同一个 case 内部测点。新测试 `test_case_difficulty_bucket_uses_only_known_input_features` 已验证：同一 case 的 `X` 不变而 `y` 改变时，分层 bucket 不变。

输入特征分层随机 5 seed 复核结果如下：

| Config | Avg Test MAPE | Std | Avg Test R2 | Per-seed MAPE |
| --- | ---: | ---: | ---: | --- |
| `rf200_l2_d16_clip5e4` | 13.96% | 1.89 | 0.9066 | 16.76 / 12.52 / 11.33 / 14.35 / 14.84 |
| `rf200_l1_d16_clip5e4` | 13.96% | 1.85 | 0.9198 | 16.81 / 12.75 / 11.33 / 14.24 / 14.67 |
| `rf150_l2_none_clip5e4` | 14.00% | 1.84 | 0.9099 | 16.80 / 12.64 / 11.43 / 14.38 / 14.74 |
| `rf150_l3_d16_clip5e4` | 14.01% | 1.85 | 0.8937 | 16.81 / 12.52 / 11.50 / 14.39 / 14.85 |
| `rf100_l3_d16_clip5e4` | 14.04% | 1.83 | 0.8982 | 16.98 / 12.51 / 11.74 / 14.38 / 14.60 |

旧五折 manifest 对照不变，最好仍为：

| Config | Avg Test MAPE | Std | Avg Test R2 | Per-fold MAPE |
| --- | ---: | ---: | ---: | --- |
| `rf200_l1_d16_clip5e4` | 16.59% | 4.20 | 0.9166 | 22.68 / 17.12 / 15.45 / 18.01 / 9.70 |

本次追加结论：

- 改成只用预测前已知信息分层后，RF signed-log + 裁剪 MAPE 权重仍能把分层随机 5 seed 平均 Test MAPE 压到 15% 以下。
- 最优平均 MAPE 从真实压力分层的 12.56% 回升到 13.96%，但口径更干净，且平均 R2 提升到 0.9066。
- 该结果可以作为当前最可信的“低于 15% 强候选”保留。
- 仍需注意：5 个 seed 中 seed 20260611 的 Test MAPE 为 16.76%，说明方案还没有做到每一折都低于 15%。
- 若要登记为正式最佳，下一步应冻结输入特征分层规则和 `rf200_l2_d16_clip5e4`，再建立正式分层五折，并与同一 manifest 下的 MLP baseline 做成对复核。

复现命令：

```powershell
python mape_rf_screening.py --output-folder ensemble_outputs/mape_rf_screening_input_stratified --legacy-fivefold
```

## 16. 2026-06-12 追加：输入特征分层规则正式五折复核

本轮是在第 15 节“仅使用预测前已知信息分层”的基础上继续推进的正式复核。与上一轮 `ensemble_outputs/mape_rf_screening_input_stratified` 的“5 个 stratified seed”筛选口径不同，本轮重新生成一套正式 stratified case-wise 五折 manifest，确保 39 个 case 每个且仅一次进入外层 test。

本轮冻结规则如下：

- 分层只使用预测前已知输入特征，不使用真实超压 `y`。
- 输入量包括 `blast`、case 内 `Z = R / blast^(1/3)` 的中位值和最大值。
- 分层 bucket 固定为 `low_charge_near`、`low_charge_far`、`mid_charge_near`、`mid_charge_far`、`high_charge_near`、`high_charge_far`。
- 同一个 case 内部测点不拆分，始终保持 case-wise。
- 正式 manifest seed 固定为 `20260612`。
- 39 个 case 的 test 分配覆盖检查：五折 test size 为 9 / 8 / 8 / 7 / 7，所有 case 在 test 中出现且只出现一次。

本轮冻结候选为：

- `rf200_l2_d16_clip5e4`
- `RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_depth=16, random_state=2002, n_jobs=1)`
- 目标：`sign(overpressure) * log1p(abs(overpressure))`
- 反变换：`sign(pred) * expm1(abs(pred))`
- 样本权重：`clip(1 / abs(overpressure), 0, 1 / 50000)` 后归一化

同一套 manifest 上同时训练原始 MLP baseline 和 RF 候选，得到逐折成对结果如下：

说明：这里的“原始 MLP baseline”是为了和最终 RF 候选做公平成对比较，在本轮新冻结的输入特征分层 case-wise 五折 manifest 上重新训练得到的结果；它与前文历史参考中的“原始 case-wise MLP 基线 17.23%”不是同一次数据划分下的结果。17.23% 对应早期评估口径，可作为历史基准；25.17% 对应最终正式五折复核口径，用于衡量 RF 在同一 train/test case 划分下相对 MLP 的提升。

| Fold | MLP Test MAPE | MLP Test R2 | RF Test MAPE | RF Test R2 | RF - MLP MAPE | RF < 15% |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 14.10% | 0.7902 | 12.15% | 0.8874 | -1.96 | 是 |
| 2 | 38.49% | 0.6014 | 12.11% | 0.8671 | -26.37 | 是 |
| 3 | 29.16% | 0.7193 | 13.80% | 0.9839 | -15.36 | 是 |
| 4 | 23.22% | 0.9318 | 13.43% | 0.9353 | -9.79 | 是 |
| 5 | 20.90% | 0.8540 | 15.79% | 0.8083 | -5.10 | 否 |

汇总结果：

| 方法 | Avg Test MAPE | Std Test MAPE | Avg Test R2 | Std Test R2 |
| --- | ---: | ---: | ---: | ---: |
| 原始 MLP baseline | 25.17% | 8.22 | 0.7793 | 0.1133 |
| RF `rf200_l2_d16_clip5e4` | 13.46% | 1.35 | 0.8964 | 0.0598 |

成对差值：

- RF 平均 Test MAPE 比 MLP baseline 低 11.72 个百分点。
- RF 平均 Test R2 高于 MLP baseline。
- RF 有 4 / 5 折低于 15%。
- RF 平均 Test MAPE 低于 15%，但第 5 折为 15.79%，因此不能表述为“每一折均低于 15%”。

本轮结论：

- `rf200_l2_d16_clip5e4` 可以登记为当前最可信的正式复核结果，平均 Test MAPE 已低于 15%。
- 该结论比第 15 节的输入特征分层随机 5 seed 结果更正式，因为本轮使用一套完整五折 manifest，并与同 manifest 下的原始 MLP baseline 做了成对比较。
- 由于第 5 折仍高于 15%，当前推荐表述为“正式分层五折平均 MAPE 低于 15%”，而不是“所有 fold 均低于 15%”。
- 下一步若继续推进，应优先诊断第 5 折 test cases，而不是继续更换分层规则。

关键产物：

- `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/formal_manifest.csv`
- `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/frozen_input_stratification_rule.csv`
- `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/formal_pair_results.csv`
- `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/formal_fivefold/formal_summary.csv`
- `ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/repro_commands.txt`

复现命令：

```powershell
python mape_rf_screening.py --formal-fivefold --output-folder ensemble_outputs/mape_rf_screening_formal_stratified_fivefold --data-folder data/collect_pressure_peak --case-info-path data/case_info.csv --rf-configs "rf200_l2_d16_clip5e4:200:2:16:2002:50000" --formal-candidate-config rf200_l2_d16_clip5e4 --formal-folds 5 --formal-seed 20260612 --epochs 400 --batch-size 64 --baseline-seed 42 --threshold-percent 15.0
```
