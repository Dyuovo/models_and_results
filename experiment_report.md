# xmo 实验复用报告

## 1. 实验目标
对广东电价序列进行 15 分钟粒度的多模型预测对比，评估基础模型与残差融合方案在不同分时段（谷/平/峰）上的表现，并输出可复现的指标、预测与图表结果。

## 2. 数据与输入
- 训练集：guangdong_2025_complete.csv
- 验证集：guangdong_2026_complete.csv
- 时间字段：times（强制 15 分钟连续）
- 目标字段：day_ahead_price、real_time_price

相关入口与参数见 [run_compare_prices.py](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L26-L91)。

### 2.1 切分与范围
- 训练数据与验证数据按年度文件分离，便于复现对比。
- 训练与验证均要求 15 分钟连续时间序列，避免时序缺口影响滚动预测。

## 3. 核心流程与特征
### 3.1 分段规则
按日内时段分为谷/平/峰：
- 谷：00:00–08:00
- 平：08:00–10:00、12:00–14:00、19:00–24:00
- 峰：其余时间

见 [get_segments](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L108-L117)。

### 3.2 静态特征
小时、分钟、星期、96 点位编码 + 正余弦、节假日与工作日特征、节前后 1 天与节日名称 one-hot。  
见 [build_static_features](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L147-L184)。

### 3.3 动态特征
多阶滞后（包含 1/96/672 与扩展滞后）、滚动均值、滚动标准差、差分特征。  
见 [build_dynamic_features](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L187-L197)。

### 3.4 评估指标
MAE、RMSE、MAPE%、sMAPE%，以及自定义 sMAPE（用于分段选择）。  
见 [calc_metrics](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L200-L231)。

### 3.5 目标与派生指标
- 基础目标：day_ahead_price、real_time_price
- spread 目标：按日计算峰段均价与谷段均价的差值，产物示例见 [spread_daily_series_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke/spread_daily_series_day_ahead_price.csv)

## 4. 模型方案
### 4.1 基础模型
- XGBoost 分段滚动重训
- Chronos-2 基础模型

### 4.2 残差融合
1) Chronos + XGB 残差  
2) XGB + Chronos 残差  
见 [run_for_target](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L664-L753)。

### 4.3 分段混合
对每个分段选取最佳模型输出混合结果，可用 MAE 或 sMAPE 作为选择指标。  
见 [choose_segment_best_models](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L491-L570)。

### 4.4 实时价峰段增强
针对 real_time_price，允许峰段更频繁重训与参数增强（n_estimators、max_depth、learning_rate）。  
见 [run_for_target](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L649-L662)。

### 4.5 指标排序与混合命名
- 指标表 metrics_comparison_* 按 MAE 升序排序。
- 混合模型命名随选择指标变化：hybrid_segment_best_mae 或 hybrid_segment_best_smape_image。  
见 [run_compare_prices.py](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L762-L797)。

## 5. 输出产物结构
每个结果目录包含：
- metrics_comparison_*.csv：模型指标对比
- predictions_*_*.csv：预测与真实值
- segment_model_selection_*.csv：分段最佳模型选择
- retrain_log_*.csv：滚动重训记录
- forecast_compare_*.png：可视化对比图

产物解读要点：
- metrics_comparison_*：比较模型整体指标，用于选择全局最佳模型。
- predictions_*：包含时间、目标、实际值、分段与预测，便于绘图与误差分析。
- segment_model_selection_*：展示谷/平/峰的最优模型选择。

参考目录：results、results_smoke2、results_smoke_walk2、results_rt_tuned 等。

## 6. 实验结果汇总（关键指标）
以下“最佳模型”按指标文件排序结果取首行。

### 6.1 Day-Ahead 结果
| 结果目录 | 最佳模型 | MAE | RMSE | sMAPE |
| --- | --- | ---:| ---:| ---:|
| results | hybrid_segment_best_mae | 19.6697 | 40.9121 | 0.9032 |
| results_smape_patch_smoke | xgb_chronos_residual | 18.9487 | 21.8106 | 0.9251 |
| results_smoke | hybrid_segment_best_mae | 89.2108 | 133.9659 | - |
| results_smoke2 | xgb_chronos_residual | 35.8205 | 37.4820 | - |
| results_smoke_walk2 | hybrid_segment_best_mae | 20.7785 | 37.0284 | - |
| results_smoke_strict2 | hybrid_segment_best_mae | 89.2108 | 133.9659 | - |

### 6.2 Real-Time 结果
| 结果目录 | 最佳模型 | MAE | RMSE | sMAPE |
| --- | --- | ---:| ---:| ---:|
| results | chronos_xgb_residual | 46.4553 | 86.7612 | 0.7803 |
| results_smoke2 | xgb_chronos_residual | 91.6327 | 132.4354 | - |
| results_rt_tune_smoke | chronos_xgb_residual | 61.1625 | 97.9917 | 0.5881 |
| results_rt_tuned | chronos_xgb_residual | 46.5627 | 86.6510 | 0.7797 |
| results_rt_tuned_aggr | chronos_xgb_residual | 46.7123 | 86.8692 | 0.7793 |

### 6.3 Spread 结果位置
- 结果目录：results_spread_smoke、results_spread_smoke2、results_fi_all_quick
- 指标文件：metrics_comparison_spread_*（示例：[metrics_comparison_spread_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick/metrics_comparison_spread_day_ahead_price.csv)）

**指标来源**  
见各目录的 metrics_comparison_* 文件，例如：
- [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results/metrics_comparison_day_ahead_price.csv)
- [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results/metrics_comparison_real_time_price.csv)
- [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned/metrics_comparison_real_time_price.csv)

## 7. 复用与复现实验
建议最小复用命令：
```
py run_compare_prices.py --train_csv guangdong_2025_complete.csv --valid_csv guangdong_2026_complete.csv --targets day_ahead_price,real_time_price --out_dir results
```

结果目录与实验映射见 [result_experiment_mapping.md](file:///c:/Users/Joey/Desktop/xmo/result_experiment_mapping.md)。

可调关键参数：
- --xgb_history_mode：walk_forward / strict_recursive
- --hybrid_select_metric：MAE / sMAPE
- --rt_peak_retrain_steps：实时价峰段重训频率
- --rt_peak_*：峰段 XGB 超参增强

## 8. 结论要点
- 残差融合模型普遍优于单模型，尤其在 day_ahead 上效果稳定。
- real_time 的 MAPE 极易被低值放大，建议用 MAE / RMSE / sMAPE 作为主指标。
- walk_forward 在 day_ahead 的效果明显优于 strict_recursive。

## 9. 提升点与下一步清单
### 9.1 评估与验证
- 增加滚动窗口验证或多折时间切分，降低单一年度切分偏差。
- 对 real_time 增加稳定性指标汇总，单列展示 MAE/RMSE/sMAPE 的对比排序。
- 针对节假日、极端负荷日、异常天气日单独评估子集表现。

### 9.2 数据质量与诊断
- 增加缺失率、极值分布、异常点统计与处理说明。
- 对 MAPE 极大值区间做标记与解释，避免误判模型效果。

### 9.3 模型与特征
- 引入分位数预测或不确定性区间（Chronos 支持分位数输出）。

### 9.4 复现与对比
- 增加跨年或跨区域验证，检验模型外推性。
- 固化实验配置快照（参数、数据版本、模型版本），保证可追溯。
