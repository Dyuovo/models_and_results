# 结果目录与实验对应表

本文档用于说明各个 results* 目录对应的实验类型，帮助读者快速定位产物与实验关系。说明主要依据目录命名与目录内产物文件推断。

## 主实验与对比

| 结果目录 | 实验说明 | 关键产物示例 |
| --- | --- | --- |
| [results](file:///c:/Users/Joey/Desktop/xmo/results) | 基线全量对比（day_ahead_price + real_time_price），分段混合按 MAE 选优 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results/metrics_comparison_day_ahead_price.csv), [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results/metrics_comparison_real_time_price.csv), predictions_*_hybrid_segment_best_mae.csv |

## Smoke / 调试实验

| 结果目录 | 实验说明 | 关键产物示例 |
| --- | --- | --- |
| [results_smoke](file:///c:/Users/Joey/Desktop/xmo/results_smoke) | day_ahead 的 smoke 快速验证 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smoke/metrics_comparison_day_ahead_price.csv) |
| [results_smoke2](file:///c:/Users/Joey/Desktop/xmo/results_smoke2) | day_ahead + real_time 的 smoke 快速验证 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smoke2/metrics_comparison_day_ahead_price.csv), [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smoke2/metrics_comparison_real_time_price.csv) |
| [results_smoke_walk2](file:///c:/Users/Joey/Desktop/xmo/results_smoke_walk2) | walk_forward 模式的 smoke 验证 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smoke_walk2/metrics_comparison_day_ahead_price.csv) |
| [results_smoke_strict2](file:///c:/Users/Joey/Desktop/xmo/results_smoke_strict2) | strict_recursive 模式的 smoke 验证 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smoke_strict2/metrics_comparison_day_ahead_price.csv) |
| [results_smape_patch_smoke](file:///c:/Users/Joey/Desktop/xmo/results_smape_patch_smoke) | sMAPE 指标修正后的 smoke 验证 | [metrics_comparison_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_smape_patch_smoke/metrics_comparison_day_ahead_price.csv) |

## 实时价峰段调参

| 结果目录 | 实验说明 | 关键产物示例 |
| --- | --- | --- |
| [results_rt_tune_smoke](file:///c:/Users/Joey/Desktop/xmo/results_rt_tune_smoke) | real_time_price 峰段调参的 smoke 验证 | [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_rt_tune_smoke/metrics_comparison_real_time_price.csv) |
| [results_rt_tuned](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned) | real_time_price 峰段调参全量实验 | [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned/metrics_comparison_real_time_price.csv) |
| [results_rt_tuned_aggr](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned_aggr) | real_time_price 峰段调参的汇总输出 | [metrics_comparison_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned_aggr/metrics_comparison_real_time_price.csv) |

## Spread 实验

| 结果目录 | 实验说明 | 关键产物示例 |
| --- | --- | --- |
| [results_spread_smoke](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke) | day_ahead + 日度 spread 目标的 smoke 验证 | [spread_daily_series_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke/spread_daily_series_day_ahead_price.csv) |
| [results_spread_smoke2](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke2) | day_ahead + 日度 spread 目标的扩展 smoke（含 spread 预测图与 hybrid 选择日志） | [forecast_compare_spread_day_ahead_price.png](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke2/forecast_compare_spread_day_ahead_price.png), [hybrid_selection_log_spread_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_spread_smoke2/hybrid_selection_log_spread_day_ahead_price.csv) |

## 特征重要性实验

| 结果目录 | 实验说明 | 关键产物示例 |
| --- | --- | --- |
| [results_fi_smoke](file:///c:/Users/Joey/Desktop/xmo/results_fi_smoke) | 特征重要性 smoke（含 day_ahead 与 spread） | [feature_importance_day_ahead_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_fi_smoke/feature_importance_day_ahead_price.csv) |
| [results_fi_all_quick](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick) | 特征重要性全量 quick（含 day_ahead、real_time、spread） | [feature_importance_detail_real_time_price.csv](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick/feature_importance_detail_real_time_price.csv) |
| [results_fi_all_quick_viz](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick_viz) | 特征重要性可视化输出（由 analyze_feature_importance.py 生成） | [analysis_report.md](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick_viz/analysis_report.md) |

## 说明与参考

- 目录中常见的输出结构见 [README.md](file:///c:/Users/Joey/Desktop/xmo/README.md) 与 [experiment_report.md](file:///c:/Users/Joey/Desktop/xmo/experiment_report.md)。
- 目录命名与对应实验类型若需要更精确，可在后续将运行命令或参数快照写入该目录以增强可追溯性。
