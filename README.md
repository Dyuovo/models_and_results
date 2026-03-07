# dyuovo
电价预测实验与对比框架，基于 Chronos-2 与分段 XGBoost，支持残差融合、分段选择、滚动重训，以及日度 spread 实验。

## 功能概览
- 15 分钟粒度电价预测对比（day_ahead / real_time）
- 分段（谷/平/峰）建模与混合选择
- Chronos ↔ XGBoost 双向残差融合
- 实时价峰段增强重训策略
- 日度 spread（峰均价 - 谷均价）实验
- 统一输出预测、指标、重训日志、对比图

## 目录结构
- run_compare_prices.py：主实验脚本
- guangdong_2025_complete.csv：训练集
- guangdong_2026_complete.csv：验证集
- results*/：实验输出目录
- experiment_report.md：实验复用报告
- model_design_differences.txt：模型设计差异说明
- final_results_summary.csv：最终结果总表
- avg_intraday_prices.xlsx：平均分时电价图

## 环境依赖
- Python 3.10+
- pandas, numpy, matplotlib, xgboost, chronos-forecasting
- chinese_calendar

安装示例：
```bash
py -m pip install pandas numpy matplotlib xgboost chronos-forecasting chinese_calendar
```

## 快速开始
```bash
py run_compare_prices.py --train_csv guangdong_2025_complete.csv --valid_csv guangdong_2026_complete.csv --targets day_ahead_price,real_time_price --out_dir results
```

## 结果目录速查
- 全量主实验输出在 [results](file:///c:/Users/Joey/Desktop/xmo/results)
- 实时价峰段调参输出在 [results_rt_tuned](file:///c:/Users/Joey/Desktop/xmo/results_rt_tuned)
- 特征重要性全量输出在 [results_fi_all_quick](file:///c:/Users/Joey/Desktop/xmo/results_fi_all_quick)
- 目录与实验映射见 [result_experiment_mapping.md](file:///c:/Users/Joey/Desktop/xmo/result_experiment_mapping.md)

## 常用参数
- --xgb_history_mode：walk_forward / strict_recursive
- --hybrid_select_metric：MAE / sMAPE_image
- --rt_peak_retrain_steps：实时价峰段重训频率
- --rt_peak_*：峰段 XGB 超参增强
- --enable_spread_experiment：开启日度 spread 实验
- --spread_targets：spread 目标列
- --spread_lags：日频滞后
- --spread_retrain_steps：日频重训间隔

## 输出说明
每个结果目录包含：
- metrics_comparison_*.csv：模型指标对比
- predictions_*_*.csv：预测与真实值
- segment_model_selection_*.csv：分段最佳模型选择
- retrain_log_*.csv：滚动重训记录
- forecast_compare_*.png：可视化对比图

指标文件按 MAE 排序，混合模型命名随选择指标变化（hybrid_segment_best_mae 或 hybrid_segment_best_smape_image）。实现见 [run_compare_prices.py](file:///c:/Users/Joey/Desktop/xmo/run_compare_prices.py#L762-L797)。

## 文档与报告
- 实验复用报告：[experiment_report.md](file:///c:/Users/Joey/Desktop/xmo/experiment_report.md)
- 结果目录对应实验说明：[result_experiment_mapping.md](file:///c:/Users/Joey/Desktop/xmo/result_experiment_mapping.md)
- 模型差异说明：[model_design_differences.txt](file:///c:/Users/Joey/Desktop/xmo/model_design_differences.txt)
