# Feature Importance 结果说明（精简版）

本报告基于 `results_fi_all_quick` 的特征重要性输出，图表已做对比度增强与动态范围压缩（分位截断），重点展示 Top-10 特征与特征族占比。

## 核心结论
- **day_ahead_price**：短期滞后与短窗滚动均值占主导，`lag_1`、`roll_mean_4` 权重大幅领先。
- **real_time_price**：短期滞后与滚动均值仍是主力，节假日相关特征开始显著进入前列。
- **spread_day_ahead_price**：日历/节假日特征重要性明显高于 15 分钟主任务。
- **spread_real_time_price**：滚动均值与波动性特征（`roll_std_7`）占比提高。

## 图表说明
- **top_features_by_target.png**：每个目标 Top-10 特征条形图。
- **feature_family_share_by_target.png**：按特征族的贡献占比。
- **heatmap_feature_scheme_<target>.png**：特征 × 方案热力图（Top-10），使用分位数截断提升对比度。

## 阅读建议
- 先看 `top_features_by_target.png` 判断主导特征类型，再看热力图对比不同方案的特征使用差异。
- 若关注稳定性，查看 `feature_importance_stability.csv` 的高 CV 特征（不稳定贡献）。

