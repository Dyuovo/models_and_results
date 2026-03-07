# Feature Importance Analysis Report

## Dataset Overview
- Summary rows: 900
- Detail rows: 17,964
- Targets: day_ahead_price, real_time_price, spread_day_ahead_price, spread_real_time_price
- Schemes: chronos_xgb_residual_spread_xgb_part, chronos_xgb_residual_xgb_part, xgb_only, xgb_only_spread, xgb_oof_train, xgb_oof_train_spread

## Visual Outputs
- top_features_by_target.png: Top-N features per target (mean importance).
- feature_family_share_by_target.png: Family share stacked bars.
- heatmap_feature_scheme_<target>.png: Feature vs scheme heatmaps (Top-N).

## Top Features by Target
### day_ahead_price
- lag_1 (lag): mean_importance=0.471515
- roll_mean_4 (roll_mean): mean_importance=0.285501
- lag_2 (lag): mean_importance=0.031874
- sin_slot96 (calendar_time): mean_importance=0.026557
- diff_1 (diff): mean_importance=0.014903
- lag_96 (lag): mean_importance=0.014627
- slot96 (calendar_time): mean_importance=0.013255
- minute (calendar_time): mean_importance=0.012601
- roll_mean_16 (roll_mean): mean_importance=0.011548
- diff_96 (diff): mean_importance=0.011520

### real_time_price
- lag_1 (lag): mean_importance=0.300541
- roll_mean_4 (roll_mean): mean_importance=0.252444
- holiday_name_Dragon Boat Festival (holiday_name): mean_importance=0.067247
- roll_mean_16 (roll_mean): mean_importance=0.044184
- roll_mean_96 (roll_mean): mean_importance=0.019814
- is_holiday (holiday_binary): mean_importance=0.018632
- roll_mean_672 (roll_mean): mean_importance=0.018057
- lag_2 (lag): mean_importance=0.017086
- slot96 (calendar_time): mean_importance=0.016170
- diff_96 (diff): mean_importance=0.016103

### spread_day_ahead_price
- is_holiday (holiday_binary): mean_importance=0.257841
- is_workday (holiday_binary): mean_importance=0.179789
- diff_7 (diff): mean_importance=0.052879
- roll_mean_7 (roll_mean): mean_importance=0.052690
- dayofweek (calendar_time): mean_importance=0.051858
- roll_mean_14 (roll_mean): mean_importance=0.051388
- month (calendar_time): mean_importance=0.049051
- after_holiday_1d (holiday_binary): mean_importance=0.044239
- roll_mean_3 (roll_mean): mean_importance=0.042041
- lag_14 (lag): mean_importance=0.041687

### spread_real_time_price
- roll_mean_3 (roll_mean): mean_importance=0.119689
- roll_mean_14 (roll_mean): mean_importance=0.096944
- lag_28 (lag): mean_importance=0.083160
- after_holiday_1d (holiday_binary): mean_importance=0.079067
- roll_std_7 (roll_std): mean_importance=0.076587
- is_holiday (holiday_binary): mean_importance=0.068513
- diff_7 (diff): mean_importance=0.066595
- dayofweek (calendar_time): mean_importance=0.065732
- month (calendar_time): mean_importance=0.064806
- roll_mean_7 (roll_mean): mean_importance=0.062025

## Feature Family Share (Top 5 each target)
### day_ahead_price
- lag: 43.93%
- roll_mean: 38.15%
- diff: 6.37%
- calendar_time: 5.49%
- roll_std: 4.25%

### real_time_price
- roll_mean: 39.65%
- lag: 29.68%
- diff: 7.13%
- roll_std: 6.52%
- holiday_name: 6.35%

### spread_day_ahead_price
- holiday_binary: 35.80%
- roll_mean: 16.85%
- roll_std: 14.31%
- lag: 12.02%
- calendar_time: 11.88%

### spread_real_time_price
- roll_mean: 26.65%
- roll_std: 21.97%
- lag: 17.27%
- calendar_time: 13.01%
- holiday_binary: 11.55%

## Stability (By Retrain CV)
- `importance_cv = std / |mean|`; larger means less stable across retrains.
### day_ahead_price
- holiday_name_Tomb-sweeping Day (xgb_oof_train/peak): cv=7.211, mean=0.000004, retrains=52
- holiday_name_Tomb-sweeping Day (xgb_oof_train/valley): cv=7.211, mean=0.000013, retrains=52
- is_weekend (xgb_oof_train/flat): cv=7.211, mean=0.000265, retrains=52
- is_workday (xgb_oof_train/global): cv=7.211, mean=0.000115, retrains=52
- holiday_name_Dragon Boat Festival (xgb_oof_train/valley): cv=7.211, mean=0.000010, retrains=52

### real_time_price
- holiday_name_National Day (xgb_oof_train/flat): cv=7.211, mean=0.000007, retrains=52
- holiday_name_Labour Day (xgb_oof_train/flat): cv=5.058, mean=0.000210, retrains=52
- is_workday (xgb_oof_train/global): cv=3.718, mean=0.003930, retrains=52
- is_weekend (xgb_oof_train/global): cv=3.505, mean=0.001502, retrains=52
- holiday_name_Mid-autumn Festival (xgb_oof_train/peak): cv=3.346, mean=0.000041, retrains=52

### spread_day_ahead_price
- is_weekend (xgb_oof_train_spread/all): cv=2.678, mean=0.006343, retrains=103
- is_workday (xgb_oof_train_spread/all): cv=0.959, mean=0.119745, retrains=103
- month (xgb_oof_train_spread/all): cv=0.941, mean=0.064393, retrains=103
- before_holiday_1d (xgb_oof_train_spread/all): cv=0.717, mean=0.048229, retrains=103
- roll_mean_3 (xgb_oof_train_spread/all): cv=0.604, mean=0.051930, retrains=103

### spread_real_time_price
- is_weekend (xgb_oof_train_spread/all): cv=1.835, mean=0.016451, retrains=99
- is_workday (xgb_oof_train_spread/all): cv=1.815, mean=0.012913, retrains=99
- is_makeup_workday (xgb_oof_train_spread/all): cv=1.469, mean=0.014303, retrains=99
- diff_7 (xgb_oof_train_spread/all): cv=0.638, mean=0.076417, retrains=99
- is_holiday (xgb_oof_train_spread/all): cv=0.637, mean=0.044476, retrains=99
