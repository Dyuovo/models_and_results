import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def train_predict_plot(df, target_col, feature_cols, split_date, task_name="Task"):
    print(f"\n>>> [模块三] 正在训练: {task_name} ...")

    # 切分
    train = df[df['datetime'] < split_date].copy()
    valid = df[df['datetime'] >= split_date].copy()

    if len(valid) == 0:
        print("错误：验证集为空！")
        return None, None

    print(f"验证集范围: {valid['datetime'].min()} -> {valid['datetime'].max()}")

    # 训练
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        train[feature_cols], train[target_col],
        eval_set=[(train[feature_cols], train[target_col]), (valid[feature_cols], valid[target_col])],
        verbose=False
    )

    # 预测
    preds = model.predict(valid[feature_cols])
    valid['Prediction'] = preds

    # 评估 (修复了 squared 参数问题)
    rmse = np.sqrt(mean_squared_error(valid[target_col], preds))
    mape = mean_absolute_percentage_error(valid[target_col], preds)

    print(f"--- {task_name} 结果 ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")

    # 画图
    plt.figure(figsize=(15, 6))
    plt.plot(valid['datetime'], valid[target_col], label='实际值', color='black', alpha=0.6)
    plt.plot(valid['datetime'], valid['Prediction'], label='XGBoost预测', color='red', alpha=0.8)

    # 画对比线
    d1_col = '负荷预测D-1' if task_name == '负荷预测' else '日前电价'
    if d1_col in valid.columns:
        plt.plot(valid['datetime'], valid[d1_col], label=f'原始预测 ({d1_col})', color='blue', linestyle='--',
                 alpha=0.5)

    plt.title(f'{task_name} (MAPE: {mape:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model, valid