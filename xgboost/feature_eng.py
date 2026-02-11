import pandas as pd


def feature_engineering(df):
    print(">>> [模块二] 正在构建特征...")
    data = df.copy()

    # 1. 时间特征
    data['hour'] = data['datetime'].dt.hour
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['quarter'] = data['datetime'].dt.quarter

    # 2. 滞后特征
    data['lag_load_1'] = data['负荷实际值'].shift(1)  # 15分钟前
    data['lag_load_96'] = data['负荷实际值'].shift(96)  # 24小时前

    if '实时电价' in data.columns:
        data['lag_price_1'] = data['实时电价'].shift(1)
        data['lag_price_96'] = data['实时电价'].shift(96)

    # 3. 删除空行
    data = data.dropna()
    print(f">>> [模块二] 特征构建完成，有效数据: {len(data)} 行")
    return data