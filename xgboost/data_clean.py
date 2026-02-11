import pandas as pd
import numpy as np


def prepare_data(file_path):
    print(f">>> [模块一] 正在读取: {file_path}")
    df = pd.read_csv(file_path)

    # 1. 修正 24:00
    df['日期'] = pd.to_datetime(df['日期'])
    mask_24 = df['时间'] == '24:00'
    if mask_24.any():
        print(f"检测到 {mask_24.sum()} 行包含 '24:00'，正在修正...")
        df.loc[mask_24, '日期'] += pd.Timedelta(days=1)
        df.loc[mask_24, '时间'] = '00:00'

    # 2. 合并时间
    df['datetime'] = pd.to_datetime(df['日期'].dt.strftime('%Y-%m-%d') + ' ' + df['时间'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 3. 清洗数值
    def clean_value(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            x = x.strip()
            if x in ['-', '', 'NULL']: return np.nan
            if '%' in x: return float(x.replace('%', '')) / 100.0
        return pd.to_numeric(x, errors='coerce')

    for col in df.columns:
        if col not in ['日期', '时间', 'datetime']:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(clean_value)

    # 4. 填充
    df = df.ffill().bfill()
    print(">>> [模块一] 数据清洗完成。")
    return df