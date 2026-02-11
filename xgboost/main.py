import pandas as pd
# === 关键：导入同目录下的其他模块 ===
from data_clean import prepare_data
from feature_eng import feature_engineering
from model_train import train_predict_plot

if __name__ == "__main__":
    # 1. 设置文件
    file_name = 'shanxi6.csv'

    try:
        # Step 1: 清洗
        df_clean = prepare_data(file_name)

        # Step 2: 特征
        df_final = feature_engineering(df_clean)

        # Step 3: 训练
        # 自动设定切分日期为最后 7 天
        split_date = df_final['datetime'].max() - pd.Timedelta(days=7)

        # 定义特征
        features_load = ['hour', 'dayofweek', 'lag_load_1', 'lag_load_96', '负荷预测D-1', '风电预测D-1', '光伏预测D-1']

        print("\n=== 开始负荷预测任务 ===")
        model, result = train_predict_plot(
            df_final,
            target_col='负荷实际值',
            feature_cols=features_load,
            split_date=split_date,
            task_name='负荷预测'
        )

    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback

        traceback.print_exc()