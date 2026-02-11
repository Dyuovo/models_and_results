电力市场负荷与电价偏差修正系统 (Power Market Load & Price Bias Correction)
本项目利用 XGBoost 机器学习算法，针对电力市场中的日前预测（Day-ahead Forecast）进行实时偏差修正。通过整合历史负荷、气象预测、日前价格及时间周期特征，显著降低了负荷与实时电价的预测误差，为电力交易策略提供数据支撑。

📊 核心成果
通过对山西电力市场（示例数据）的测试，模型表现如下：

负荷预测误差 (MAPE)：从原始日前的 2.40% 降低至 0.78%。

电价预测精度 (RMSE)：较日前报价指导提升了约 32%，有效捕捉价格尖峰。
<img width="928" height="376" alt="image" src="https://github.com/user-attachments/assets/f5046a2d-a8ad-4e97-8341-c491e867f653" />



🛠️ 技术栈
语言：Python 3.14

核心库：XGBoost, Pandas, Scikit-learn, Matplotlib

算法：Gradient Boosting Decision Trees (GBDT)
