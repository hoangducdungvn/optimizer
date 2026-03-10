import csv
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

filepath = 'Middle_East_Economic_Data_1990_2024_with_Oil.csv'

X_raw = []
y_raw = []

with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            # Giá Dầu, FDI, Xuất Khẩu
            x1 = float(row['Brent_Oil_Price_USD_per_barrel'])
            x2 = float(row['FDI_net_inflows_pct_GDP'])
            x3 = float(row['Exports_pct_GDP'])
            
            # Dự đoán GDP
            y_val = float(row['GDP_growth_annual_pct'])
            
            X_raw.append([x1, x2, x3])
            y_raw.append(y_val)
        except ValueError:
            pass 

X = np.array(X_raw)
y = np.array(y_raw)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Bắt đầu huấn luyện XGBoost Regressor trên {len(X_train)} mẫu...")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)


xgb_model.fit(X_train, y_train)

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"MSE Tập Huấn luyện: {mse_train:.4f}  |  R2: {r2_train:.4f}")
print(f"MSE Tập Kiểm thử:   {mse_test:.4f}  |  R2: {r2_test:.4f}")
feature_names = ['Oil Price', 'FDI/GDP', 'Exports/GDP']
importances = xgb_model.feature_importances_

print("\n--- Độ quan trọng của đặc trưng ---")
for f_name, imp in zip(feature_names, importances):
    print(f"{f_name}: {imp:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, edgecolors='k')

min_val = min(min(y_test), min(y_pred_test))
max_val = max(max(y_test), max(y_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Dự đoán hoàn hảo')

plt.title('Dự đoán vs Thực tế (XGBoost) - Tập Test')
plt.xlabel('Tăng trưởng GDP thực tế (%)')
plt.ylabel('Tăng trưởng GDP dự đoán (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
sorted_idx = np.argsort(importances)
pos = np.arange(len(sorted_idx)) + 0.5
plt.barh(pos, importances[sorted_idx], align='center', color='green', alpha=0.7)
plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.title('Mức độ quan trọng của Đặc trưng (XGBoost)')
plt.xlabel('Trọng số tầm quan trọng')

plt.tight_layout()
plt.show()
