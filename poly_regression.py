import csv
import numpy as np
import matplotlib.pyplot as plt
from core import Value

# 1. Đọc dữ liệu từ CSV
filepath = 'Middle_East_Economic_Data_1990_2024_with_Oil.csv'

x_data_raw = []
y_data_raw = []

with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Làm ví dụ minh họa trên Saudi Arabia
        if row['Country_Code'] == 'SAU':
            try:
                x_val = float(row['Brent_Oil_Price_USD_per_barrel'])
                y_val = float(row['GDP_growth_annual_pct'])
                x_data_raw.append(x_val)
                y_data_raw.append(y_val)
            except ValueError:
                pass

# 2. Chuẩn hóa dữ liệu (Z-Score Normalization)
x_data_np = np.array(x_data_raw)
y_data_np = np.array(y_data_raw)

x_mean, x_std = x_data_np.mean(), x_data_np.std()
y_mean, y_std = y_data_np.mean(), y_data_np.std()

x_data = (x_data_np - x_mean) / x_std
y_data = (y_data_np - y_mean) / y_std

# 3. Khởi tạo tham số mô hình đa thức bậc 2: y = a*x^2 + b*x + c
a = Value(np.random.randn())
b = Value(np.random.randn())
c = Value(np.random.randn())

# Tham số Adam Optimizer
m_a, v_a = 0.0, 0.0
m_b, v_b = 0.0, 0.0
m_c, v_c = 0.0, 0.0
lr = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8

epochs = 100
n_samples = len(x_data)

print(f"Bắt đầu huấn luyện Polynomial Regression bậc 2...")

loss_history = []

for epoch in range(1, epochs + 1):
    loss = Value(0.0)
    
    # Tính Loss (MSE)
    for i in range(n_samples):
        x_i = Value(x_data[i])
        y_target = Value(y_data[i])
        
        # Phương trình: y_pred = a*(x^2) + b*x + c
        y_pred = (a * (x_i**2)) + (b * x_i) + c
        
        # Sai số bình phương
        diff = y_pred - y_target
        sq_diff = diff**2
        loss = loss + sq_diff
        
    # Tính trung bình loss
    loss = loss * Value(1.0 / n_samples)
    
    # Zero gradients
    a.grad, b.grad, c.grad = 0.0, 0.0, 0.0
    
    # Backpropagation
    loss.backward()
    
    # --- Cập nhật a theo Adam ---
    m_a = beta1 * m_a + (1-beta1)*a.grad
    v_a = beta2 * v_a + (1-beta2)*a.grad**2
    m_a_hat = m_a / (1-beta1**epoch)
    v_a_hat = v_a / (1-beta2**epoch)
    a.data -= lr * m_a_hat / (np.sqrt(v_a_hat) + eps)
    
    # --- Cập nhật b theo Adam ---
    m_b = beta1 * m_b + (1-beta1)*b.grad
    v_b = beta2 * v_b + (1-beta2)*b.grad**2
    m_b_hat = m_b / (1-beta1**epoch)
    v_b_hat = v_b / (1-beta2**epoch)
    b.data -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    # --- Cập nhật c theo Adam ---
    m_c = beta1 * m_c + (1-beta1)*c.grad
    v_c = beta2 * v_c + (1-beta2)*c.grad**2
    m_c_hat = m_c / (1-beta1**epoch)
    v_c_hat = v_c / (1-beta2**epoch)
    c.data -= lr * m_c_hat / (np.sqrt(v_c_hat) + eps)
    
    loss_history.append(loss.data)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3}: Loss = {loss.data:.4f}, a={a.data:.3f}, b={b.data:.3f}, c={c.data:.3f}")

# 4. Vẽ biểu đồ 
plt.figure(figsize=(12, 5))

# Biểu đồ 1: Đường cong Polynomial
plt.subplot(1, 2, 1)
plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='Dữ liệu thực tế (SAU)')

# Tạo dải điểm X mượt để vẽ đường cong
x_line = np.linspace(min(x_data)-0.5, max(x_data)+0.5, 100)
y_line = a.data * (x_line**2) + b.data * x_line + c.data

plt.plot(x_line, y_line, color='red', linewidth=2, label='Đường vòng (Adam)')
plt.title('Hồi quy Đa thức (Polynomial): Giá Dầu vs Tăng trưởng GDP')
plt.xlabel('Giá Dầu Brent (Z-Score)')
plt.ylabel('Tăng trưởng GDP (%) (Z-Score)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 2: Hàm loss
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), loss_history, 'g-', linewidth=2)
plt.title('Hàm Loss hội tụ')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
