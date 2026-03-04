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
        # Lấy dữ liệu của Saudi Arabia (SAU) làm ví dụ minh họa
        if row['Country_Code'] == 'SAU':
            try:
                # x: Giá dầu Brent, y: Tăng trưởng GDP
                x_val = float(row['Brent_Oil_Price_USD_per_barrel'])
                y_val = float(row['GDP_growth_annual_pct'])
                x_data_raw.append(x_val)
                y_data_raw.append(y_val)
            except ValueError:
                pass # Bỏ qua các hàng thiếu dữ liệu

# 2. Chuẩn hóa dữ liệu (Normalization) giúp Adam hội tụ tốt hơn và tránh bị tràn số
x_data_np = np.array(x_data_raw)
y_data_np = np.array(y_data_raw)

x_mean, x_std = x_data_np.mean(), x_data_np.std()
y_mean, y_std = y_data_np.mean(), y_data_np.std()

x_data = (x_data_np - x_mean) / x_std
y_data = (y_data_np - y_mean) / y_std

# 3. Khởi tạo tham số mô hình (weights và bias) một cách ngẫu nhiên
w = Value(np.random.randn())
b = Value(np.random.randn())

# Tham số Adam
m_w, v_w = 0.0, 0.0
m_b, v_b = 0.0, 0.0
lr = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8

epochs = 100
n_samples = len(x_data)

print(f"Bắt đầu huấn luyện Linear Regression trên {n_samples} mẫu dữ liệu...")

# Lưu lịch sử loss để vẽ biểu đồ
loss_history = []

for epoch in range(1, epochs + 1):
    # Khởi tạo loss ban đầu = 0
    loss = Value(0.0)
    
    # Tính Loss (Mean Squared Error - Trung bình bình phương sai số)
    for i in range(n_samples):
        # y_pred = w * x + b
        x_i = Value(x_data[i])
        y_pred = w * x_i + b
        
        # Vì class Value trong core.py hiện tại CHƯA định nghĩa hàm __sub__ (trừ), 
        # nên ta phải đổi y_pred - y_target thành y_pred + (y_target * -1.0)
        y_target = Value(y_data[i] * -1.0)
        diff = y_pred + y_target
        
        # Tính bình phương sai số (Value cũng chưa có __pow__ nên ta nhân 2 lần diff)
        sq_diff = diff * diff
        loss = loss + sq_diff
    
    # Tính giá trị trung bình (Mean)
    # Vì core.py chưa định nghĩa __truediv__ (chia), ta biến đổi thành phép nhân phân số
    loss = loss * Value(1.0 / n_samples)
    
    # Gán lại zero_grad
    w.grad = 0.0
    b.grad = 0.0
    
    # Gọi hàm Backpropagation tự code
    loss.backward()
    
    # Cập nhật w
    m_w = beta1 * m_w + (1-beta1)*w.grad
    v_w = beta2 * v_w + (1-beta2)*w.grad**2
    m_w_hat = m_w / (1-beta1**epoch)
    v_w_hat = v_w / (1-beta2**epoch)
    w.data -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
    
    # Cập nhật b
    m_b = beta1 * m_b + (1-beta1)*b.grad
    v_b = beta2 * v_b + (1-beta2)*b.grad**2
    m_b_hat = m_b / (1-beta1**epoch)
    v_b_hat = v_b / (1-beta2**epoch)
    b.data -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    
    loss_history.append(loss.data)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3}: Loss = {loss.data:.4f}, w = {w.data:.4f}, b = {b.data:.4f}")

print("\nĐã huấn luyện xong! Mở cửa sổ đồ thị...")

# 4. Vẽ kết quả minh họa
plt.figure(figsize=(12, 5))

# Biểu đồ 1: Đường hồi quy
plt.subplot(1, 2, 1)
plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='Dữ liệu thực tế (SAU)')
x_line = np.linspace(min(x_data), max(x_data), 100)
y_line = w.data * x_line + b.data
plt.plot(x_line, y_line, color='red', linewidth=2, label='Đường hồi quy Adam')
plt.title('Hồi quy tuyến: Giá Dầu vs Tăng trưởng GDP\n(Đã Chuẩn hóa)')
plt.xlabel('Giá Dầu Brent (Z-Score)')
plt.ylabel('Tăng trưởng GDP (%) (Z-Score)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 2: Sự giảm dần của Loss Function
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), loss_history, 'g-', linewidth=2)
plt.title('Hàm Loss hội tụ qua từng Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
