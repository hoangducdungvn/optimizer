import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# 2. Chuẩn hóa dữ liệu (Normalization) giúp thuật toán hội tụ tốt hơn
x_data = np.array(x_data_raw)
y_data = np.array(y_data_raw)

x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()

x_data_norm = (x_data - x_mean) / x_std
y_data_norm = (y_data - y_mean) / y_std

n_samples = len(x_data_norm)

# 3. Định nghĩa Hàm mục tiêu (Loss) và Đạo hàm (Gradient)
def loss_and_gradient(params):
    """
    Hàm này được thư viện L-BFGS gọi liên tục.
    params: Mảng 1D chứa [w, b]
    Trả về: (loss_value, gradient_array)
    """
    w, b = params
    
    # Tính dự đoán (Vectorized bằng Numpy siêu nhanh)
    y_pred = w * x_data_norm + b
    
    # Tính Loss (Mean Squared Error)
    diff = y_pred - y_data_norm
    loss = (1.0 / n_samples) * np.sum(diff**2)
    
    # Tính Vector Đạo hàm (Gradient) theo công thức toán học ma trận
    grad_w = (2.0 / n_samples) * np.sum(diff * x_data_norm)
    grad_b = (2.0 / n_samples) * np.sum(diff)
    
    # Trả về Loss vô hướng và Mảng Gradient [grad_w, grad_b]
    return loss, np.array([grad_w, grad_b])

print(f"Bắt đầu huấn luyện Linear Regression bằng L-BFGS trên {n_samples} mẫu dữ liệu...")

# Khởi tạo tham số ngẫu nhiên
initial_params = np.random.randn(2)
print(f"Tham số khởi tạo ngẫu nhiên ban đầu: w = {initial_params[0]:.4f}, b = {initial_params[1]:.4f}")

# Cấu trúc lưu lại lịch sử loss qua từng vòng lặp (giống callback của Keras)
loss_history = []

def callbackF(params):
    loss, _ = loss_and_gradient(params)
    loss_history.append(loss)
    print(f"Vòng lặp (Iteration) {len(loss_history)}: Loss = {loss:.6f}, w = {params[0]:.4f}, b = {params[1]:.4f}")

# 4. GỌI SIÊU THUẬT TOÁN L-BFGS 🚀
# Lưu ý: jac=True nghĩa là hàm mục tiêu của chúng ta trả về sẵn đạo hàm, SciPy sẽ không phải tự tính xấp xỉ đạo hàm.
res = minimize(loss_and_gradient, initial_params, method='L-BFGS-B', jac=True, callback=callbackF)

# Lấy kết quả tốt nhất
w_opt, b_opt = res.x
final_loss = res.fun

print("\n--- KẾT QUẢ TỐI ƯU ---")
print(f"L-BFGS đã hội tụ BẮN TỐC ĐỘ chỉ sau {res.nit} vòng lặp (Iterations)!")
print(f"Loss tối thiểu đạt được: {final_loss:.6f}")
print(f"Tham số tối ưu cuối cùng: w = {w_opt:.4f}, b = {b_opt:.4f}")
print(f"Trạng thái thành công: {res.success} ({res.message})")

# Đảm bảo loss history có chứa ít nhất giá trị gốc và giá trị cuối cùng để vẽ biểu đồ cho mượt
if len(loss_history) == 0:
	loss_history.append(loss_and_gradient(initial_params)[0])
	loss_history.append(final_loss)

# 5. Vẽ kết quả minh họa so sánh
plt.figure(figsize=(12, 5))

# Biểu đồ 1: Đường hồi quy
plt.subplot(1, 2, 1)
plt.scatter(x_data_norm, y_data_norm, color='blue', alpha=0.6, label='Dữ liệu phân tán (SAU)')
x_line = np.linspace(min(x_data_norm), max(x_data_norm), 100)
y_line = w_opt * x_line + b_opt
plt.plot(x_line, y_line, color='red', linewidth=2, label='Đường hồi quy L-BFGS')
plt.title('Hồi quy tuyến tính: Giá Dầu vs Tăng trưởng GDP\n(Tối ưu bằng đạo hàm bậc 2)')
plt.xlabel('Giá Dầu Brent (Z-Score)')
plt.ylabel('Tăng trưởng GDP (%) (Z-Score)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Biểu đồ 2: Sự hội tụ chớp nhoáng của Loss Function
plt.subplot(1, 2, 2)
# Thêm vòng lặp số 0 (chưa tinh chỉnh) để đồ thị đẹp hơn
full_loss_history = [loss_and_gradient(initial_params)[0]] + loss_history
plt.plot(range(0, len(full_loss_history)), full_loss_history, 'g-o', linewidth=2, markersize=8)
plt.title('Hàm Loss L-BFGS (Hội tụ cấp số nhân)')
plt.xlabel('Vòng lặp (Iteration)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(range(0, len(full_loss_history))) # Chỉnh x-axis tick nguyên vẹn vì số vòng lặp rất ít
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
