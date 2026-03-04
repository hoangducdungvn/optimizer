import numpy as np
import matplotlib.pyplot as plt
from core import Value

# 1. Tạo dữ liệu để vẽ mặt cắt (contour plot) của đáy bát tĩnh: loss = x^2 + y^2
x_grid = np.linspace(-12, 12, 100)
y_grid = np.linspace(-12, 12, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = X**2 + Y**2

# Vẽ background mặt cắt
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
plt.plot(0, 0, 'r*', markersize=15, label='Đáy bát (0, 0)') # Điểm đích

# 2. Setup Optimizer giống bài cũ
x = Value(10.0)
y = Value(10.0)

m_x, v_x = 0.0, 0.0
m_y, v_y = 0.0, 0.0
lr = 0.5
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Lưu lại đường đi
path_x, path_y = [x.data], [y.data]

# 3. Chạy quá trình tối ưu
for epoch in range(1, 101):
    loss = x*x + y*y
    
    x.grad = 0.0
    y.grad = 0.0
    loss.backward()
    
    # Cập nhật x
    m_x = beta1 * m_x + (1-beta1)*x.grad
    v_x = beta2 * v_x + (1-beta2)*x.grad**2
    m_x_hat = m_x / (1-beta1**epoch)
    v_x_hat = v_x / (1-beta2**epoch)
    x.data -= lr * m_x_hat / (np.sqrt(v_x_hat) + eps)
    
    # Cập nhật y
    m_y = beta1 * m_y + (1-beta1)*y.grad
    v_y = beta2 * v_y + (1-beta2)*y.grad**2
    m_y_hat = m_y / (1-beta1**epoch)
    v_y_hat = v_y / (1-beta2**epoch)
    y.data -= lr * m_y_hat / (np.sqrt(v_y_hat) + eps)
    
    # Ghi lại vị trí mới
    path_x.append(x.data)
    path_y.append(y.data)

# 4. Hiển thị thông số đồ thị
plt.plot(path_x, path_y, 'ro-', markersize=4, linewidth=1.5, label='Quỹ đạo Adam')

plt.title('Minh họa Adam Optimizer hướng viên bi về đáy bát', fontsize=14)
plt.xlabel('x (Tọa độ X)')
plt.ylabel('y (Tọa độ Y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Mở cửa sổ đồ họa
plt.show()
