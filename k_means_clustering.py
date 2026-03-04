import csv
import numpy as np
import matplotlib.pyplot as plt

filepath = 'Middle_East_Economic_Data_1990_2024_with_Oil.csv'

# 1. Trích xuất dữ liệu: Ta lấy GDP bình quân (GDP_per_capita) và Lạm phát (Inflation) của năm 2023
countries = []
X_raw = []

with open(filepath, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Year'] == '2023':
            try:
                gdp_pc = float(row['GDP_per_capita_current_USD'])
                inflation = float(row['Inflation_consumer_prices_annual_pct'])
                
                countries.append(row['Country_Code'])
                X_raw.append([gdp_pc, inflation])
            except ValueError:
                pass # Bỏ qua QG thiếu số liệu năm này

X = np.array(X_raw)

# 2. Chuẩn hóa dữ liệu (Z-score)
# VÔ CÙNG QUAN TRỌNG TRONG K-MEANS: Vì dải đo của GDP (hàng chục ngàn USD) và Lạm phát (%) lệch nhau quá lớn,
# nếu không chuẩn hóa, khoảng cách Euclidean sẽ hoàn toàn bị chi phối bởi trục GDP.
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 3. Thuật toán K-Means tự code từ Numpy
K = 3  # Chia thành 3 nhóm kinh tế
np.random.seed(42)

# Khởi tạo K tâm cụm (Centroids) ngẫu nhiên từ các điểm dữ liệu có sẵn
random_indices = np.random.choice(len(X_scaled), K, replace=False)
centroids = X_scaled[random_indices]

max_iters = 100
colors = ['red', 'green', 'blue']

print(f"Bắt đầu phân cụm {len(X)} quốc gia thành {K} nhóm...")

for i in range(max_iters):
    # Bước 3.1: Gán mỗi điểm cho cụm có tâm gần nhất (Euclidean Distance)
    # X_scaled có shape (N, 2), centroids có shape (K, 2)
    distances = np.sqrt(((X_scaled[:, np.newaxis] - centroids)**2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    
    # Bước 3.2: Cập nhật lại tâm cụm bằng trung bình tọa độ các điểm trong cụm
    new_centroids = np.array([X_scaled[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k] for k in range(K)])
    
    # Kiểm tra điều kiện dừng: Nếu các tâm cụm không xê dịch nữa
    if np.allclose(centroids, new_centroids):
        print(f"Thuật toán hội tụ sau {i+1} vòng lặp!")
        break
        
    centroids = new_centroids

# 4. Chuyển Centroids về không gian số liệu thật để dễ in ra phân tích
centroids_real = (centroids * X_std) + X_mean

print("\n--- KẾT QUẢ K-MEANS ---")
for k in range(K):
    cluster_points = np.array(countries)[labels == k]
    print(f"Cụm {k+1} (Tâm cụm thực - GDP PC: ${centroids_real[k][0]:.0f}, Lạm phát: {centroids_real[k][1]:.1f}%):")
    print(f"   Các quốc gia: {', '.join(cluster_points)}")


# 5. Vẽ đồ thị
plt.figure(figsize=(10, 6))

for k in range(K):
    # Trích xuất các điểm thuộc cụm k (Vẽ trên không gian CHƯA CHUẨN HÓA cho người dùng dễ đọc trục)
    cluster_data = X[labels == k]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100, c=colors[k], label=f'Cụm {k+1}', alpha=0.7, edgecolors='k')
    
# Plot các tâm cụm (Centroids)
plt.scatter(centroids_real[:, 0], centroids_real[:, 1], s=250, c='yellow', marker='*', edgecolors='black', label='Tâm cụm (Centroids)')

# Gắn nhãn tên quốc gia
for i, txt in enumerate(countries):
    plt.annotate(txt, (X[i, 0], X[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.title('K-Means: Phân cụm các nền kinh tế Trung Đông (2023)\n(Dựa trên Thu nhập bình quân và Lạm phát)', fontsize=14)
plt.xlabel('GDP Bình quân đầu người ($ USD)')
plt.ylabel('Lạm phát hằng năm (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
