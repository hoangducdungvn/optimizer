# Optimizer & Economic Analysis Engine
*(Scroll down for English version)*

## 🇻🇳 Tiếng Việt

Tool tự xây dựng từ đầu (from scratch) để thực hiện tính toán đạo hàm tự động (Autograd), tối ưu hóa bằng thuật toán Adam, và áp dụng vào các bài toán phân tích dữ liệu kinh tế thực tế.

### Cấu trúc files trong dự án:

1. **`core.py` (Autograd Engine & Adam Optimizer):**
   Phần lõi của dự án chứa lớp `Value` - một engine tính toán đạo hàm tự động thu nhỏ tương tự như PyTorch. Nó theo dõi các phép toán (cộng, trừ, nhân, lũy thừa, v.v.) thành một đồ thị và hỗ trợ lan truyền ngược (backpropagation) qua hàm `backward()`.

2. **`visualize.py` (Trực quan hóa Adam):**
   Sử dụng thư viện `matplotlib` hiển thị trực quan cách thuật toán Adam (được định nghĩa trong `core.py`) tối ưu hóa hướng "viên bi" lăn về phía điểm cực tiểu của một mặt cắt đồ thị hàm số 3D (đáy bát).

3. **`poly_regression.py` (Hồi quy Đa thức):**
   Ứng dụng `core.py` vào thực tế Machine Learning: Khớp một đường cong hồi quy đa thức bậc 2 ($y = ax^2 + bx + c$) cho tập dữ liệu thực tế "Giá Dầu Brent" và "Tăng trưởng GDP". Code tự tìm ra các hệ số $a, b, c$ mà không cần viện tới thư viện ML có sẵn như scikit-learn.

4. **`k_means_clustering.py` (Phân cụm K-Means):**
   Xây dựng thuật toán phân cụm K-Means chỉ bằng thư viện ma trận `numpy`. File này phân tích và chia các quốc gia thuộc khu vực Trung Đông (năm 2023) thành 3 nhóm kinh tế dựa vào chỉ số GDP Bình quân đầu người (GDP per capita) và tỷ lệ Lạm phát.

5. **`Middle_East_Economic_Data_1990_2024_with_Oil.csv`:**
   Tập dữ liệu định dạng CSV chứa các thông số kinh tế vĩ mô của các quốc gia khu vực Trung Đông trong nhiều thập kỷ, được dùng làm đầu vào để huấn luyện mô hình.

### Cách chạy:
Cài đặt thư viện: `pip install numpy matplotlib`
Chạy các file script bằng Python để xem đồ thị tương ứng: `python poly_regression.py` hoặc `python k_means_clustering.py`.

---

## 🇬🇧 English

This project is an ecosystem built from scratch to perform Automatic Differentiation (Autograd), optimize parameters using Adam Optimizer, and apply these concepts to real-world economic data analysis tasks.

### Project Structure:

1. **`core.py` (Autograd Engine & Adam Optimizer):**
   The core of the project contains the `Value` class - a miniature automatic differentiation engine conceptually similar to PyTorch. It tracks operations (addition, subtraction, multiplication, power, etc.) into a computational graph and supports backpropagation via the `backward()` method.

2. **`visualize.py` (Adam Visualization):**
   Uses `matplotlib` to visually demonstrate how the Adam optimization algorithm finds the minimum of a 3D bowl-shaped function (simulating a marble rolling down a slope).

3. **`poly_regression.py` (Polynomial Regression):**
   Applies `core.py` to practical Machine Learning: Fitting a 2nd-degree polynomial regression curve ($y = ax^2 + bx + c$) to a real dataset involving "Brent Oil Price" and "GDP Growth". The code self-optimizes coefficients $a, b, c$ without resorting to pre-built ML libraries like scikit-learn.

4. **`k_means_clustering.py` (K-Means Clustering):**
   Implements the K-Means clustering algorithm completely from scratch using only matrix operations in `numpy`. This script analyzes and groups Middle Eastern countries (data from 2023) into 3 economic categories based on GDP per Capita and Inflation rates.

5. **`Middle_East_Economic_Data_1990_2024_with_Oil.csv`:**
   The CSV dataset containing macroeconomic parameters of Middle Eastern countries over several decades, which serves as the training input for our models.

### How to Run:
Install prerequisites: `pip install numpy matplotlib`
Run the python scripts to see the generated graphs: `python poly_regression.py` or `python k_means_clustering.py`.
