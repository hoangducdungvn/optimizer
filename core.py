import numpy as np 

class Value:
    def __init__(self, data, _children=()):
        self.data = data 
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children) 

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        out._backward = _backward 
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward 
        return out
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Hiện tại chỉ hỗ trợ số mũ int/float"
        out = Value(self.data**other, (self,))
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited  = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Thả viên bi ở một vị trí ngẫu nhiên rất xa đáy bát (x=10, y=10)
x = Value(10.0)
y = Value(10.0)

# Trạng thái của Adam cho từng biến
m_x, v_x = 0.0, 0.0
m_y, v_y = 0.0, 0.0
lr = 0.5  # Learning rate
beta1, beta2, eps = 0.9, 0.999, 1e-8

print("Tìm đáy bát với Adam...")

for epoch in range(1, 101):
    
    loss = x*x + y*y

    x.grad = 0.0
    y.grad = 0.0
    loss.backward()

    m_x = beta1 * m_x + (1-beta1)*x.grad
    v_x = beta2 * v_x + (1-beta2)*x.grad**2
    m_x_hat = m_x / (1-beta1**epoch)
    v_x_hat = v_x / (1-beta2**epoch)
    x.data -= lr * m_x_hat / (np.sqrt(v_x_hat) + eps)

    m_y = beta1 * m_y + (1-beta1)*y.grad
    v_y = beta2 * v_y + (1-beta2)*y.grad**2
    m_y_hat = m_y / (1-beta1**epoch)
    v_y_hat = v_y / (1-beta2**epoch)
    y.data -= lr * m_y_hat / (np.sqrt(v_y_hat) + eps)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3}: x = {x.data:7.4f}, y = {y.data:7.4f}, Loss = {loss.data:.4f}")