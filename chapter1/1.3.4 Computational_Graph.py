# Repeat 노드
import numpy as np
D, N = 8, 7
x = np.random.randn(1, D)   # 입력
y = np.repeat(x, N, axis=0)   # 순전파
dy = np.random.randn(N, D)  # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True)  # 역전파

print("Repeat 노드 결과값 출력")
print("-" * 70)
print(x)
print('-' * 70)
print(y)
print('-' * 70)
print(dy)
print('-' * 70)
print(dx)

# Sum 노드

import numpy as np
D, N = 8, 7
x = np.random.randn(N, D) # 입력
y = np.sum(x, axis=0, keepdims=True) # 순전파

dy = np.random.randn(1, D) # 무작위 기울기
dx = np.repeat(dy, N, axis=0) # 역전파

print("Sum 노드 결과값 출력")
print("-" * 70)
print(x)
print("-" * 70)
print(y)
print("-" * 70)
print(dy)
print("-" * 70)
print(dx)
print()

