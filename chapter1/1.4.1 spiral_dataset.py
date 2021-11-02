import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape) # 입력 데이터
print('t', t.shape) # 정답 데이터, 원핫 벡터 형태