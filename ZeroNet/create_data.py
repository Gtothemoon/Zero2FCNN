import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
Pi = math.pi
A = 24
B = 3.2
C = 42
D = 2.3

# train data
x_train = np.random.uniform(-2 * Pi, 2 * Pi, (1000, 1))
y_train = [A * math.sin(B * i) + C * math.cos(D * i) for i in x_train]
y_train = np.array(y_train).reshape(1000, 1)
# 保存数据
pd.DataFrame(x_train).to_csv('data/x_train.csv')
pd.DataFrame(y_train).to_csv('data/y_train.csv')
np.savetxt('data/x_train.csv', x_train, delimiter=",")
np.savetxt('data/y_train.csv', y_train, delimiter=",")
# test data
x_test = np.random.uniform(-2 * Pi, 2 * Pi, (100, 1))
y_test = [A * math.sin(B * i) + C * math.cos(D * i) for i in x_test]
y_test = np.array(y_test).reshape(100, 1)
# 保存数据
pd.DataFrame(x_test).to_csv('data/x_test.csv')
pd.DataFrame(y_test).to_csv('data/y_test.csv')
np.savetxt('data/x_test.csv', x_test, delimiter=",")
np.savetxt('data/y_test.csv', y_test, delimiter=",")
