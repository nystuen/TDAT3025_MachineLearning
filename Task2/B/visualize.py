import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from Task2.B.compute import W1, W2, b

fig = plt.figure()
ax = fig.gca(projection='3d')

# Observed/training input and output

x_train = np.mat([[0], [1], [1], [0]])
y_train = np.mat([[1], [1], [0], [0]])
z_train = np.mat([[1], [0], [1], [1]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# ax.plot(x_train, y_train, z_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.scatter(x_train, y_train, z_train, marker='o', label='$y = f(x, z) = xW1 + zW2 + b$')


class LinearRegressionModel:
    def __init__(self, W1, W2, b):
        self.W1 = W1
        self.W2 = W2
        self.b = b

    # Predictor
    def f(self, x, y):
        res = self.g(x * self.W1 + y * self.W2 + self.b)
        print("res: ", res)
        return res

    def g(self, z):
        return 1 / (1 + np.exp(-z))


'''
    # Predictor
    def f(self, x, y):
        return x * self.W1 + y * self.W2 + self.b
'''

model = LinearRegressionModel(W1,W2, b)

x1_test = np.arange(0, 1, 0.001)
x2_test = np.arange(0, 1, 0.001)
x1_test, x2_test = np.meshgrid(x1_test, x2_test)
z_test = model.f(x1_test, x2_test)

#model = LinearRegressionModel(np.mat(W1),np.mat(W2),np.mat(b))
'''
x_surface1 = [[np.min(x_train1)], [np.min(x_train1)], [np.max(x_train1)], [np.max(x_train1)]]
x_surface2 = [[np.min(x_train2)], [np.min(x_train2)], [np.max(x_train2)], [np.max(x_train2)]]
x_surface1, x_surface2 = np.meshgrid(x_surface1, x_surface2)
z_surface = model.f(x_surface1, x_surface2)
'''
ax.plot_wireframe(x1_test, x2_test, z_test)
ax.view_init(30, 30)
plt.show()
#print(x_surface, y_surface, z_surface)

#ax.plot_surface(x_surface1, x_surface2, z_surface, alpha=0.3, color='blue')
# rotate the axes and update
