import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from Task2.C.compute import W1, W2, b1, b2

fig = plt.figure()
ax = fig.gca(projection='3d')

# Observed/training input and output

# The training data
x_train = np.mat([[0,1], [1,1], [1,0], [0,0]])
y_train = np.mat([[1], [0], [1], [0]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot(x_train, y_train, z_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
#ax.scatter(x_train1, x_train2, y_train, marker='o', label='$y = f(x, z) = xW1 + zW2 + b$')

class SigmoidModel:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2


    def g(self, z):
        return 1 / (1 + np.exp(-z))

    # Predictor
    def f(self, x):
        f1 = self.g(x * self.W1 + self.b1)
        f2 = self.g(f1 * self.W2 + self.b2)
        return f2


model = SigmoidModel(W1, W2, b1, b2)
x1_test = np.arange(0, 1, 0.001)
x2_test = np.arange(0, 1, 0.001)

'''
x1_test, x2_test = np.meshgrid(x1_test, x2_test)
z_test = model.f(x1_test, x2_test)
'''
'''x = np.mat([[np.min(x1_test)], [np.max(x2_test)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')'''

x_grid, y_grid = np.meshgrid(np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 10), np.linspace(np.min(x_train[:, 1]), np.max(x_train[:, 1]), 10))
z_grid = np.empty(x_grid.shape)
for i in range(0, z_grid.shape[0]):
    for j in range(0, z_grid.shape[1]):
        z_grid[i, j] = model.f([[x_grid[i, j], y_grid[i, j]]])
ax.plot_wireframe(x_grid, y_grid, z_grid, color='green', label='$y = f(x) = xW+b$')



#z_test = np.arange(0, 1, 0.001)

#model = LinearRegressionModel(np.mat(W1),np.mat(W2),np.mat(b))
'''
x_surface1 = [[np.min(x_train1)], [np.min(x_train1)], [np.max(x_train1)], [np.max(x_train1)]]
x_surface2 = [[np.min(x_train2)], [np.min(x_train2)], [np.max(x_train2)], [np.max(x_train2)]]
x_surface1, x_surface2 = np.meshgrid(x_surface1, x_surface2)
z_surface = model.f(x_surface1, x_surface2)
'''
#ax.plot_wireframe(x1_test, x2_test, z_test)
ax.view_init(30, 30)
plt.show()
#print(x_surface, y_surface, z_surface)

#ax.plot_surface(x_surface1, x_surface2, z_surface, alpha=0.3, color='blue')
# rotate the axes and update
