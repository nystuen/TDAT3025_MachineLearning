from math import e

from matplotlib import pyplot as plt
import numpy as np
from compute import W, b, x_train, y_train

print("w: %s  b: %s" % (W, b))

fig, ax = plt.subplots()

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x')
ax.set_ylabel('y')

class NonLinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return 20 * self.g(np.matmul(x, self.W) + self.b) + 31
        #return x * self.W + self.b

    def g(self, z):
        return 1 / (1 + np.exp(-z))

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = NonLinearRegressionModel(np.mat(W), np.mat(b))

x = np.expand_dims(np.arange(np.min(x_train), np.max(x_train), 100), 1)
print(x)
#x = np.mat([[np.min(x_train)], [np.max(x_train)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

print('loss:', model.loss(x_train, y_train))

ax.legend()
plt.show()
