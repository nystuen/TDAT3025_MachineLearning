from matplotlib import pyplot as plt
import numpy as np
from compute import W1, W2, b, x_train, y_train, z_train

print("w: %s  b: %s" % (W, b))

fig, ax = plt.subplots()

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x')
ax.set_ylabel('y')

class LinearRegressionModel:
    def __init__(self, W1, W2, b):
        self.W1 = W1
        self.W2 = W2
        self.b = b

    # Predictor
    def f(self, y, z):
        return np.matmul(y, self.W1) + np.matmul(z, self.W2) + self.b

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = LinearRegressionModel(np.mat(W1),np.mat(W2), np.mat(b))

x = np.mat([[np.min(x_train)], [np.max(x_train)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

print('loss:', model.loss(x_train, y_train))

ax.legend()
plt.show()
