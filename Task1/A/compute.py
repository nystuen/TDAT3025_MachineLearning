import numpy as np
import tensorflow as tf
import pandas as pd

data = np.array(pd.read_csv(
    "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv")).transpose()
print(data)
# Observed/training input and output
x_train = np.expand_dims(data[0], 1)
y_train = np.expand_dims(data[1], 1)


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])
        # Predictor
        f = tf.matmul(self.x, self.W) + self.b
        # Uses Mean Squared Error
        self.loss = tf.losses.mean_squared_error(f, self.y)
        #self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TF operations
session = tf.Session()

# Init tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()