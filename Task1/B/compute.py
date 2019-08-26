import numpy as np
import tensorflow as tf
import pandas as pd

data = np.array(pd.read_csv(
    "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv")).transpose()
print(data)
# Observed/training input and output
x_train = np.expand_dims(data[0], 1)
y_train = np.expand_dims(data[1], 1)
z_train = np.expand_dims(data[2], 1)


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.z = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[0.0]])
        self.W2 = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.y, self.W1) + tf.matmul(self.z, self.W2) + self.b
        # Uses Mean Squared Error
        self.loss = tf.losses.mean_squared_error(f, self.x)
        #self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(100).minimize(model.loss)

# Create session object for running TF operations
session = tf.Session()

# Init tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train, model.z: z_train})

# Evaluate training accuracy
W1, W2, b, loss = session.run([model.W1, model.W2, model.b, model.loss], {model.x: x_train, model.y: y_train, model.z: z_train})
print("W1 = %s, W2 = %s, b = %s, loss = %s" % (W1, W2, b, loss))

session.close()