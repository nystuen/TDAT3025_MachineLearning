import tensorflow as tf
import numpy as np

class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1))
        self.W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1))
        self.b1 = tf.Variable([[0.0, 0.0]])
        self.b2 = tf.Variable([[0.0]])

        # Predictors
        self.f1 = tf.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        self.f2 = tf.sigmoid(tf.matmul(self.f1, self.W2) + self.b2)

        self.loss = tf.reduce_mean(((self.y * tf.log(self.f2)) + ((1 - self.y) * tf.log(1.0 - self.f2))) * -1)


# The training data
x_train = np.mat([[0, 1], [1, 1], [1, 0], [0,0]])
y_train = np.mat([[1], [0], [1], [0]])

model = SigmoidModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(1).minimize(model.loss)

# Create session object for running TF operations
session = tf.Session()

# Init tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss], {model.x: x_train, model.y: y_train})
print("W1 = %s,W2 = %s, b1 = %s, b2 = %s, loss = %s" % (W1, W2, b1, b2, loss))

session.close()

'''x_train2 = np.mat([[1], [1], [0], [1], [0]])
y_train2 = np.mat([[0], [0], [1], [0], [1]])'''
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f(x_train2), 1),tf.argmax(y_train2, 1)),tf.float32))


##print("result: " + model.f(W, int(b), 0))
