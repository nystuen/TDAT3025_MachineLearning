import tensorflow as tf
import numpy as np
import pandas as pd

class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])
        # Logits
        logits = tf.matmul(self.x, self.W) + self.b
        # Predictor
        #self.f = tf.sigmoid(logits)
        # Uses Cross Entropy
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)
        ##self.loss = tf.nn.losses.sigmoid_cross_entropy_with_logits(self.y, logits)
       ## self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
    def f(self, W, b, x):
        logits = tf.matmul(x, W) + b
        return tf.sigmoid(logits)

model = SigmoidModel()

x_train = np.mat([[0], [1]])
y_train = np.mat([[1], [0]])


least_loss = 1000;
gradientInt = 10000;

# Training: adjust the model so that its loss is minimized
print("Using gradient int: %s" % gradientInt)
minimize_operation = tf.train.GradientDescentOptimizer(gradientInt).minimize(model.loss)

# Create session object for running TF operations
session = tf.Session()

# Init tf.Variable objects
session.run(tf.global_variables_initializer())
iteration = 0;
for epoch in range(10000):
    if (epoch % 10000) == 0:
        iteration += 100
        print("10k iteration %s" % iteration)
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
'''

'''
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

x_train2 = np.mat([[1], [1], [0], [1], [0]])
y_train2 = np.mat([[0], [0], [1], [0], [1]])
##accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f(x_train2), 1),tf.argmax(y_train2, 1)),tf.float32))

'''
0.00015009
0.00030043
'''


##print("result: " + model.f(W, int(b), 0))
