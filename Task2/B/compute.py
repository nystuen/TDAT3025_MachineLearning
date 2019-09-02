import tensorflow as tf
import numpy as np
import pandas as pd


class SigmoidModel:
    def __init__(self):
        # Model input
        self.x1 = tf.placeholder(tf.float32)
        self.x2 = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        # Model variables
        self.W1 = tf.Variable([[0.0]])
        self.W2 = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])
        # Logits.
        logits = tf.matmul(self.x1, self.W1) + tf.matmul(self.x2, self.W2) + self.b
        # Predictor
        # self.f = tf.sigmoid(logits)
        # Uses Cross Entropy
       # self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)


    '''
    def f(self, W, b, x):
        logits = tf.matmul(x, W) + b
        return tf.sigmoid(logits)
    '''

model = SigmoidModel()

#x_train = np.mat([[0,0], [0,1], [1,1], [0,1], [1,0]])
x_train1 = np.mat([[0], [1], [1], [0]])
x_train2 = np.mat([[1], [1], [0], [0]])

y_train = np.mat([[1], [0], [1], [1]])

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(1).minimize(model.loss)

# Create session object for running TF operations
session = tf.Session()

# Init tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x1: x_train1,model.x2: x_train2, model.y: y_train})

# Evaluate training accuracy
W1, W2, b, loss = session.run([model.W1,model.W2, model.b, model.loss], {model.x1: x_train1,model.x2: x_train2, model.y: y_train})
print("W1 = %s,W2 = %s, b = %s, loss = %s" % (W1, W2, b, loss))

session.close()

x_train2 = np.mat([[1], [1], [0], [1], [0]])
y_train2 = np.mat([[0], [0], [1], [0], [1]])
#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f(x_train2), 1),tf.argmax(y_train2, 1)),tf.float32))


##print("result: " + model.f(W, int(b), 0))
