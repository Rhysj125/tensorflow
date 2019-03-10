import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_Data/", one_hot=True)

# Number of inputs
x = tf.placeholder(tf.float32, shape=[None, 784])

# Stores probability predicted for that digit 0-9
# Number of outputs
y_ = tf.placeholder(tf.float32, [None, 10])

# Weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss as cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Each training step want to minize the cross entropy error
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Set number of training steps
    num_training_steps = 1000

    # Set number of iterations to complete before providing an update on the accuracy
    display_interval = 100

    # Perform given number of training steps
    for i in range(num_training_steps):
        # Get 100 random data points from data. batch_xs = images
        # batch_ys is the label
        batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if (i % display_interval) == 0:
            # Compare highest probability output with label
            # actual = y, predicted = y_
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels})

            print("Test accuracy: {0}%".format(test_accuracy * 100.0))

