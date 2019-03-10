import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Help functions to create weights and bais variables, convolution and pooling layers
# Using RELU activation function. Must be initialised to a small positive number
# Add some noise so not to end up at zero when comparing differences
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial

# Strides is how far and what direction the convolution moves
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#K size is the kernel size - area being pulled together
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Create interactive session
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convert it back into an image rather than a list
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# Feature is 5x5 pixels, 1 input channel, number of features determined by each filter (number of nodes?)
W_conv1 = weight_variable([5,5,1,32])

# Creating the biases for the first layer
b_conv1 = bias_variable([32])

# Do convolution on images, add bias and apply RELU activation function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# push through first convolutional layer outputs
h_pool1 = max_pool_2x2(h_conv1)

## Do same again but for a second layer, firstly passing though the activation of pool1
# Feature of 5x5 pixels, 32 input channels (previous pooling layer outputs), feature size of 64 (number of nodes)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# Connect 1st layer with second layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Create a fully connected layer
# bit confused as to why 7*7*64. (Just roll with it for the moment)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# Connect output of pooling layer 2 as input to full connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop neurons to prevent the system from over learning
keep_prob = tf.placeholder(tf.float32) # Dropout probability as training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
# Connecting previous fully connected layer to the 10 output values
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Loss optimisation
# Adam is a variant on gradient decent. Learning rate is varied
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

import time

num_steps = 3000
display_interval = 100

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if (i % display_interval) == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()

        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}".format(i, end_time - start_time, train_accuracy * 100.0))

# Display Summary
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}) * 100.0))