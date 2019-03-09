import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def normalize(list):
    normalise_list = list / np.linalg.norm(list)
    return normalise_list

num_house = 160
np.random.seed(42)

# Generate some house sizes
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate House prices from size
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# Plot size against cost
plt.plot(house_size, house_price, 'bx')
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()

# Define number of training samples
num_train_samples = math.floor(num_house * 0.7)

# Define Training Data
train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asarray(house_price[:num_train_samples:])

# Normalise Data
train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

# Define Test Data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

# Normalise Data
test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Set up placeholder that get updated while performing gradient decent
tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="price")

# Random value based variables that will be trained
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# Inference equation y = mx + c
# Use tensor flow operators to make it clear to Tensorflow
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# Loss function - Mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_house_price, 2))/(2 * num_train_samples)

# Learning rate
learning_rate = 0.1

# Gradient decent optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initialise Tensorflow variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Set number of training iterations and how often progress should be displayed.
    display_every = 2
    num_training_iter = 50

    # Train over given iterations
    for iteration in range(num_training_iter):

        # Fit training data
        for (x, y) in zip(train_house_size_norm, train_house_price_norm):
            sess.run(optimiser, feed_dict={tf_house_size: x, tf_house_price: y})

        if(iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})

            print("Iteration #:",  '%040d' % (iteration + 1), "cost=", "{:.9f}".format(c), "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimisation finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_house_price: train_house_price_norm})
    print("Training Cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_house_price_mean = train_house_price.mean()
    train_house_price_std = train_house_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_house_price_std + train_house_price_mean,
             label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()