import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

