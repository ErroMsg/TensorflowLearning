import tensorflow as tf
import numpy as np


print("tensor version is ", tf.__version__)
print("numpy version is ", np.__version__)

# generate data
x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data * 0.9 + 0.3  # 0.9 is weight, 0.3 is b

# create tensor structure
Weight = tf.Variable(tf.random_uniform([1], -1, 1))  # define Weight range
biases = tf.Variable(tf.zeros([1]))  # fix begin 0
y = Weight * x_data + biases  # prediction method

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
# define train
train = optimizer.minimize(loss)

# initial = tf.initialize_all_variables()
initial = tf.global_variables_initializer()
session = tf.Session()
session.run(initial)  # important!!

for step in range(0, 101):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(Weight), session.run(biases))
