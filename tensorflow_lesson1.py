import tensorflow as tf
import numpy as np

print("tensorflow version is ", tf.__version__)
print("numpy version is ", np.__version__)

# generate data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.9 + 0.3  # 0.9 is weight, 0.3 is b

# create tensorflow structure
Weight = tf.Variable(tf.random_uniform([1], -1, 1))
biases = tf.Variable(tf.zeros([1]))
y = Weight * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

initial = tf.initialize_all_variables()
session = tf.Session()
session.run(initial)  # important!!

for step in range(0, 201):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(Weight), session.run(biases))
