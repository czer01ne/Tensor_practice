import numpy as np

num_points = 1000
vectors_set = []
for i in range(num_points):
         x1 = np.random.normal(0.0, 0.55)
         y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

#x_data = [v[0] for v in vectors_set]
#y_data = np.array([v[1] for v in vectors_set]) * 1000
#print(x_data)
#print(y_data)
x_data = [19., 18., 28., 33., 32., 31., 46., 37., 37.]
#y_data = np.array([ 16884.92,   1725.55,   4449.46,  21984.47,   3866.86,   3756.62, 8240.59,   7281.51,   6406.41]) / 10000

import matplotlib.pyplot as plt


import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(3000):
    sess.run(train)
    print(step, sess.run(W), sess.run(b), sess.run(loss))
