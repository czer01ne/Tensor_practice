import tensorflow as tf

a = tf.constant([[3., 3.]])  # 1 x 2 matrix
b = tf.constant([[2.], [2.]]) # 2 x 1 matrix

product = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(product))
