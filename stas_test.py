import tensorflow as tf

x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(tf.reduce_mean(x)))
    print(sess.run(tf.reduce_mean(x, axis=0)))
    print(sess.run(tf.reduce_mean(x, axis=1)))
    print(sess.run(tf.reduce_mean(x, axis=-1)))
    print(sess.run(tf.reduce_mean(x, axis=-2)))
    print(sess.run(tf.reduce_mean(x, axis=0, keepdims=True)))
