import tensorflow as tf
import numpy as np

sess = tf.Session()

# 값 상수 텐서

c1 = tf.constant(1.0)
print(type(c1), ",", c1, ",", c1.shape, ",", c1.dtype, ",", sess.run(c1))
c2 = tf.constant([1.0, 2.0, 3.0])
print(c2.shape, ",", c2.dtype, ",", sess.run(c2))
c3 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(c3.shape, ",", c3.dtype, ",", sess.run(c3))
c4 = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
print(c4.shape, ",", c4.dtype, ",", sess.run(c4))
c5 = tf.constant(1.0, shape=(2,3))
print(c5.shape, ",", c5.dtype, ",", sess.run(c5))
c6 = tf.constant([1.0, 2.0], shape=(2,3))
print(c6.shape, ",", c6.dtype, ",", sess.run(c6))
c7 = tf.constant([1, 2, 3])
print(c7.dtype, ",", sess.run(c7))

# 특정값 상수 텐서

print(sess.run(tf.zeros([3])))
print(sess.run(tf.ones([3])))
print(sess.run(tf.fill([3], 2.0)))
print(sess.run(tf.zeros_like((c2))))
print(sess.run(tf.ones_like((c2))))

# 시퀀스

c8 = tf.range(1, 5)
print(type(c8), ",", c8, ",", sess.run(c8))
c9 = tf.range(5)
print(type(c9), ",", c9, ",", sess.run(c9))
print(sess.run(tf.lin_space(1.0, 3.0, 3)))

sess.close()
