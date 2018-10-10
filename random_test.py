import tensorflow as tf
import numpy as np

# 유니폼 분포 난수 생성 텐서 연산, 난수 씨앗값

r1 = tf.random_uniform([5])

tf.set_random_seed(2018)
r2 = tf.random_uniform([5])

sess = tf.Session()
print(sess.run(r1))
print(sess.run(r2))
sess.close()
sess = tf.Session()
print(sess.run(r1))
print(sess.run(r2))
sess.close()

# 기타 분포 난수 생성

sess = tf.Session()
r3 = tf.random_normal([5])
print(sess.run(r3))
r4 = tf.truncated_normal([5])
print(sess.run(r4))
r5 = tf.multinomial([[0.5, 0.25, 0.25]], 5)
print(sess.run(r5))
r6 = tf.random_gamma([5], 0.5, 1.5)
print(sess.run(r6))
r7 = tf.multinomial(tf.log([[10., 5., 5.], [10., 10., 10.]]), 5)
print(sess.run(r7))
r8 = tf.random_shuffle([[1, 2], [3, 4], [5, 6]])
print(sess.run(r8))
r9 = tf.random_crop([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [2, 2])
print(sess.run(r9))

sess.close()
