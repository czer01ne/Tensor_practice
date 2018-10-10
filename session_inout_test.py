import tensorflow as tf

sess = tf.Session()

a = tf.constant(1)
b = tf.constant(2)
c = a + b
x = tf.constant([10, 20])
y = tf.constant([1.0, 2.0])

v = sess.run(a)
print("%s: %r\n" % (type(v), v))

v = sess.run(c)
print("%s: %r\n" % (type(v), v))

v = sess.run(x)
print("%s: %r\n" % (type(v), v))

v = sess.run([x, y])
print("%s: %r\n" % (type(v), v))

import collections
MyData = collections.namedtuple('MyData', ['x', 'y'])
v = sess.run({'k1': MyData(x, y), 'k2': [y, x]})
print("%s: %r\n" % (type(v), v))

v = sess.run(c, feed_dict={a: 3, b: 4})
print(v)
