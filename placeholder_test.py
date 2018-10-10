import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

print(type(a), a)

add = tf.add(a, b) # same with 'a + b'
mul = tf.multiply(a, b) # same with 'a * b'
print(mul)

with tf.Session() as sess:
    print("Addition with variables: %d" % sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with variables: %d" % sess.run(mul, feed_dict={a:2, b:3}))
