import tensorflow as tf

v1 = tf.get_variable("v1", (2,))
v2 = tf.get_variable("v2", (2,), dtype=tf.int32)
v3 = tf.get_variable("v3", dtype=tf.int32, initializer=tf.constant([3, 4]))
assign = v1.assign(tf.constant([3.0, 4.0]))
print(type(v1), v1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([v1, v2, v3, assign]))
