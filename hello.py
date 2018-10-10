import tensorflow as tf

sess = tf.Session()
hello = tf.constant('Hello, TensorFlow!') # node (operation)
print(sess.run(hello)) # data flow graph의 실행
