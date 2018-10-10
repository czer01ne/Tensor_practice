import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","  # use CPU only
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf
import time

for size in [1000, 2000, 4000, 8000, 16000, 24000]:  # 12 MB, 48 MB, 192 MB, 768 MB, 3 GB, 6.75 GB
    a = tf.random_uniform((size, size), 0.0, 1.0)
    b = tf.random_uniform((size, size), 0.0, 1.0)
    c = tf.matmul(a, b)

    with tf.Session() as sess:
        start = time.time()
        sess.run(c)
        end = time.time()

    print("%s: %f 초" % (c.shape, end - start))
    a = b = c = None    # 쓰레기 수집
    tf.reset_default_graph()
