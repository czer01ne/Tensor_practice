import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","  # use CPU only
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf

print("tf.test.is_built_with_cuda():", tf.test.is_built_with_cuda())
print("tf.test.is_gpu_available():", tf.test.is_gpu_available())

sess = tf.Session()
if (sess.list_devices): # for tensorflow 1.3+
      for d in sess.list_devices():
          print(d.name)
sess.close()

# Undocumented feature
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
