import tensorflow as tf
from tensorflow.python.client import timeline
import sys
import time

# Creates a graph.
device_name = sys.argv[1]
shape = (int(sys.argv[2]), int(sys.argv[2]))

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    sum_operation = tf.reduce_sum(random_matrix)

config = tf.ConfigProto(log_device_placement=True)
if len(sys.argv) <= 3:
    pass
elif sys.argv[3] == "growth":
    config.gpu_options.allow_growth = True
else:
    config.gpu_options.per_process_gpu_memory_fraction = float(sys.argv[3])

with tf.Session(config=config) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    print(sess.run(sum_operation, options=run_options, run_metadata=run_metadata))
    trace_file = tf.gfile.Open(name='timeline.json', mode='w')
    tl = timeline.Timeline(run_metadata.step_stats)
    trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

print("after Session.run() ...")
time.sleep(5)
