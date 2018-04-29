import tensorflow as tf


# print tf version
print(tf.__version__)

# list devices available to tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# assign computation to gpu and check for error
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))




