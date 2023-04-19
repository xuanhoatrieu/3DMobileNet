import tensorflow as tf

with tf.Session() as sess:
    # Allocate a small amount of GPU memory
    a = tf.zeros((1, 1))
    sess.run(a)
