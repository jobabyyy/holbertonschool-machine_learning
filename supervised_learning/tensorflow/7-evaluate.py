#!/usr/bin/env python3
"""evaluates the output of a neural network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """returns network prediction, accuracy, & loss, respectively"""
    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        y_pred = graph.get_tensor_by_name("y_pred:0")
        loss = graph.get_tensor_by_name("loss:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        prediction, acc, l = sess.run([y_pred, accuracy, loss],
                                      feed_dict={x: X, y: Y})

    return prediction, acc, l
