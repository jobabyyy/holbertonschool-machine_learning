#!/usr/bin/env python3
"""Trains a loaded Neural Network model using mini-batch gradient descent"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural model"""

    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        total_batches = -(-len(X_train) // batch_size)  # Ceil division

        for epoch in range(epochs):
            print("After {} epochs:".format(epoch+1))
            train_cost = 0
            train_accuracy = 0

            # Shuffle the training data for each epoch
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            for batch in range(total_batches):
                start = batch * batch_size
                end = min(start + batch_size, len(X_train))
                X_batch = shuffled_X[start:end]
                Y_batch = shuffled_Y[start:end]

                feed_dict = {x: X_batch, y: Y_batch}
                _, step_cost, step_accuracy =
                sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                train_cost += step_cost
                train_accuracy += step_accuracy

                if (batch + 1) % 100 == 0 or batch == total_batches - 1:
                    print("\tStep {}: ".format(batch + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

            train_cost /= total_batches
            train_accuracy /= total_batches

            valid_cost, valid_accuracy =
            sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
