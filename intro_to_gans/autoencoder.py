# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    
Original Source: 
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

Modified by Zach Bessinger
"""
from __future__ import division, print_function, absolute_import

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

out_dir = 'ae_out/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Parameters
learning_rate = 0.005
n_epochs = 200
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


def plot_test_set():
    # Applying encode and decode over test set
    n = examples_to_show
    encode_decode = sess.run(X_hat, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    fig, a = plt.subplots(2, n, figsize=(n, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='Greys_r')
        a[0][i].axis('off')
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='Greys_r')
        a[1][i].axis('off')

    return fig


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
n_batches = int(mnist.train.num_examples / batch_size)

# Construct model
encoder_op = encoder(X)  # Encode our data, X
X_hat = decoder(encoder_op)  # Reconstruction of X, X_hat

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(X - X_hat, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
i_step = 0

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(n_epochs):
        # Loop over all batches
        for i in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: X_batch})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

        fig = plot_test_set()
        plt.savefig(out_dir + '{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

    print("Optimization Finished!")
