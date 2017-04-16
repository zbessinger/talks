# -*- coding: utf-8 -*-

""" Vanilla GAN Example.
Using an Vanilla GAN on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
    
    Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
    Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." 
    In Advances in neural information processing systems, pp. 2672-2680. 2014.
    
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    

Original Source: https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py

Modified by Zach Bessinger
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

out_dir = 'gan_out/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


n_epochs = 20
batch_size = 128
Z_dim = 100

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for sample in samples:
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_batches = int(mnist.train.num_examples / batch_size)

# Construct model
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# Define losses and optimizers, minimize KL divergence (up to a constant)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Initializing the variables
init = tf.global_variables_initializer()
i_step = 0

# Launch the graph
with tf.Session() as sess:
    # Training cycle
    sess.run(init)
    for epoch in range(n_epochs):
        # Loop over all batches
        for i in range(n_batches):
            if i_step % 1000 == 0:
                samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

                fig = plot(samples)
                plt.savefig(out_dir + '{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)

            X_batch, _ = mnist.train.next_batch(batch_size)

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_batch, Z: sample_Z(batch_size, Z_dim)})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

            if i_step % 1000 == 0:
                print('Iter: {}'.format(i_step))
                print('D loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()

    print("Optimization Finished!")
