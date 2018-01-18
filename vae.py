import tensorflow as tf
import numpy as np

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.constant_initializer(0.))

class VariationalAutoEncoder(object):

    def __init__(self, lr=1e-3, bs=100, latent_dim=10, dim_image=784):
        self.learning_rate = lr
        self.batch_size = bs
        self.latent_dim = latent_dim
        self.dim_image = dim_image

        # Build all graph

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.dim_image])
        mu, sigma = self.buildEncoder(50)
        self.latent = self.generate_latent(mu, sigma)
        self.reconstruct = self.buildDecoder(self.latent, 50)
        self.total_loss = self.loss(self.reconstruct, mu, sigma)
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
            .minimize(self.total_loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build encoder
    # Returns (mu, sigma)
    def buildEncoder(self, hidden_dim):
        W1 = weight_variable("we1", [self.dim_image, hidden_dim])
        b1 = bias_variable("be1", [hidden_dim])
        h = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
        W2 = weight_variable("we2", [hidden_dim, 2 * self.latent_dim])
        b2 = bias_variable("be2", [2 * self.latent_dim])
        z = tf.nn.sigmoid(tf.matmul(h, W2) + b2)
        print ("latent shape", z.get_shape())
        return z[:, :self.latent_dim], z[:, self.latent_dim:]

    def generate_latent(self, mu, sigma):
        print("Mu and sigma shape", mu.get_shape(), sigma.get_shape())
        latent =  mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        return latent

    # Build decoder
    # Returns the decoded vector y from mu and sigma
    def buildDecoder(self, latent, hidden_dim):
        W1 = weight_variable("wd1", [self.latent_dim, hidden_dim])
        b1 = bias_variable("bd1", [hidden_dim])
        reconstruct = tf.nn.tanh(tf.matmul(latent, W1) + b1)
        W2 = weight_variable("wd2", [hidden_dim, self.dim_image])
        b2 = bias_variable("bd2", [self.dim_image])
        reconstruct = tf.nn.sigmoid(tf.matmul(reconstruct, W2) + b2)
        return reconstruct

    # Given initial data, reconstructed data and parameters, compute the loss.
    def loss(self, reconstruct, mu, sigma):
        likelihood = tf.reduce_sum(\
            self.x * tf.log(reconstruct) + (1 - self.x) * tf.log(1 - reconstruct), 1)
        kldiv = 0.5 * tf.reduce_sum(\
            tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
        return tf.reduce_mean(kldiv) - tf.reduce_mean(likelihood)

    def train_step(self, x):
        _, loss, reconstruct = self.sess.run([self.train, self.total_loss, self.reconstruct], feed_dict={self.x: x})
        return loss, reconstruct

    def generate(self, z):
        bernoulli = tf.distributions.Bernoulli(probs=self.reconstruct).sample(1)
        return self.sess.run(bernoulli, feed_dict={self.latent: z})
