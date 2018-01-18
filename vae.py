import tensorflow as tf
import numpy as np

def weight_variable(shape):
    # TODO: Find out how to properly initialize
    weight = tf.zeros(shape)
    return tf.Variable(weight)
def bias_variable(shape):
    # TODO: Find out how to properly initialize
    bias = tf.zeros(shape)
    return tf.Variable(bias)

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
        self.total_loss = self.loss(self.x, reconstruct, mu, sigma)
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                .minimize(self.total_loss)

    # Build encoder
    # Returns (mu, sigma)
    def buildEncoder(self, hidden_dim):
        W1 = weight_variable([self.dim_image, hidden_dim])
        b1 = bias_variable([hidden_dim])
        z = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
        W2 = weight_variable([hidden_dim, 2 * self.latent_dim])
        b2 = bias_variable([2 * self.latent_dim])
        z = tf.nn.sigmoid(tf.matmul(z, W2) + b2)
        return z[:self.latent_dim], z[self.latent_dim:]

    def generate_latent(self, mu, sigma):
        distribution = tf.distributions.Normal(loc=mu, scale=sigma)
        latent = distribution.sample([self.batch_size])
        return latent

    # Build decoder
    # Returns the decoded vector y from mu and sigma
    def buildDecoder(self, latent, hidden_dim):
        W1 = weight_variable([self.latent_dim, hidden_dim])
        b1 = bias_variable([hidden_dim])
        reconstruct = tf.nn.sigmoid(tf.matmul(latent, W1) + b1)
        W2 = weight_variable([hidden_dim, self.dim_image])
        b2 = bias_variable([self.dim_image])
        reconstruct = tf.nn.sigmoid(tf.matmul(reconstruct, W2) + b2)
        return reconstruct

    # Given initial data, reconstructed data and parameters, compute the loss.
    def loss(self, reconstruct, mu, sigma):
        bernoulli = tf.distributions.Bernoulli(probs=reconstruct)
        values = bernoulli.log_prob(self.x)
        expectation = tf.reduce_mean(values)
        distribution = tf.distributions.Normal(loc=mu, scale=sigma)
        std = tf.distributions.Normal(\
            loc=np.zeros(self.latent_dim),
            scale=np.ones(self.latent_dim))
        kldiv = tf.distributions.kl_divergence(distribution, std)
        return tf.reduce_sum(expectation, kldiv)

    def train(self, x):
        loss = self.sess.run([self.train, self.total_loss], feed_dict={self.x: x})
        return loss

    def generate(self, z):
        logits = self.sess.run(self.reconstruct, feed_dict={self.latent: z})
        bernoulli = tf.distributions.Bernoulli(probs=logits)
        return bernoulli.sample([0])
