import tensorflow as tf
import numpy as np

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.constant_initializer(0.))

class VariationalAutoEncoder(object):

    def __init__(self,
            lr=1e-3,
            batch_size=100,
            latent_dim=10,
            dim_image=784,
            hidden_dim=500,
            saved_path=None):
        self.learning_rate = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.dim_image = dim_image
        self.hidden_dim = hidden_dim
        self.saved_path=saved_path
        # Build all graph
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_image])
        self.buildEncoder()
        self.latent = self.generate_latent()
        self.reconstruct = self.buildDecoder(self.latent)
        self.loss()
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
            .minimize(self.total_loss)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if self.saved_path is not None:
            self.saver.restore(self.sess, self.saved_path)
            print("Restored model")

    # Build encoder
    # Returns (mu, sigma)
    def buildEncoder(self):
        W1 = weight_variable("we1", [self.dim_image, self.hidden_dim])
        b1 = bias_variable("be1", [self.hidden_dim])
        h = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
        W2 = weight_variable("we2", [self.hidden_dim, 2 * self.latent_dim])
        b2 = bias_variable("be2", [2 * self.latent_dim])
        z = tf.matmul(h, W2) + b2
        print ("latent shape", z.get_shape())
        self.mu = z[:, :self.latent_dim]
        self.sigma = tf.exp(z[:, self.latent_dim:])

    def generate_latent(self):
        print("Mu and sigma shape", self.mu.get_shape(), self.sigma.get_shape())
        latent =  self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        return latent

    # Build decoder
    # Returns the decoded vector y from mu and sigma
    def buildDecoder(self, latent):
        W1 = weight_variable("wd1", [self.latent_dim, self.hidden_dim])
        b1 = bias_variable("bd1", [self.hidden_dim])
        reconstruct = tf.nn.tanh(tf.matmul(latent, W1) + b1)
        W2 = weight_variable("wd2", [self.hidden_dim, self.dim_image])
        b2 = bias_variable("bd2", [self.dim_image])
        reconstruct = tf.nn.sigmoid(tf.matmul(reconstruct, W2) + b2)
        return reconstruct

    # Given initial data, reconstructed data and parameters, compute the loss.
    def loss(self):
        self.likelihood = -tf.reduce_mean(tf.reduce_sum(
            self.x * tf.log(self.reconstruct) +
            (1 - self.x) * tf.log(1 - self.reconstruct), 1))
        self.kldiv = tf.reduce_mean(0.5 * tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) -
            tf.log(1e-8 + tf.square(self.sigma)) - 1, 1))
        self.total_loss = self.kldiv + self.likelihood

    def train_step(self, x):
        _, total_loss, likelihood, kldiv = \
            self.sess.run([self.train, self.total_loss, self.likelihood, self.kldiv], feed_dict={self.x: x})
        return total_loss, likelihood, kldiv

    def train_model(self, data, num_epochs=50, batch_size=100):
        if self.saved_path is  None:
            num_sample = data.num_examples
            for epoch in range(num_epochs):
                for iter in range(num_sample // batch_size):
                    batch = data.next_batch(batch_size)[0]
                    losses = self.train_step(batch)
                print('[Epoch {}] Loss: {}'.format(epoch, losses))
            self.saved_path = self.saver.save(self.sess, './model')
            print("Model saved in file: %s" % self.saved_path)

    def get_reconstruct(self, x):
        reconstructed = self.sess.run(
            self.reconstruct, feed_dict={self.x:x}
        )
        return reconstructed

    def get_generated(self, N):
        random_latent = np.random.normal(loc=0, scale=1, size=[N, self.latent_dim])
        return self.sess.run(self.reconstruct, feed_dict={self.latent:random_latent})
