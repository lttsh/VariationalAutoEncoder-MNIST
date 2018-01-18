import numpy as np
import matplotlib.pyplot as plt
from vae import VariationalAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print ("Loaded MNIST dataset")
image_dim = mnist.train.images[0].shape[0]
num_sample = mnist.train.num_examples
print ("Image dimension %d, number of samples %d" % (image_dim, num_sample))

model = VariationalAutoEncoder(dim_image=image_dim)


for epoch in range(50):
    for iter in range(num_sample // BATCH_SIZE):
        batch = mnist.train.next_batch(BATCH_SIZE)[0]
        loss, reconstruct = model.train_step(batch)
    print('[Epoch {}] Loss: {}'.format(epoch, loss))

# Construct new example
latent = np.random.normal(0, 1, size=(100, 10))
x = model.generate(latent)[0]
print x.shape
for i in range(x.shape[0]):
    plt.imshow(x[i].reshape((28, 28)))
    plt.show()
