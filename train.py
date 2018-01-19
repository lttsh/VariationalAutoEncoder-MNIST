import numpy as np
import matplotlib.pyplot as plt
from vae import VariationalAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LATENT_DIM =20
HIDDEN_DIM = 500
NUM_EPOCH = 50
WIDTH=HEIGHT = 28
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print ("Loaded MNIST dataset")
image_dim = mnist.train.images[0].shape[0]
num_sample = mnist.train.num_examples
print ("Image dimension %d, number of samples %d" % (image_dim, num_sample))

model = VariationalAutoEncoder(\
    dim_image=image_dim,
    batch_size=BATCH_SIZE,
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM)

for epoch in range(NUM_EPOCH):
    for iter in range(num_sample // BATCH_SIZE):
        batch = mnist.train.next_batch(BATCH_SIZE)[0]
        losses = model.train_step(batch)
    print('[Epoch {}] Loss: {}'.format(epoch, losses))


# Test the trained model: reconstruction
batch = mnist.test.next_batch(100)[0]
x_reconstructed = model.get_reconstruct(batch)

n = np.sqrt(model.batch_size).astype(np.int32)
I_reconstructed = np.empty((HEIGHT*n, 2*WIDTH*n))
for i in range(n):
    for j in range(n):
        x = np.concatenate(
            (x_reconstructed[i*n+j, :].reshape(HEIGHT, WIDTH),
             batch[i*n+j, :].reshape(HEIGHT, WIDTH)),
            axis=1
        )
        I_reconstructed[i*HEIGHT:(i+1)*HEIGHT, j*2*WIDTH:(j+1)*2*WIDTH] = x

fig = plt.figure()
plt.imshow(I_reconstructed, cmap='gray')
plt.savefig('I_reconstructed.png')
plt.close(fig)
