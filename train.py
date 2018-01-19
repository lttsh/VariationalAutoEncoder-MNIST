import numpy as np
import matplotlib.pyplot as plt
from vae import VariationalAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import argparse

WIDTH = HEIGHT = 28


## Parsing arguments
parser = argparse.ArgumentParser(description='Parse arguments for MNIST VAE.')
parser.add_argument('--train', help='train the autoencoder')
parser.add_argument('--reconstruct', help='train the autoencoder')
parser.add_argument('--generate', help='train the autoencoder')
parser.add_argument('--batch_size', type=int, default=100, help='Give batch size for training')
parser.add_argument('--latent_dim', type=int, default=10, help='Give latent dimension for model')
parser.add_argument('--hidden_dim', type=int, default=500, help='Give latent dimension for model')
parser.add_argument('--num_epochs', type=int, default=50, help='Give latent dimension for model')
parser.add_argument('--saved_path', type=str, default=None, help='Give latent dimension for model')
parser.add_argument('--num_gen', type=int, default=9, help='Give latent dimension for model')
args = parser.parse_args()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print ("Loaded MNIST dataset")
image_dim = mnist.train.images[0].shape[0]
num_sample = mnist.train.num_examples
print ("Image dimension %d, number of samples %d" % (image_dim, num_sample))

# Test the trained model: reconstruction

# Visualisation from https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py

def test_reconstruction(model):
    batch = mnist.test.next_batch(169)[0]
    x_reconstructed = model.get_reconstruct(batch)

    n = np.sqrt(169).astype(np.int32)
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


# Generates N new images and plot them
def test_generation(model, N):
    x_generated = model.get_generated(N)
    n = np.sqrt(N).astype(np.int32)
    print (n)
    I_generated = np.empty((HEIGHT*n, WIDTH*n))
    for i in range(n):
        for j in range(n):
            x =x_generated[i*n+j, :].reshape(HEIGHT, WIDTH)
            I_generated[i*HEIGHT:(i+1)*HEIGHT, j*WIDTH:(j+1)*WIDTH] = x
    fig = plt.figure()
    plt.imshow(I_generated, cmap='gray')
    plt.savefig('I_generated.png')
    plt.close(fig)

model = VariationalAutoEncoder(\
    dim_image=image_dim,
    batch_size=args.batch_size,
    latent_dim=args.latent_dim,
    hidden_dim=args.hidden_dim,
    saved_path=args.saved_path)
# model = train()
if args.train:
    model.train_model(mnist.train, num_epochs=args.num_epochs)
if args.reconstruct:
    test_reconstruction(model)
if args.generate:
    test_generation(model, args.num_gen)
else:
    print("Error")
