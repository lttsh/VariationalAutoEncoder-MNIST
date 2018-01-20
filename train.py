import numpy as np
import matplotlib.pyplot as plt
from vae import VariationalAutoEncoder
from tensorflow.examples.tutorials.mnist import input_data
import argparse

WIDTH = HEIGHT = 28

## Parsing arguments
parser = argparse.ArgumentParser(description='Parse arguments for MNIST VAE.')
parser.add_argument('--batch_size', type=int, default=100, help='Give batch size for training')
parser.add_argument('--latent_dim', type=int, default=10, help='Give latent dimension for model')
parser.add_argument('--hidden_dim', type=int, default=500, help='Give hidden dimension for NN')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train on')
parser.add_argument('--num_gen', type=int, default=9, help='Number of examples to generate')
args = parser.parse_args()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print ("Loaded MNIST dataset")
image_dim = mnist.train.images[0].shape[0]
num_sample = mnist.train.num_examples

model = VariationalAutoEncoder(
    dim_image=image_dim,
    batch_size=args.batch_size,
    latent_dim=args.latent_dim,
    hidden_dim=args.hidden_dim)

model.train_model(mnist.train, num_epochs=args.num_epochs)
