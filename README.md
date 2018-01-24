# VariationalAutoEncoder-MNIST

## Overview ## 
This is a TensorFlow implementation of the Variational Auto Encoder architecture as described in the paper trained on the MNIST dataset. 


## Architecture used ## 
* The architecture for the encoder is a simple MLP with one hidden layer that outputs the latent distribution's mean vector and standard deviation vector.
* The latent variable is sampled from a gaussian distribution using the mean and  variances given by the encoder network.
* The architecture for the decoder is an MLP with one hidden layer and a softmax final layer  

## Quick start ## 
 ### Training ### 
 This trains the VAE network on the MNIST dataset and saves the learned encoder-decoded weights in the `trained_model` subfolders.
 ```
 python train.py 
 ``` 
 Options
  * `--batch_size` defaults to 100.
  * `--latent_dim` defaults to 10.
  * `--hidden_dim` defaults to 500.
  * `--num_epochs` defaults to 50.
 
 ### Visualisation ###
 - This generates a visualisation of reconstructed images and the latent features and saves them in `result/`
 ```
 python evaluate.py --reconstruct 1
 ```
 - This generates a visualisation of generated images and saves it in `result/`
 ``` 
 python evaluate.py --generate 1
 ```
 - Needed Arguments:
  * `--saved_path`: filepath to the saved model
  * `--latent_dim`: dimension of the latent variable used in the model
  * `--hidden_dim`: dimension of the hidden layer used in the model
  * `--batch_size`: batch_size used during training of the model
  * `--num_epochs`: number of epochs used during the training
  * `--num_gen`: number of reconstruction examples to be visualised.

## Results ## 

### Reconstruction ### 

Example of images reconstructed from MNIST dataset.
For each column, left is the reconstructed image, right is the original input. 

| Latent Dimension: 10| Latent Dimension: 5 | Latent Dimension: 2 |
| :-------------: |:-------------:| :-----:|
|![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_reconstructed_hd500_ld10_e50.png)|![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_reconstructed_hd500_ld5_e50.png) |![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_reconstructed_hd500_ld2_e50.png) |



### Generation ### 
Example of generated images from latent variables randomly sampled from a normal distribution.

| Latent Dimension: 10| Latent Dimension: 5 | Latent Dimension: 2 |
| :-------------: |:-------------:| :-----:|
|![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_generated_hd500_ld10_e50.png) |![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_generated_hd500_ld5_e50.png) |![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/I_generated_hd500_ld2_e50.png) |

### Latent space visualisation ###
We plot the latent variables obtained from the MNIST dataset according to their labels. If the latent dimension is > 10, the variables are first embedded in 2D using t-SNE embedding.

| Latent Dimension: 10| Latent Dimension: 5 | Latent Dimension: 2 |
| :-------------: |:-------------:| :-----:|
|![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/latent_space_hd500_ld10_e50.png)|![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/latent_space_hd500_ld5_e50.png) |![generated](https://github.com/Nocty-chan/VariationalAutoEncoder-MNIST/blob/master/result/latent_space_hd500_ld2_e50.png) |


## References ## 
* [Autoencoding Variational Bayes, Kingma and Welling](https://arxiv.org/pdf/1312.6114.pdf)
* [What is a variational auto-encoder ? Tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
* https://github.com/hwalsuklee/tensorflow-mnist-VAE
* https://github.com/shaohua0116/VAE-Tensorflow
