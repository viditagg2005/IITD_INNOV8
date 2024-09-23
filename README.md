# IITD INNOV8 Challenge

This is our submission to the INNOV8'24 Challenge at IIT Delhi.

We include the following files as data tables:
- personnel_behavior_info.csv
- personnel_financial_info.csv
- personnel_general_info.csv
- personnel_offences_info.csv

Our entire code is present in innov8-iitd.ipynb where we combine the files into our dataframe. This is then fed to the autoencoder model to identify the most likely perpetrators. An overview of the model:

Autoencoders are a type of neural network designed for unsupervised learning, primarily used for dimensionality reduction or feature extraction. They work by compressing input data into a lower-dimensional latent space (encoding) and then reconstructing the data from this compressed representation (decoding).

An autoencoder consists of three components: the encoder, the latent space, and the decoder. The encoder maps the input data `x` to a latent representation `z` using a function `f(x; θ)`, where `θ` represents the network parameters. For our purposes, we use two dense layers for encoding. Another two dense layers are used for decoder which then reconstructs the original data `x_hat` from `z` using another function `g(z; φ)`, with `φ` as its parameters.

The network is trained by minimizing the difference between the original input and the reconstructed output, using mean squared error (MSE) loss:
 
## Results
The autoencoder gives us the top 50 people most likely to default to the Phyrgian scoundrels. The result is given in the top_likely_defectors_roll_numbers.csv . Note that in our synthetic dataset we have marked the traitors by having their enrollment numbers start with '3'. Also note that this does NOT introduce any bias in the predictive analysis because enrollment ids are not given to the model as input and is just given for easy identification purposes.
