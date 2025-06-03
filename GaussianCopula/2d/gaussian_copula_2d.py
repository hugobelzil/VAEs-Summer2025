import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import GaussianCopula

from custom_vae_models import *
#from vae_models import *

np.random.seed(0)

corr = np.array([[1, 0.7],
                 [0.7, 1]])
copula = GaussianCopula(corr = corr)
data = copula.rvs(nobs = 11000)
np.save('training_data_gaussian_2d.npy',data)

# Scatter plot of the training data we gathered
plt.scatter(data[:,0], data[:,1], s=10, marker = 'x')
plt.title('Training data from Gaussian Copula')
plt.show()

# Creating the VAE and fitting
vae = Std_VAE_LogitNormal(latent_dim = 12, input_dim=2, LAYER_1_N=8, LAYER_2_N=12)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),#sur Windows tf.optimizers.Adam
            loss=negative_log_likelihood)

vae.fit(data[:10000,:],data[:10000,:],
        #validation_data=(eval_dataset,eval_dataset),
        batch_size=16,
        epochs=50,
        #callbacks = [model_checkpoint_callback]
       )

N_samples = 5000
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('Sampled data from a Gaussian Copula')
plt.show()

np.save('samples_vae_gaussian_2d.npy',samples_vae.numpy())