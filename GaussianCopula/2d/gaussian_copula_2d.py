import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import GaussianCopula

#from Models.custom_vae_models import *
#from Models.TruncatedNormalVAE import *
from Models.LogitNormalVAE import *
#from Models.BetaDistributionVAE import *

corr = np.array([[1, 0.8],
                 [0.8, 1]])
copula = GaussianCopula(corr = corr)
data = copula.rvs(nobs = 11000)
np.save('training_data_gaussian_2d.npy',data)

# Scatter plot of the training data we gathered
plt.scatter(data[:,0], data[:,1], s=10, marker = 'x')
plt.title('Training data from Gaussian Copula')
plt.show()

# Creating the VAE and fitting
vae = Std_VAE_LogitNormal(latent_dim = 12, input_dim=2, LAYER_1_N=8, LAYER_2_N=12, KL_WEIGHT=0.1)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),#sur Windows tf.optimizers.Adam
            loss=negative_log_likelihood)

vae.fit(data[:10000,:],data[:10000,:],
        #validation_data=(eval_dataset,eval_dataset),
        batch_size=32,
        epochs=40,
        #callbacks = [model_checkpoint_callback]
       )

N_samples = 7000
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('Sampled data from a Gaussian Copula')
plt.show()

np.save('samples_vae_gaussian_2d.npy',samples_vae.numpy())


### THRESHOLD

