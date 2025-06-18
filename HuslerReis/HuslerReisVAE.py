from Models.LogitNormalVAE import *
import numpy as np
import matplotlib.pyplot as plt


# IMPORTING SAMPLED DATA FROM HR COPULA
dataHR =  np.load('husler_reiss_samples.npy')
plt.scatter(dataHR[:,0], dataHR[:,1], s=10, marker='x')
plt.title('Husler Reis Samples')
plt.show()

# INITIALIZING THE VAE
vae = Std_VAE_LogitNormal(latent_dim=4, input_dim = 2, LAYER_1_N=10,
                          LAYER_2_N = 12, KL_WEIGHT=0.1)
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
#+ sigmoid(rv.x.loc + rv.x.scale^2.sample().reorganize - rank_unif()^2)

vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=negative_log_likelihood)

vae.fit(dataHR,dataHR, batch_size=64, epochs=150)

#PLOT OF SAMPLED DATA
N_samples = 8000
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('Simulated data from a HR copula')
plt.show()

#PLOT OF THE MARGINS
plt.hist(samples_vae[:,0], bins='auto',edgecolor='blue',density=True)
plt.title('Histogram of first marginal Husler Reis Samples')
plt.show()

plt.hist(samples_vae[:,1], bins='auto',edgecolor='blue',density=True)
plt.axline((0,1),(1,1))
plt.title('Histogram of second marginal Husler Reis Samples')
plt.show()