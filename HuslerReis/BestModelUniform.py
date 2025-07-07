from Models import LogitNormalVAE
from Models.LogitNormalVAE import *
from HuslerReisVAE import uniformity_penalty, custom_loss, dataHR, eval_dataHR
import tensorflow as tf
import matplotlib.pyplot as plt

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
WEIGHT = 0.0001

model = Std_VAE_LogitNormal(latent_dim=12, input_dim = 2, LAYER_1_N=10,
                          LAYER_2_N = 12, KL_WEIGHT=0.1)

model.load_weights("best_model_LogitNormal")
#PLOT OF SAMPLED DATA

N_samples = 8000
prior_samples = model.encoder.prior.sample(N_samples)
samples_vae = model.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('Simulated data from a HR copula by the VAE')
plt.show()

#PLOT OF THE MARGINS
plot_margins = True

if plot_margins:
    plt.hist(samples_vae[:, 0], bins='auto', edgecolor='black', density=True)
    plt.axline((0, 1), (1, 1), color='black')
    plt.title('Histogram of first marginal Husler-Reis Samples')
    plt.show()

    plt.hist(samples_vae[:, 1], bins='auto', edgecolor='black', density=True)
    plt.axline((0, 1), (1, 1), color='black')
    plt.title('Histogram of second marginal Husler-Reis Samples')
    plt.show()


