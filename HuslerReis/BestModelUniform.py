from Models import LogitNormalVAE
from Models.LogitNormalVAE import *
from HuslerReisVAE import uniformity_penalty, custom_loss, dataHR, eval_dataHR
import tensorflow as tf
import matplotlib.pyplot as plt

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
WEIGHT = 0.0001

model = tf.keras.models.load_model("best_model_LogitNormal.keras",
                                   custom_objects={"LogitNormalVAE": Std_VAE_LogitNormal,
                                                   "loss": negative_log_likelihood},
                                   compile=False)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss = custom_loss(uniformity_weight=WEIGHT))


model.fit(dataHR,dataHR, validation_data = (eval_dataHR, eval_dataHR),
        batch_size=32, epochs=15)

#PLOT OF SAMPLED DATA
N_samples = 8000
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
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


