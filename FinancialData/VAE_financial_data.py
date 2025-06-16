import numpy as np
import matplotlib.pyplot as plt
from Models.GaussianVAE import *


training_dataset = np.load("data/train_financial_dataset.npy")
eval_dataset = np.load("data/eval_financial_dataset.npy")

# Creating the VAE and fitting
vae = Std_VAE(latent_dim = 12 ,input_dim=6, LAYER_1_N=8, LAYER_2_N=12, KL_WEIGHT=0.1)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),#sur Windows tf.optimizers.Adam
            loss=negative_log_likelihood)

vae.fit(training_dataset,training_dataset,
        validation_data=(eval_dataset,eval_dataset),
        batch_size=64,
        epochs=80,
        #callbacks = [model_checkpoint_callback]
       )

N_samples = 12000
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('First 2 components of the sampled data from the VAE-reproduced distribution')
plt.show()
np.save("data\vae_samples.npy", samples_vae)
print(samples_vae.shape)

