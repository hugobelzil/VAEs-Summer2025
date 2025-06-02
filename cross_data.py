import numpy as np
import matplotlib.pyplot as plt

from vae_models import *

# goal : generate data that looks like a cross in 2D, see if better results than half-circle
np.random.seed(42)

def data_generation(n=5000):
    list = []
    for i in range(n):
        X = np.random.uniform(0,1.4)
        threshold = np.random.uniform(0,1)
        if threshold > 0.5:
            Y = np.cos(X)+np.random.normal(0,0.05)
            list.append([X,Y])
        else:
            Y = 0.8*X**2 + np.random.normal(0,0.05+0.05*X)
            list.append([X,Y])
    return np.array(list)

data = data_generation()
plt.scatter(data[:,0], data[:,1], s=10, marker='x')
plt.title('Training data for the VAE')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Creating the VAE

vae = Std_VAE()

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),#sur Windows tf.optimizers.Adam
            loss=negative_log_likelihood)

vae.fit(data[:2500,:],data[:2500,:],
        #validation_data=(eval_dataset,eval_dataset),
        batch_size=16,
        epochs=100,
        #callbacks = [model_checkpoint_callback]
       )

N_samples = 2500
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.show()

# ANALYSE DES MARGINALES ÉCHANTILLONNÉES
# On espère observer que les marginal soient uniformes [0,1]

plt.hist(samples_vae[:,0], density=True, edgecolor='black', color ='blue')
plt.title('First marginal of sampled data form VAE')
plt.show()