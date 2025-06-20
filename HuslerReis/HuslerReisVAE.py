import numpy as np
import matplotlib.pyplot as plt
from Models.TruncatedNormalVAE import *
from Models.LogitNormalVAE import *
from IPython.display import clear_output

# IMPORTING SAMPLED DATA FROM HR COPULA
dataHR =  np.load('husler_reiss_samples.npy')
eval_dataHR = np.load('husler_reiss_samples_eval.npy')
plt.scatter(dataHR[:,0], dataHR[:,1], s=10, marker='x')
plt.title('Husler-Reis Samples (Training data)')
plt.show()

# CUSTOM FUNCTION FOR PENALIZING NON-UNIFORM MARGINS
def uniformity_penalty(rv_x, n_samples=64):
    samples = rv_x.sample(n_samples)  # shape: (n_samples, latent_dim)
    samples = tf.reshape(samples, (n_samples, -1))  # Ensure shape is (n_samples, latent_dim)

    latent_dim = tf.shape(samples)[1]
    uniform_grid = tf.linspace(0.0, 1.0, n_samples)

    penalties = tf.TensorArray(dtype=tf.float32, size=latent_dim)

    def body(i, penalties):
        marginal = samples[:, i]                # shape: (n_samples,)
        marginal_sorted = tf.sort(marginal)     # shape: (n_samples,)
        penalty = tf.reduce_mean((marginal_sorted - uniform_grid) ** 2)
        penalties = penalties.write(i, penalty)
        return i + 1, penalties

    def condition(i, penalties):
        return i < latent_dim

    i = tf.constant(0)
    _, penalties = tf.while_loop(condition, body, [i, penalties])
    return tf.reduce_sum(penalties.stack())

# CUSTOM LOSS : LOG_LIKELIHOOD + UNIFORM PENALTY DEFINED ABOVE
def custom_loss(x, rv_x):
    nll = -rv_x.log_prob(x)
    penalty = uniformity_penalty(rv_x, n_samples=64)
    return nll + 0.001*penalty

#DISPLAY FOR DYNAMIC PLOTTING OF THE LOSS FUNCTION
class LiveLossPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs = None):
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs.get("val_loss"))
        clear_output(wait=True)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (ELBO)")
        plt.legend()
        plt.title("Training and Validation Loss Live Progress")
        plt.grid(True)
        plt.show()

# INITIALIZING THE VAE
vae = Std_VAE_LogitNormal(latent_dim=12, input_dim = 2, LAYER_1_N=10,
                          LAYER_2_N = 12, KL_WEIGHT=0.1)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) # Standard ELBO used for certain experiments

vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=negative_log_likelihood) #loss = custom_loss for uniform margins

# TRAINING THE VAE
vae.fit(dataHR,dataHR, validation_data = (eval_dataHR, eval_dataHR),
        batch_size=32, epochs=100, callbacks=[LiveLossPlot()]) #150 epochs de base

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
