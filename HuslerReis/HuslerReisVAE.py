import numpy as np
import matplotlib.pyplot as plt
from Models.TruncatedNormalVAE import *
from Models.LogitNormalVAE import *
from Models.ProbitVAE import *
from IPython.display import clear_output
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


# IMPORTING SAMPLED DATA FROM HR COPULA
dataHR =  np.load('husler_reiss_samples.npy')
eval_dataHR = np.load('husler_reiss_samples_eval.npy')
plt.scatter(dataHR[:,0], dataHR[:,1], s=10, marker='o', alpha=0.2)
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

def custom_loss(uniformity_weight=0.001):
    def loss(x, rv_x):
        nll = -rv_x.log_prob(x)
        penalty = uniformity_penalty(rv_x, n_samples=64)
        return nll + uniformity_weight * penalty
    return loss

#LOSS FOR TRAINING ON ELBO ONLY
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) # S

#DISPLAY FOR DYNAMIC PLOTTING OF ELBO
class LiveLossPlotELBO(tf.keras.callbacks.Callback):
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

# DISPLAY FOR DYNAMIC PLOTTING OF THE ELBO, KL AND LL

class LiveLossPlotELBOLogLikKl(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data

    def on_train_begin(self, logs=None):
        self.train_elbo = []
        self.train_kl = []
        self.train_loglikelihood = []
        self.val_elbo = []
        self.val_kl = []
        self.val_loglikelihood = []

    def on_epoch_end(self, epoch, logs=None):
        train_metrics = self.model.get_metrics(self.train_data)
        val_metrics = self.model.get_metrics(self.val_data)

        self.train_elbo.append(train_metrics["elbo"])
        self.train_kl.append(0.1*train_metrics["kl"])
        self.train_loglikelihood.append(train_metrics["nll"])
        self.val_elbo.append(val_metrics["elbo"])
        self.val_kl.append(0.1*val_metrics["kl"])
        self.val_loglikelihood.append(val_metrics["nll"])

        clear_output(wait=True)

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(range(1, len(self.train_elbo)+1), self.train_elbo, label="Train ELBO")
        ax1.plot(range(1,len(self.train_kl)+1), self.train_kl, label="Train KL")
        ax1.plot(range(1, len(self.train_loglikelihood)+1), self.train_loglikelihood, label="Train LogLike")
        ax1.set_xlabel("Epochs")
        ax1.set_title("Training metrics")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(range(1, len(self.val_elbo) + 1), self.val_elbo, label="Val ELBO")
        ax2.plot(range(1, len(self.val_kl) + 1), self.val_kl, label="Val KL")
        ax2.plot(range(1, len(self.val_loglikelihood) + 1), self.val_loglikelihood, label="Val LogLike")
        ax2.set_xlabel("Epochs")
        ax2.set_title("Validation metrics")

        fig.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"Epoch {epoch + 1}:")
        print(f"  Train ELBO: {train_metrics['elbo']:.5f}")
        print(f"  Train NLL : {train_metrics['nll']:.5f}")
        print(f"  Train KL  : {0.1 * train_metrics['kl']:.5f}")  # multiply if you used a weight in ELBO

        print(f"  Val   ELBO: {val_metrics['elbo']:.5f}")
        print(f"  Val   NLL : {val_metrics['nll']:.5f}")
        print(f"  Val   KL  : {0.1 * val_metrics['kl']:.5f}")

#CALLBACK TO SAVE THE BEST WEIGHTS
best_model_callback = ModelCheckpoint(filepath = "best_model_LogitNormal_2",
                                      save_weights_only = True,
                                      monitor = 'val_loss',
                                      mode = 'min',
                                      save_best_only = True)

# INITIALIZING THE VAE
vae = Std_VAE_LogitNormal(latent_dim=12, input_dim = 2, LAYER_1_N=10,
                          LAYER_2_N = 12, KL_WEIGHT=0.1)


vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=negative_log_likelihood) #loss = custom_loss for enforcing uniform margins


if __name__== "__main__":
    # TRAINING THE VAE
    vae.fit(dataHR,dataHR, validation_data = (eval_dataHR, eval_dataHR),
            batch_size=32, epochs=14,
            #callbacks=[LiveLossPlotELBOLogLikKl(train_data=dataHR, val_data=eval_dataHR)].
            callbacks = [LiveLossPlotELBO(),best_model_callback]
            ) #150 epochs de base

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
        plt.hist(samples_vae[:, 0], bins=30, edgecolor='black', density=True)
        plt.axline((0, 1), (1, 1), color='black')
        plt.title('Histogram of first marginal Husler-Reis Samples')
        plt.show()

        plt.hist(samples_vae[:, 1], bins=30, edgecolor='black', density=True)
        plt.axline((0, 1), (1, 1), color='black')
        plt.title('Histogram of second marginal Husler-Reis Samples')
        plt.show()
