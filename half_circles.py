import numpy as np
import matplotlib.pyplot as plt

def sample_two_moons(n_samples=10_000, radius=1.0, gap=0.3,
                     noise_std=0.05, seed=None):
    """
    Simple 2-D distribution: two interleaving half-circles.
    Returns an (n_samples, 2) array.
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2

    # angles for upper and lower moons
    theta = rng.uniform(0, np.pi, n)              # 0 … π
    # upper moon
    x1 = radius * np.cos(theta) + rng.normal(0, noise_std, n)
    y1 = radius * np.sin(theta) + rng.normal(0, noise_std, n)
    # lower moon (shifted right and down to create a gap)
    x2 = radius * np.cos(theta) + radius + gap + rng.normal(0, noise_std, n)
    y2 = -radius * np.sin(theta) - gap            + rng.normal(0, noise_std, n)

    return np.vstack([np.column_stack([x1, y1]),
                      np.column_stack([x2, y2])])

training_data = sample_two_moons()
plt.scatter(training_data[:5000,0],training_data[:5000,1])
plt.title('Training data')
plt.show()

from Models.vae_models import *
vae = Std_VAE()
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),#sur Windows tf.optimizers.Adam
            loss=negative_log_likelihood)

vae.fit(training_data[:2500,:],training_data[:2500,:],
        #validation_data=(eval_dataset,eval_dataset),
        batch_size=16,
        epochs=100,
        #callbacks = [model_checkpoint_callback]
       )

N_samples = 2500
prior_samples = vae.encoder.prior.sample(N_samples)
samples_vae = vae.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='x')
plt.title('Sampled data')
plt.show()