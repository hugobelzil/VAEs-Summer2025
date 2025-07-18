from Models.LogitNormalVAE import *
from HuslerReisVAE import uniformity_penalty, custom_loss, dataHR, eval_dataHR
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import kstest

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

model = Std_VAE_LogitNormal(latent_dim=12, input_dim = 2, LAYER_1_N=10,
                          LAYER_2_N = 12, KL_WEIGHT=0.1)

model.load_weights("best_model_LogitNormal")

#PLOT OF SAMPLED DATA
N_samples = 15000 #same number as training data
prior_samples = model.encoder.prior.sample(N_samples)
samples_vae = model.decoder(prior_samples).sample()
plt.scatter(samples_vae[:,0],samples_vae[:,1], s=10, marker='o', alpha=0.2)
plt.title(f'Simulated data from a HR copula by the VAE({N_samples} samples)')
plt.show()

#PLOTS WITH FEWER NUMBER OF SAMPLES
n_samples_small = 150
indices_1 = np.random.choice(eval_dataHR.shape[0], size=n_samples_small)
indices_2 = np.random.choice(eval_dataHR.shape[0], size=n_samples_small)
indices_3 = np.random.choice(eval_dataHR.shape[0], size=n_samples_small)

samplehr_1 = eval_dataHR[indices_1]
samplehr_2 = eval_dataHR[indices_2]
samplehr_3 = eval_dataHR[indices_3]

sample_model_1 = model.decoder(model.encoder.prior.sample(n_samples_small)).sample()
sample_model_2 = model.decoder(model.encoder.prior.sample(n_samples_small)).sample()
sample_model_3 = model.decoder(model.encoder.prior.sample(n_samples_small)).sample()

fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns

# First subplot
axs[0].scatter(samplehr_1[:,0], samplehr_1[:,1], color = 'red', s=10, marker='o', alpha=0.5, label = "HR")
axs[0].scatter(sample_model_1[:,0], sample_model_1[:,1], color = 'blue', s=10, marker='o', alpha=0.5, label = "Model")
axs[0].set_title(f"First sample ({n_samples_small} pts)")

# Second subplot
axs[1].scatter(samplehr_2[:,0], samplehr_2[:,1], color = 'red', s=10, marker='o', alpha=0.5, label = "HR")
axs[1].scatter(sample_model_2[:,0], sample_model_2[:,1], color = 'blue', s=10, marker='o', alpha=0.5, label = "Model")
axs[1].set_title(f"Second sample ({n_samples_small} pts)")

# Third subplot
axs[2].scatter(samplehr_3[:,0], samplehr_3[:,1], color = 'red', s=10, marker='o', alpha=0.5, label = "HR")
axs[2].scatter(sample_model_3[:,0], sample_model_3[:,1], color = 'blue', s=10, marker='o', alpha=0.5, label = "Model")
axs[2].set_title(f"Third sample ({n_samples_small} pts)")

axs[0].legend(loc='best')
axs[1].legend(loc='best')
axs[2].legend(loc='best')


plt.tight_layout()
plt.show()



#PLOT OF THE MARGINS
plot_margins = True

if plot_margins:
    #Get KS statistic for each marginal
    stat1, pv1 = kstest(samples_vae[:,0],'uniform')
    plt.hist(samples_vae[:, 0], bins='auto', edgecolor='white', color='#3B82F6', density=True)
    plt.axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
    plt.title(f'Histogram of first marginal Husler-Reis Samples \n (KS Test p-value = {round(pv1,4)})')
    plt.show()

    stat2, pv2 = kstest(samples_vae[:,1],'uniform')
    plt.hist(samples_vae[:, 1], bins='auto', edgecolor='white', color='#3B82F6', density=True)
    plt.axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
    plt.title(f'Histogram of second marginal Husler-Reis Samples \n (KS Test p-value = {round(pv2,4)})')
    plt.show()

# MODEL RE-TRAINING : UNIFORMIZATION OF THE MARGINS
# In this part below, we retrain the lowest validation loss achieving model
# with the custom function added to the ELBO in order to enforce the uniformity of the margins
UNIFORMITY_WEIGHT = 0.002

if __name__ == '__main__':
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss(uniformity_weight=UNIFORMITY_WEIGHT))

    model.fit(dataHR, dataHR, validation_data = (eval_dataHR, eval_dataHR),
              batch_size = 32,
              epochs = 20)

    print(f"Uniformization model has finished training with uniformity weight = {UNIFORMITY_WEIGHT}")
    post_training_analysis = True

    if post_training_analysis:
        prior_samples = model.encoder.prior.sample(N_samples)
        samples_vae_AU = model.decoder(prior_samples).sample()
        plt.scatter(samples_vae_AU[:, 0], samples_vae_AU[:, 1], s=10, marker='o', alpha=0.2)
        plt.title(f'Simulated data from a HR copula by the VAE({N_samples} samples) \n AFTER uniformization (Uniformity Weight = {UNIFORMITY_WEIGHT})')
        plt.show()

        statistic1, pvalue1 = kstest(samples_vae_AU[:, 0], 'uniform', N=2000)
        plt.hist(samples_vae_AU[:, 0], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        plt.axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        plt.title(f'Histogram of first marginal Husler-Reis Samples \n AFTER uniformization (KS Test p-value = {round(pvalue1,4)}), Uniformity Weight = {UNIFORMITY_WEIGHT}')
        plt.show()

        statistic2, pvalue2 = kstest(samples_vae_AU[:, 1], 'uniform', N=2000)
        plt.hist(samples_vae_AU[:, 1], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        plt.axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        plt.title(f'Histogram of second marginal Husler-Reis Samples \n AFTER uniformization (KS Test p-value = {round(pvalue2,4)}), Uniformity Weight = {UNIFORMITY_WEIGHT}')
        plt.show()

        #COMPARISON OF MARGINS BEFORE AND AFTER UNIFORMIZATION
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].hist(samples_vae[:, 0], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        axs[0].axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        axs[0].set_title(f"First margin before uniformization \n (KS p-value = {pv1})")

        axs[1].hist(samples_vae_AU[:, 0], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        axs[1].axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        axs[1].set_title(f"First margin after uniformization \n (KS p-value = {pvalue1})")

        fig.suptitle(f"Comparison of 1st margin before and after uniformization (Uniformity weight = {UNIFORMITY_WEIGHT})")
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].hist(samples_vae[:, 1], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        axs[0].axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        axs[0].set_title(f"Second margin before uniformization \n (KS p-value = {pv2})")

        axs[1].hist(samples_vae_AU[:, 1], bins='auto', edgecolor='white', color='#3B82F6', density=True)
        axs[1].axline((0, 1), (1, 1), color='grey', linestyle='--', linewidth=1)
        axs[1].set_title(f"Second margin after uniformization \n (KS p-value = {pvalue2})")

        fig.suptitle(f"Comparison of 2nd margin before and after uniformization (Uniformity weight = {UNIFORMITY_WEIGHT})")
        plt.tight_layout()
        plt.show()


