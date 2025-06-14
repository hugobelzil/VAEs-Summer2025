import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import numpy as np
warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# ENCODER ASSUMING A MULTIVARIATE INDEPENDENT BETA DISTRIBUTION FOR THE PRIOR
# AND A BETA ENCODER AS WELL

class Std_Encoder_Beta(tfk.Model):  # Encodeur

    def __init__(self, encoded_size, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_Encoder_Beta, self).__init__()
        self.encoded_size = encoded_size  # taille de l'espace latent
        self.prior = tfd.Independent(distribution = tfd.Beta(concentration1 = 1.05*np.ones(self.encoded_size, dtype=np.float32),
                                                             concentration0 = 1.05*np.ones(self.encoded_size, dtype=np.float32)),
                                     reinterpreted_batch_ndims = 1)
        self.dense1 = tfkl.Dense(units=LAYER_1_N, activation='relu')
        self.dense2 = tfkl.Dense(units=LAYER_2_N, activation='relu')
        self.dense3 = tfkl.Dense(2*self.encoded_size)
        self.ind_beta = tfpl.DistributionLambda(
            lambda params: tfd.Independent(
                tfd.Beta(concentration1=1e-2 + tf.nn.softplus(params[..., :self.encoded_size]), concentration0=1e-2 + tf.nn.softplus(params[..., self.encoded_size:])),
                reinterpreted_batch_ndims=1),
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=KL_WEIGHT),
            name='beta_encoded_parameters'
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.ind_beta(x)
        return x


#BETA DISTRIBUTION PROBABILISTIC DECODER

class Std_Decoder_Beta(tfk.Model):
    def __init__(self, input_dim, LAYER_1_N, LAYER_2_N):
        super(Std_Decoder_Beta,self).__init__()
        self.K = input_dim
        self.dense1 = tfkl.Dense(units = LAYER_2_N, use_bias=True, activation='relu')
        self.dense2 = tfkl.Dense(units = LAYER_1_N, use_bias=True, activation='relu')
        self.param_layer = tfkl.Dense(2*self.K) #alphas and betas of Beta distribution
        self.indep_Beta_distribution = tfpl.DistributionLambda(
            lambda params : tfd.Independent(
                tfd.Beta(concentration1 = 0.2 + tf.nn.softplus(params[...,:self.K]), concentration0 = 0.2 + tf.nn.softplus(params[...,self.K:])),
                reinterpreted_batch_ndims=1
            ),
            name = 'beta'
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.param_layer(x)
        x = self.indep_Beta_distribution(x)
        return x



# BETA VAE CLASS
class Std_VAE_Beta(tfk.Model):
    def __init__(self, latent_dim, input_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_VAE_Beta, self).__init__()
        self.encoder = Std_Encoder_Beta(latent_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT)
        self.decoder = Std_Decoder_Beta(input_dim,LAYER_1_N, LAYER_2_N)

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))