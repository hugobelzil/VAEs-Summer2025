import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import numpy as np
warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions



# MULTIVARIATE GAUSSIAN ENCODER AND GAUSSIAN PRIOR

class Std_Encoder_GaussianMixture (tfk.Model):  # Encodeur

    def __init__(self, encoded_size, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_Encoder_LogitNormal, self).__init__()
        self.encoded_size = encoded_size  # taille de l'espace latent
        #self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.encoded_size))
        #self.covariance_matrix = 0.3*tf.eye(encoded_size, dtype=tf.float32)
        #self.covariance_matrix += 0.7*tf.ones((encoded_size, encoded_size), dtype=tf.float32)
        #self.prior = tfd.MultivariateNormalFullCovariance(covariance_matrix=self.covariance_matrix)
        self.prior = tfd.Independent(tfd.LogitNormal(loc=tf.zeros(encoded_size), scale=tf.ones(encoded_size)),
                                     reinterpreted_batch_ndims=1)
        self.dense1 = tfkl.Dense(units = LAYER_1_N, activation='leaky_relu')
        self.dense2 = tfkl.Dense(units = LAYER_2_N, activation='leaky_relu')
        self.dense3 = tfkl.Dense(tfpl.IndependentNormal.params_size(self.encoded_size))
        self.ind_logit_normal = tfpl.DistributionLambda(
            lambda params: tfd.Independent(
                tfd.LogitNormal(
                    loc=params[..., :encoded_size],
                    scale=1e-3 + tf.nn.softplus(params[..., encoded_size:])
                ),
                reinterpreted_batch_ndims=1
            ),
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=KL_WEIGHT)
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.ind_logit_normal(x)
        return x

    def encode_params(self, inputs):
        """Returns loc and scale parameters before wrapping them in a distribution."""
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        loc = x[..., :self.encoded_size]
        scale = 1e-3 + tf.nn.softplus(x[..., self.encoded_size:])
        return loc, scale

class Std_Decoder_LogitNormal(tfk.Model):  # Décodeur
    def __init__(self, input_dim, LAYER_1_N, LAYER_2_N):
        super(Std_Decoder_LogitNormal, self).__init__()
        self.K = input_dim
        self.dense1 = tfkl.Dense(units = LAYER_2_N, use_bias=True, activation='leaky_relu')
        self.dense2 = tfkl.Dense(units = LAYER_1_N, use_bias=True, activation='leaky_relu')
        self.param_layer = tfkl.Dense(2*self.K)
        self.ind_logit_normal = tfpl.DistributionLambda(
            lambda params : tfd.Independent(
                tfd.LogitNormal(loc = params[...,:self.K], scale = 1e-3+ tf.nn.softplus(params[...,self.K:])),
            reinterpreted_batch_ndims=1),
        name = 'logit_normal_output'
        )

    def encode_params(self, inputs):
        """Returns loc and scale parameters before wrapping them in a distribution."""
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.param_layer(x)
        loc = x[..., :self.K]
        scale = 1e-3 + tf.nn.softplus(x[..., self.K:])
        return loc, scale

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.param_layer(x)
        x = self.ind_logit_normal(x)
        return x


###### STANDARD VAE WITH NORMAL ENCODER, LOGIT-NORMAL DECODER
class Std_VAE_LogitNormal(tfk.Model):
    def __init__(self, latent_dim, input_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_VAE_LogitNormal, self).__init__()
        self.encoder = Std_Encoder_LogitNormal(latent_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT)
        self.decoder = Std_Decoder_LogitNormal(input_dim,LAYER_1_N, LAYER_2_N)


    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

    def get_metrics(self, x):
        rv_x = self(x)
        nll = -rv_x.log_prob(x)
        kl = tf.reduce_mean(tfp.distributions.kl_divergence(self.encoder(x), self.encoder.prior))
        elbo = tf.reduce_mean(nll + 0.1*kl)
        return {
            "elbo": elbo.numpy(),
            "nll": tf.reduce_mean(nll).numpy(),
            "kl": kl.numpy()
        }

