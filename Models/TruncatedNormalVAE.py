import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import numpy as np
warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

### THIS SCRIPT IMPLEMENTS A VAE IN 2 STEPS:
# 1) The encoder learns the parameters of a Multivariate Truncated Normal distribution over [0,1]^d
# These parameters are mu and sigma where mu \in [0,1] and sigma>>0 ideally
# 2) The prior distribution used  is a Multivariate Truncated Normal distribution over [0,1]^d
# with parameters \mu = 0.5 and \sigma = 4, which is an approximation of Uniform[0,1]
# 3) Finally, we use again a Truncated Normal decoder

class Std_Encoder_TruncatedNormal(tfk.Model):  # Encodeur

    def __init__(self, encoded_size, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_Encoder_TruncatedNormal, self).__init__()
        self.encoded_size = encoded_size  # taille de l'espace latent
        self.prior = tfd.Independent(distribution = tfd.TruncatedNormal(loc = 0.5*np.ones(self.encoded_size, dtype=np.float32),
                                                                        scale = 4*np.ones(self.encoded_size, dtype=np.float32),
                                                                        low = np.zeros(self.encoded_size, dtype=np.float32),
                                                                        high = np.ones(self.encoded_size, dtype=np.float32)),
                                     reinterpreted_batch_ndims = 1)
        self.dense1 = tfkl.Dense(units=LAYER_1_N, activation='relu')
        self.dense2 = tfkl.Dense(units=LAYER_2_N, activation='relu')
        self.dense3 = tfkl.Dense(2*self.encoded_size)
        self.ind_TruncNorm = tfpl.DistributionLambda(
            lambda params: tfd.Independent(
                tfd.TruncatedNormal(loc=tf.nn.sigmoid(params[..., :self.encoded_size]),
                                    scale= 0.001 + tf.nn.softplus(params[..., self.encoded_size:]),
                                    low = np.zeros(self.encoded_size, dtype=np.float32),
                                    high = np.ones(self.encoded_size, dtype=np.float32)),
                reinterpreted_batch_ndims=1),
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=KL_WEIGHT),
            name='NormalTruncated_encoded_parameters'
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.ind_TruncNorm(x)
        return x


class Std_Decoder_TruncatedNormal(tfk.Model):
    def __init__(self, input_dim, LAYER_1_N, LAYER_2_N):
        super(Std_Decoder_TruncatedNormal,self).__init__()
        self.K = input_dim
        self.dense1 = tfkl.Dense(units = LAYER_2_N, use_bias=True, activation='relu')
        self.dense2 = tfkl.Dense(units = LAYER_1_N, use_bias=True, activation='relu')
        self.param_layer = tfkl.Dense(2*self.K) #alphas and betas of Beta distribution
        self.indep_TruncNorm = tfpl.DistributionLambda(
            lambda params : tfd.Independent(
                tfd.TruncatedNormal(loc=tf.nn.sigmoid(params[..., :self.K]),
                                    scale=0.001 + tf.nn.softplus(params[..., self.K:]),
                                    low = np.zeros(self.K, dtype=np.float32),
                                    high = np.ones(self.K, dtype=np.float32)),
                reinterpreted_batch_ndims=1
            ),
            name = 'truncated_normal_decoder'
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.param_layer(x)
        x = self.indep_TruncNorm(x)
        return x



####
class Std_VAE_TruncatedNormal(tfk.Model):
    def __init__(self, latent_dim, input_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_VAE_TruncatedNormal, self).__init__()
        self.encoder = Std_Encoder_TruncatedNormal(latent_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT)
        self.decoder = Std_Decoder_TruncatedNormal(input_dim,LAYER_1_N, LAYER_2_N)

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))

