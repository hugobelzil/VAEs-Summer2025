import tensorflow as tf
import tensorflow_probability as tfp
import warnings
warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class Std_Encoder(tfk.Model):  # Encodeur

    def __init__(self, encoded_size, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_Encoder, self).__init__()
        self.encoded_size = encoded_size  # taille de l'espace latent
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.encoded_size))
        self.dense1 = tfkl.Dense(LAYER_1_N, activation='relu')
        self.dense2 = tfkl.Dense(LAYER_2_N, activation='relu')
        self.dense3 = tfkl.Dense(tfpl.IndependentNormal.params_size(self.encoded_size))
        self.ind_norm1 = tfpl.IndependentNormal(self.encoded_size,
                                                activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior,
                                                                                                  weight=KL_WEIGHT))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.ind_norm1(x)
        return x


class Std_Decoder(tfk.Model):  # Décodeur
    def __init__(self, input_dim, LAYER_1_N, LAYER_2_N):
        super(Std_Decoder, self).__init__()
        self.K = input_dim
        self.dense1 = tfkl.Dense(LAYER_2_N, use_bias=True, activation='relu')
        self.dense2 = tfkl.Dense(LAYER_1_N, use_bias=True, activation='relu')
        self.dense3 = tfkl.Dense(tfpl.IndependentNormal(self.K).params_size(self.K))
        self.ind_norm1 = tfpl.IndependentNormal(self.K)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.ind_norm1(x)
        x = tfd.TransformedDistribution(distribution = x, bijector = tfp.bijectors.NormalCDF())
        return x


class Std_VAE_Probit(tfk.Model):
    def __init__(self, latent_dim, input_dim, LAYER_1_N, LAYER_2_N, KL_WEIGHT):
        super(Std_VAE_Probit, self).__init__()
        self.encoder = Std_Encoder(latent_dim,LAYER_1_N, LAYER_2_N, KL_WEIGHT)
        self.decoder = Std_Decoder(input_dim, LAYER_1_N, LAYER_2_N)

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))