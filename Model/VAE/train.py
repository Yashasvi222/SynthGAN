import numpy as np
import tensorflow as tf
import keras
# from tensorflow import keras
from keras.layers import InputLayer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose

import os


class VAE(keras.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential([
            InputLayer(input_shape=input_shape),
            Conv2D(512, 3, activation='relu', strides=2, padding='same'),
            Conv2D(256, 3, activation='relu', strides=2, padding='same'),
            Conv2D(128, 3, activation='relu', strides=2, padding='same'),
            Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            Conv2D(32, 3, activation='relu', strides=(2, 1), padding='same'),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(latent_dim),
        ])

        self.z_mean = Dense(latent_dim, name='z_mean')
        self.z_log_var = Dense(latent_dim, name='z_log_var')

        self.decoder = tf.keras.Sequential([
            InputLayer(input_shape=(latent_dim,)),
            Dense(7 * 7 * 128, activation='relu'),
            Reshape((7, 7, 128)),
            Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same'),
            Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same'),
            Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same'),
            Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            Conv2DTranspose(32, 3, activation='relu', strides=(2,1), padding='same'),
        ])

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def encode(self, data):
        x = self.encoder(data)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        return z_mean, z_log_var

    def reparameterization(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

    def decode(self, data):
        return self.decoder(data)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.reparameterization(z_mean, z_log_var)
            reconstruction = self.decode(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


input_shape = (256, 64, 1)
n_batch = 64
n_epoch = 10
latent_dim = 128
spectrograms_path = "C:\\Users\\yasha\\OneDrive\\Desktop\\music21(another)\\spectrograms"


# Load piano data
def load_paino(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train


x_train = load_paino(spectrograms_path)
vae = VAE()
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=n_epoch, batch_size=n_batch)
