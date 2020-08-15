from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class AutoEncoder:
    def __init__(self, input_dim=9724, encoding_dim=20):
        self.autoencoder = self.build_autoencoder(input_dim, encoding_dim)

    def build_autoencoder(self, input_dim, encoding_dim):
        model = keras.models.Sequential()

        model.add(Dense(256, input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(encoding_dim, activation='tanh'))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(input_dim, activation='tanh'))

        model.summary()

        rating = keras.Input(shape=(input_dim))
        rating_hat = model(rating)

        return keras.Model(rating, rating_hat)

    def loss_function(self, y_true, y_pred):
        N = K.sum(tf.cast((y_true == 1.0) | (y_true == -1.0), tf.float32))

        mask = y_true != 0
        mask = tf.cast(mask, tf.float32)
        y_pred = (y_pred + 1) / 2
        y_true = (y_true + 1) / 2

        total_cost =\
            mask * (-y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred))
        total_cost = K.sum(total_cost)

        return total_cost / N

    def train(self, rating, epochs=50, batch_size=100):
        optimizer = Adam(learning_rate=0.0003)
        self.autoencoder.compile(
            optimizer=optimizer,
            loss=self.loss_function
        )
        self.autoencoder.fit(
            rating,
            rating,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )

    def predict(self, rating, n=100):
        predictions = self.autoencoder.predict(rating)
        for uidx, items in enumerate(rating):
            for iidx, item in enumerate(items):
                if item == 1 or item == -1:
                    predictions[uidx, iidx] = -10
        recommendations = np.argsort(predictions)[:, ::-1][:, :n]
        return recommendations
