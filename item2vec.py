from itertools import combinations

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dot, Embedding, Reshape
import numpy as np


class Item2Vector:
    def __init__(self, item_dim=9724, embedding_dim=100):
        self.model = self.build_model(item_dim, embedding_dim)

    def build_model(self, item_dim, embedding_dim):
        item_center = keras.Input((1,))
        item_context = keras.Input((1,))

        embedding_layer = Embedding(item_dim, embedding_dim, input_length=1)

        center_embedding = embedding_layer(item_center)
        context_embedding = embedding_layer(item_context)

        output = Dot(axes=2)([center_embedding, context_embedding])
        output = Reshape((1,), input_shape=(1, 1))(output)
        output = Activation('sigmoid')(output)

        model = keras.models.Model(
            inputs=[item_center, item_context],
            outputs=output
        )
        model.summary()
        return model

    def train(self, rating, epochs=30, batch_size=100):
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        for epoch in range(epochs):
            indices = np.random.randint(0, rating.shape[0], batch_size)
            loss = 0
            for rdx in indices:
                positives = list()
                negatives = list()
                for cdx in rating[rdx, :].nonzero()[1]:
                    if rating[rdx, cdx] == 1:
                        positives.append(cdx)
                    elif rating[rdx, cdx] == -1:
                        negatives.append(cdx)

                centers = list()
                contexts = list()
                labels = list()
                for center, context in combinations(positives, 2):
                    centers.append(center)
                    contexts.append(context)
                    labels.append(1)

                for center, context in combinations(negatives, 2):
                    centers.append(center)
                    contexts.append(context)
                    labels.append(0)

                centers = np.array(centers)
                contexts = np.array(contexts)
                labels = np.array(labels)

                loss += self.model.train_on_batch([centers, contexts], labels)
            print(f'@Epoch {epoch:04d}\tLoss {loss:.4f}')

    def get_embeddings(self):
        return self.model.get_weights()[0]
