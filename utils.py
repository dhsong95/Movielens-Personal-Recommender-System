from collections import defaultdict
import os

from scipy import sparse
from tqdm import tqdm
import numpy as np
import pandas as pd


def load_ratings(filename):
    dirpath = './data/ml-latest-small'
    ratings = pd.read_csv(os.path.join(dirpath, filename))
    return ratings


def get_user_movie_dictionary(dataframe):
    users = dataframe.userId.unique()
    movies = dataframe.movieId.unique()

    user2idx = {user: idx for idx, user in enumerate(users)}
    movie2idx = {movie: idx for idx, movie in enumerate(movies)}

    return user2idx, movie2idx


def transform_binary_matrix(dataframe, user2idx, movie2idx):
    rows = list()
    cols = list()
    data = list()

    stat = defaultdict(int)

    for user, movie, rating in zip(
            dataframe['userId'], dataframe['movieId'], dataframe['rating']):
        user_idx = user2idx[user]
        movie_idx = movie2idx[movie]

        rows.append(user_idx)
        cols.append(movie_idx)
        if rating >= 2.0:
            data.append(1.0)
            stat['pos'] += 1
        else:
            data.append(-1.0)
            stat['neg'] += 1

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user2idx), len(movie2idx))
    )
    return matrix, stat


def split_matrix(original, user2idx, movie2idx):
    np.random.seed(2020)

    N_user = original.shape[0]
    N_movie = original.shape[1]

    rows_tr = list()
    cols_tr = list()
    data_tr = list()

    rows_val = list()
    cols_val = list()
    data_val = list()

    for rdx, cdx in tqdm(zip(*original.nonzero())):
        rated_movie = len(original[rdx, :].nonzero()[1])
        rated_user = len(original[:, cdx].nonzero()[0])

        threshold = (rated_movie / N_movie) * (rated_user / N_user) + 0.8
        random_number = np.random.rand()
        if random_number <= threshold:
            rows_tr.append(rdx)
            cols_tr.append(cdx)
            data_tr.append(original[rdx, cdx])
        else:
            rows_val.append(rdx)
            cols_val.append(cdx)
            data_val.append(original[rdx, cdx])

    train_matrix = sparse.csr_matrix(
        (data_tr, (rows_tr, cols_tr)), shape=(len(user2idx), len(movie2idx))
    )
    validation_matrix = sparse.csr_matrix(
        (data_val, (rows_val, cols_val)), shape=(len(user2idx), len(movie2idx))
    )

    return train_matrix, validation_matrix
