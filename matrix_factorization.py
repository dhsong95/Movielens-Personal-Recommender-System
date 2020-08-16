from implicit.als import AlternatingLeastSquares
import numpy as np

from metrics import n_precision
from metrics import n_recall
from utils import load_ratings
from utils import get_user_movie_dictionary
from utils import split_matrix
from utils import transform_binary_matrix


if __name__ == '__main__':
    # Load and Preprocess Dataset
    rating_df = load_ratings('ratings.csv')
    user2idx, movie2idx = get_user_movie_dictionary(rating_df)
    print(f'# of user: {len(user2idx)}\t# of movie: {len(movie2idx)}')

    rating_matrix, stat =\
        transform_binary_matrix(rating_df, user2idx, movie2idx)
    print(
        f'Positive Feedback: {stat["pos"]}',
        f'\tNegative Feedback: {stat["neg"]}'
    )

    rating_matrix_train, rating_matrix_val =\
        split_matrix(rating_matrix, user2idx, movie2idx)

    print(
        f'Train: {rating_matrix_train.count_nonzero()}\t',
        f'Validation Size: {rating_matrix_val.count_nonzero()}'
    )

    # Train ALS Model
    model = AlternatingLeastSquares(
        factors=20,
        iterations=50,
        calculate_training_loss=True,
        num_threads=4
    )
    model.fit(rating_matrix_train.T)

    # Make Prediction
    recommendations = model.recommend_all(
        user_items=rating_matrix_train,
        N=100
    )

    # Evaluate
    precison_100 = n_precision(recommendations, rating_matrix_val, 100)
    recall_100 = n_recall(recommendations, rating_matrix_val, 100)
    print(f'P@100 : {precison_100:.2%}')
    print(f'R@100 : {recall_100:.2%}')

    # Save Recommendation
    np.savez('./output/rec_mf.npz', recommendations)
