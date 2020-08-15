from item2vec import Item2Vector
from metrics import n_precision
from metrics import n_recall
from utils import load_ratings
from utils import get_user_movie_dictionary
from utils import split_matrix
from utils import transform_binary_matrix


if __name__ == '__main__':
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

    model = Item2Vector(item_dim=len(movie2idx), embedding_dim=100)
    model.train(rating_matrix_train)

    embeddings = model.get_embeddings()
    print(embeddings.shape)
