
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:
    def __init__(self, neighbors=50):
        self.neighbors = neighbors
    def fit(self, train):
        self.n_users = train["user_idx"].max() + 1
        self.n_movies = train["movie_idx"].max() + 1

        self.user_movie_matrix = csr_matrix((
            train["rating"].values,(
                train["user_idx"].values,
                train["movie_idx"].values
            )),
            shape=(self.n_users, self.n_movies)
        )

        item_matrix = self.user_movie_matrix.T.tocsr()

        k = self.neighbors
        batch_size = 128

        rows = []
        cols = []
        values = []

        for start in range(0, self.n_movies, batch_size):
            end = min(start + batch_size, self.n_movies)

            similarities = cosine_similarity(
                item_matrix[start:end],
                item_matrix
            )

            for i in range(end - start):
                movie_idx = start + i
                scores = similarities[i]

                neighbors = np.argpartition(
                    scores, -min(k + 1, self.n_movies)
                )[-min(k + 1, self.n_movies):]

                neighbors = neighbors[neighbors != movie_idx]
                neighbors = neighbors[scores[neighbors] > 0]

                neighbors = neighbors[
                    np.argsort(scores[neighbors])[-k:]
                ]

                rows.extend([movie_idx] * len(neighbors))
                cols.extend(neighbors.tolist())
                values.extend(scores[neighbors].tolist())

        self.item_similarity = csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_movies, self.n_movies)
        )


    def recommend(self, user_idx, seen_movies, k=100):
        user_ratings = self.user_movie_matrix[user_idx]

        #scores = user_ratings.dot(self.item_similarity).toarray().flatten()
        scores = user_ratings.dot(self.item_similarity.T).toarray().ravel()

        scores[ list(seen_movies) ] = 0

        recommendations = np.argsort(scores)[::-1]

        recommendations = [
            movie for movie in recommendations
            if scores[movie] > 0
        ]

        return recommendations[:k]