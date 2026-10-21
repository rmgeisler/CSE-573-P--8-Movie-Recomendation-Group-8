
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSModel:
    def __init__(self, factors=64, regularization=0.1, iterations=20):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, train):
        self.n_users = train["user_idx"].max() + 1
        self.n_movies = train["movie_idx"].max() + 1

        interactions = train[train["rating"] >= 4.0]

        self.user_movie_matrix = csr_matrix((
                interactions["rating"].values, (
                    interactions["user_idx"].values,
                    interactions["movie_idx"].values
                )
            ),
            shape=(self.n_users, self.n_movies)
        )

        self.model = AlternatingLeastSquares(
        factors=self.factors,
        iterations=self.iterations,
        regularization=self.regularization,
        random_state=42
)

        self.model.fit(self.user_movie_matrix)


    def recommend(self, user_idx, seen_movies, k=100):
        recommendations = self.model.recommend(
            user_idx,
            self.user_movie_matrix[user_idx],
            N=k,
            filter_already_liked_items=True
        )

        return recommendations[0].tolist()
