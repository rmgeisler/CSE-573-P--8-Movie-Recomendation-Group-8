class PopularityModel:
    def fit(self, train):
        self.popularity = (
            train[train["rating"] >= 4.0]
            .groupby("movie_idx")
            .size()
            .sort_values(ascending=False)
        )

    def recommend(self, seen_movies, k=100):
        recommendations = self.popularity[
            ~self.popularity.index.isin(seen_movies)
        ]
        return recommendations.head(k).index.tolist()