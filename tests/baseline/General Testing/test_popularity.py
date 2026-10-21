import pandas as pd
from src.models.popularity import PopularityModel

train = pd.read_parquet(
    "data/processed/train.parquet",
    columns=["user_idx", "movie_idx", "rating"]
)

model = PopularityModel()
model.fit(train)

seen_movies = set(
    train[train["user_idx"] == 0]["movie_idx"]
)

recommendations = model.recommend(seen_movies, k=10)

metadata = pd.read_parquet(
    "data/processed/metadata.parquet",
    columns=["movie_idx", "title"]
)

results = metadata[metadata["movie_idx"].isin(recommendations)].copy()
results["rank"] = results["movie_idx"].map(
    {movie: i + 1 for i, movie in enumerate(recommendations)}
)

print(results.sort_values("rank")[["rank", "title"]].to_string(index=False))