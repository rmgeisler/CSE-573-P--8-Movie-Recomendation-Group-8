
import pandas as pd
from src.models.als import ALSModel

train = pd.read_parquet(
    "data/processed/train.parquet",
    columns=["user_idx", "movie_idx", "rating"]
)

model = ALSModel()
model.fit(train)

print("User factors shape:", model.model.user_factors.shape)
print("Movie factors shape:", model.model.item_factors.shape)

user_idx = 0

seen_movies = set(
    train[train["user_idx"] == user_idx]["movie_idx"]
)

recommendations = model.recommend(
    user_idx, seen_movies, k=10
)

print("Recommended movie indices:", recommendations)
print("Number of recommendations:", len(recommendations))
