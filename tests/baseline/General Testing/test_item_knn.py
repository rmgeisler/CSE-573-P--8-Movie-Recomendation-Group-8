import pandas as pd
from src.models.item_knn import ItemKNN
import numpy as np

train = pd.read_parquet(
    "data/processed/train.parquet",
    columns=["user_idx", "movie_idx", "rating"]
)

model = ItemKNN()
model.fit(train)

print("Matrix shape:", model.user_movie_matrix.shape)
print("Stored ratings:", model.user_movie_matrix.nnz)
print("Expected ratings:", len(train))
print("Storage format:", model.user_movie_matrix.format)


print("Similarity matrix shape:", model.item_similarity.shape)
print("Stored similarities:", model.item_similarity.nnz)

movie_idx = 0
similar_movies = model.item_similarity[movie_idx].toarray()[0]
top_neighbors = np.argsort(similar_movies)[-5:][::-1]

print("Top 5 similar movie indices:", top_neighbors)
print("Similarity scores:", similar_movies[top_neighbors])

user_idx = 0
seen_movies = set(
    train[train["user_idx"] == user_idx]["movie_idx"]
)

recommendations = model.recommend(user_idx, seen_movies, k=10)

print("Recommended movie indices:", recommendations)
