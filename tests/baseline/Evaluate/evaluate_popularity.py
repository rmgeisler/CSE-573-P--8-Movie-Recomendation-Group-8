
import pandas as pd
import numpy as np
from src.models.popularity import PopularityModel

train = pd.read_parquet(
    "data/processed/train.parquet",
    columns=["user_idx", "movie_idx", "rating"]
)

val = pd.read_parquet(
    "data/processed/val.parquet",
    columns=["user_idx", "movie_idx", "rating"]
)

model = PopularityModel()
model.fit(train)

ranked_movies = model.popularity.index.tolist()

train_seen = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()
val_relevant = (
    val[val["rating"] >= 4.0]
    .groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)

recall_scores = []
precision_scores = []
ndcg_scores = []

for user, relevant in val_relevant.items():
    seen = train_seen.get(user, set())

    recommendations = [
        movie for movie in ranked_movies
        if movie not in seen
    ][:100]

    if not relevant:
        continue

    hits = sum(movie in relevant for movie in recommendations)
    recall_scores.append(hits / len(relevant))

    if recommendations:
        precision_scores.append(hits / len(recommendations))

    top10 = recommendations[:10]
    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, movie in enumerate(top10)
        if movie in relevant
    )

    ideal_hits = min(len(relevant), 10)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

print(f"Recall@100:    {np.mean(recall_scores):.4f}")
print(f"Precision@100: {np.mean(precision_scores):.4f}")
print(f"NDCG@10:       {np.mean(ndcg_scores):.4f}")