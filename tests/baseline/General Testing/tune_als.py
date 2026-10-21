
import pandas as pd
import numpy as np

from src.models.als import ALSModel
from src.evaluation import recall_at_k, precision_at_k, ndcg_at_k


train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")

train_seen = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()

relevant = (
    val[val["rating"] >= 4.0]
    .groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)

configurations = [
    (32, 0.01),
    (32, 0.1),
    (64, 0.01),
    (64, 0.1),
    (128, 0.01),
    (128, 0.1)
]

results = []

for factors, regularization in configurations:
    print(f"\nTraining ALS: factors={factors}, regularization={regularization}")

    model = ALSModel(
        factors=factors,
        regularization=regularization,
        iterations=20
    )

    model.fit(train)

    recalls = []
    precisions = []
    ndcgs = []

    for user_idx, relevant_movies in relevant.items():
        seen_movies = train_seen.get(user_idx, set())

        recommendations = model.recommend(
            user_idx,
            seen_movies,
            k=100
        )

        recommendations = [
            movie for movie in recommendations
            if movie not in seen_movies
        ][:100]

        recalls.append(
            recall_at_k(recommendations, relevant_movies, k=100)
        )

        precisions.append(
            precision_at_k(recommendations, relevant_movies, k=100)
        )

        ndcgs.append(
            ndcg_at_k(recommendations, relevant_movies, k=10)
        )

    recall = np.mean(recalls)
    precision = np.mean(precisions)
    ndcg = np.mean(ndcgs)

    results.append({
        "factors": factors,
        "regularization": regularization,
        "recall@100": recall,
        "precision@100": precision,
        "ndcg@10": ndcg
    })

    print(f"Recall@100: {recall:.4f}")
    print(f"Precision@100: {precision:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")


results = pd.DataFrame(results)
results = results.sort_values("ndcg@10", ascending=False)

print("\nALS Tuning Results:")
print(results.to_string(index=False))

best = results.iloc[0]

print("\nBest ALS Configuration:")
print(f"Factors: {int(best['factors'])}")
print(f"Regularization: {best['regularization']}")
print(f"NDCG@10: {best['ndcg@10']:.4f}")
