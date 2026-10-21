
import pandas as pd

from src.models.item_knn import ItemKNN
from src.evaluation import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k
)

train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")

train_seen = (
    train.groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)

relevant_val = (
    val[val["rating"] >= 4.0]
    .groupby("user_idx")["movie_idx"]
    .apply(set)
    .to_dict()
)

neighbor_counts = [20, 50, 100]
results = []

for neighbors in neighbor_counts:
    print(f"\nTesting {neighbors} neighbors...")

    model = ItemKNN(neighbors=neighbors)
    model.fit(train)

    recalls = []
    precisions = []
    ndcgs = []

    for user_idx, relevant_movies in relevant_val.items():
        seen_movies = train_seen.get(user_idx, set())

        recommendations = model.recommend(
            user_idx,
            seen_movies,
            k=100
        )

        recalls.append(
            recall_at_k(recommendations, relevant_movies, k=100)
        )
        precisions.append(
            precision_at_k(recommendations, relevant_movies, k=100)
        )
        ndcgs.append(
            ndcg_at_k(recommendations, relevant_movies, k=10)
        )

    result = {
        "neighbors": neighbors,
        "recall@100": sum(recalls) / len(recalls),
        "precision@100": sum(precisions) / len(precisions),
        "ndcg@10": sum(ndcgs) / len(ndcgs)
    }

    results.append(result)

    print(f"Recall@100:    {result['recall@100']:.4f}")
    print(f"Precision@100: {result['precision@100']:.4f}")
    print(f"NDCG@10:       {result['ndcg@10']:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ndcg@10", ascending=False)

print("\nItem-KNN Tuning Results:")
print(results_df.to_string(index=False))

print("\nBest configuration by NDCG@10:")
print(results_df.iloc[0].to_string())
