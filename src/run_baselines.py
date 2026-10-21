
import pandas as pd

from src.models.popularity import PopularityModel
from src.models.item_knn import ItemKNN
from src.models.als import ALSModel


train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")

models = {
    "Popularity": PopularityModel(),
    "Item-KNN": ItemKNN(neighbors=50),
    "ALS": ALSModel(factors=64, regularization=0.01, iterations=20)
}

for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"Training {name}")
    print(f"{'=' * 40}")

    model.fit(train)

    print(f"{name} trained successfully.")
