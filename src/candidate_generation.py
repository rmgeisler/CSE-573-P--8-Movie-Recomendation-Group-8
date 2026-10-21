def generate_candidates(item_knn, user_idx, seen_movies, k=100):
    return item_knn.recommend(
        user_idx,
        seen_movies,
        k=k
    )
