import json
import pandas as pd
import numpy as np
import faiss
import traceback
from tqdm.auto import tqdm

class MovieRAG:

    def __init__(
        self,
        client,
        embedding_model="qwen3-embedding-4b",
        chat_model="qwen35-27b",
        batch_size=32
    ):

        self.client = client
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.batch_size = batch_size

        self.index = None
        self.movies = None
        self.embeddings = None
        self.documents = None

        self.train = None
        self.kg_retriever = None
        self.semantic_neighbors = None

    # Convert metadata values safely into text
    def _value_to_text(self, value):

        if value is None:
            return ""

        if isinstance(value, (list, tuple, np.ndarray)):
            return ", ".join(
                str(v) for v in value
                if v is not None
            )

        # if not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value):
        #     return ""

        if isinstance(value, dict):
            return json.dumps(value, default=str)
    
        if pd.isna(value):
            return ""
        return str(value).strip()

    # Prepare movie information as text
    def prepare_documents(self, metadata):

        documents = []

        text_columns = [
            "title",
            "genres",
            "overview",
            "plot",
            "director",
            "directors",
            "cast",
            "keywords",
            "tagline",
            "tmdb_genres"
        ]

        available_columns = [
            col for col in text_columns
            if col in metadata.columns
        ]

        if not available_columns:
            raise ValueError(
                "No supported text columns found in metadata."
            )

        for _, row in metadata.iterrows():

            parts = []

            for col in available_columns:

                value = self._value_to_text(row[col])

                if value:
                    parts.append(f"{col}: {value}")

            documents.append(" ".join(parts))

        self.movies = metadata.reset_index(drop=True).copy()
        self.documents = documents

        return documents

    # Generate embeddings using ASU API in batches
    def _embed_texts(self, texts):

        all_embeddings = []

        with tqdm(
            total=len(texts),
            desc="Generating Movie Embeddings",
            unit="movie",
            dynamic_ncols=True
        ) as pbar:

            for start in range(0, len(texts), self.batch_size):

                batch = texts[start:start + self.batch_size]

                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )

                batch_embeddings = sorted(
                    response.data,
                    key=lambda x: x.index
                )

                all_embeddings.extend(
                    item.embedding for item in batch_embeddings
                )

                pbar.update(len(batch))

        embeddings = np.array(
            all_embeddings,
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        return embeddings
    
    # Generate movie embeddings and build FAISS index
    def fit(self, metadata):

        documents = self.prepare_documents(metadata)

        self.embeddings = self._embed_texts(documents)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(self.embeddings)

        print("RAG index created successfully!")
        print("Total movies:", self.index.ntotal)
        print("Embedding dimensions:", dimension)

        # Connect KG and semantic retrieval using training data
    def configure_evidence(
        self,
        train,
        kg_retriever,
        semantic_neighbors
    ):
        self.train = train.copy()

        self.kg_retriever = kg_retriever
        self.kg_retriever.fit(self.train)

        self.semantic_neighbors = semantic_neighbors
        self.semantic_neighbors.fit(self.train)

        print("KG and semantic retrieval configured successfully!")

    # Get the user's training history
    def _get_user_history(self, user_idx, max_history=20):

        if self.train is None:
            raise ValueError("Please configure evidence retrievers first.")

        history = self.train[
            self.train["user_idx"] == user_idx
        ].copy()

        if history.empty:
            raise ValueError(f"No training history found for user {user_idx}.")

        # Prefer movies rated 4 or 5.
        liked = history[history["rating"] >= 4]

        # If the user has no ratings >= 4, use their highest-rated movies.
        if liked.empty:
            max_rating = history["rating"].max()
            liked = history[history["rating"] == max_rating]

        # Use a manageable amount of history for the LLM prompt.
        liked = liked.sort_values(
            ["rating", "timestamp"],
            ascending=[False, False]
        ).head(max_history)

        liked_ids = liked["movie_idx"].astype(int).tolist()

        history_documents = []

        for movie_idx in liked_ids:

            movie = self.movies[
                self.movies["movie_idx"] == movie_idx
            ]

            if not movie.empty:
                # movie_row = movie.iloc[0]
                movie_position = movie.index[0]

                rating = liked.loc[
                    liked["movie_idx"] == movie_idx, "rating"
                ].iloc[0]
                
                # history_documents.append(
                #     f"Rating: {liked.loc[
                #         liked['movie_idx'] == movie_idx, 'rating'
                #     ].iloc[0]}"
                #     f"{self.documents[movie_position]}"
                # )
                history_documents.append(
                    f"Rating: {rating}\n"
                    f"{self.documents[movie_position]}"
                )
        # Check after processing all liked movies
        if not history_documents:
            raise ValueError(
                f"No matching movie metadata found for user {user_idx}."
            )

        user_query = "\n".join(history_documents)

        return history, liked_ids, user_query

    # Build evidence for each user-candidate movie pair
    def _build_evidence(self, user_idx, candidates):

        evidence_blocks = []

        for i, (_, movie) in enumerate(candidates.iterrows()):

            movie_idx = int(movie["movie_idx"])

            # Candidate movie metadata
            candidate_details = {}

            for col in [
                "title",
                "genres",
                "overview",
                "director",
                "directors",
                "cast",
                "keywords"
            ]:

                if col in movie.index:
                    value = self._value_to_text(movie[col])

                    if value:
                        candidate_details[col] = value[:500]

            # Knowledge graph paths connecting the candidate
            # to the user's liked training movies
            kg_paths = self.kg_retriever.explain(
                user_idx=user_idx,
                movie_idx=movie_idx
            )

            # Plot-similar movies from the user's training history
            semantic_neighbors = self.semantic_neighbors.neighbors(
                user_idx=user_idx,
                movie_idx=movie_idx,
                k=3,
                min_rating=4.0
            )

            evidence = {
                "candidate_id": i,
                "movie_idx": movie_idx,
                "candidate_metadata": candidate_details,
                "kg_paths": kg_paths,
                "semantic_neighbors": semantic_neighbors
            }

            evidence_blocks.append(json.dumps(evidence, default=str))

        return evidence_blocks

    # Retrieve movies relevant to a query
    def retrieve(
        self,
        query,
        top_k=10,
        exclude_movie_ids=None,
        candidate_k=None,
        rerank=False
    ):

        if self.index is None:
            raise ValueError("Please fit the RAG model first.")

        if exclude_movie_ids is None:
            exclude_movie_ids = set()
        else:
            exclude_movie_ids = set(exclude_movie_ids)

        if candidate_k is None:
            candidate_k = top_k

        # Embed the search query using the same model
        query_embedding = self._embed_texts([query])

        # Retrieve extra results to account for exclusions
        search_k = min(
            self.index.ntotal,
            candidate_k + len(exclude_movie_ids)
        )

        scores, indices = self.index.search(
            query_embedding,
            search_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            movie = self.movies.iloc[idx]

            movie_id = movie.get(
                "movie_idx",
                movie.get("movie_id", idx)
            )

            if int(movie_id) in exclude_movie_ids:
                continue

            result = movie.to_dict()
            result["similarity_score"] = float(score)

            results.append(result)

            if len(results) == candidate_k:
                break

        candidates = pd.DataFrame(results)

        if rerank and not candidates.empty:
            candidates = self.rerank(
                query,
                candidates,
                top_k
            )

            return candidates.head(top_k).reset_index(drop=True)

        return candidates.head(candidate_k).reset_index(drop=True)

    

    # Rerank candidates using the ASU chat API and retrieved evidence
    def rerank(
        self,
        user_query,
        candidates,
        top_k=10,
        evidence_blocks=None
    ):

        if candidates.empty:
            return candidates
            

        candidate_details = []

        for i, (_, movie) in enumerate(candidates.iterrows()):

            if evidence_blocks is not None:
                details = evidence_blocks[i]

            else:
                details = []

                for col in [
                    "title",
                    "genres",
                    "overview",
                    "director",
                    "cast"
                ]:

                    if col in movie.index:
                        value = self._value_to_text(movie[col])

                        if value:
                            details.append(f"{col}: {value[:500]}")

                details = " | ".join(details)

            candidate_details.append(
                f"Candidate {i}:\n{details}"
            )

        prompt = f"""
You are a movie recommendation expert.

User's interests based on their training history:
{user_query[:2000]}

Rank the candidate movies using the supplied evidence.

Consider:
- Similarity to movies the user liked
- Shared directors, actors, genres, and keywords
- Plot similarity from semantic neighbors
- Candidate movie attributes
- KG paths connecting the candidate to the user's history

Use the retrieved evidence. Do not invent relationships or facts.
Prefer candidates with stronger and more relevant evidence.

Return ONLY valid JSON in this format:

{{
    "ranked_ids": [2, 0, 4, 1, 3]
}}

Requirements:
- Include every candidate ID exactly once.
- Return IDs in relevance order, from best to worst.
- Use only the candidate IDs provided.
- Do not invent candidate IDs.
- Base the ranking only on the supplied evidence.
Include candidate IDs in relevance order, from best to worst.
Do not invent candidate IDs.

Candidates and evidence:
{chr(10).join(candidate_details)}
"""
        print("Prompt length:", len(prompt))
        print("Number of candidates:", len(candidates))


        try:
            response = self.client.chat.completions.create(
            model=self.chat_model,
            reasoning_effort="none",
            messages=[
                    {
                        "role": "system",
                        "content": "You are a movie recommendation ranking assistant. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            message = response.choices[0].message
            content = message.content
            finish_reason = response.choices[0].finish_reason
            # content = response.choices[0].message.content.strip()
            if content is None:
                print("LLM returned empty content.")
                print("Finish reason:", response.choices[0].finish_reason)
                print("Message:", message)
                raise ValueError("LLM response content is None")
            
            content = content.strip()
            
            if not content:
                print("LLM returned empty content.")
                print("Finish reason:", finish_reason)
                raise ValueError("LLM returned an empty response")
            
            start = content.find("{")
            end = content.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON object returned.")

            data = json.loads(content[start:end])

            ranked_ids = data["ranked_ids"]

            # Keep only valid, unique candidate positions
            valid_ids = []

            for i in ranked_ids:

                if (
                    isinstance(i, int)
                    and not isinstance(i, bool)
                    and 0 <= i < len(candidates)
                    and i not in valid_ids
                ):
                    valid_ids.append(i)

            # Append candidates omitted by the LLM
            valid_ids += [
                i for i in range(len(candidates))
                if i not in valid_ids
            ]

            return candidates.iloc[valid_ids].head(top_k).reset_index(drop=True)

        except Exception as e:
            print("LLM reranking failed:", e)
            traceback.print_exc()
            print("Using candidate order.")
            return candidates.head(top_k).reset_index(drop=True)
   
    # Retrieve and rerank movies using user history and all available evidence
    def recommend_for_user(
        self,
        user_idx,
        top_k=10,
        candidate_k=100,
        knn_candidates=None
    ):

        if self.kg_retriever is None or self.semantic_neighbors is None:
            raise ValueError(
                "Please call configure_evidence() first."
            )

        # Get the user's training history
        history, liked_ids, user_query = self._get_user_history(user_idx)

        # Exclude all movies already seen in training
        seen_ids = set(
            history["movie_idx"].astype(int).tolist()
        )

        # 1. RAG candidates from FAISS
        rag_candidates = self.retrieve(
            query=user_query,
            candidate_k=candidate_k,
            exclude_movie_ids=seen_ids
        )
        print("RAG candidates:", len(rag_candidates))
        # 2. KG candidates
        kg_ids = self.kg_retriever.recommend(
            user_idx=user_idx,
            seen=seen_ids,
            k=candidate_k
        )
        print("KG candidates:", len(kg_ids))
        # 3. Optional Item-KNN candidates
        candidate_sources = []

        def add_source(movie_ids, source_name):

            for rank, movie_id in enumerate(movie_ids, start=1):

                candidate_sources.append({
                    "movie_idx": int(movie_id),
                    "source": source_name,
                    "rank": rank
                })

        if not rag_candidates.empty:
            add_source(
                rag_candidates["movie_idx"].tolist(),
                "rag"
            )

        add_source(kg_ids, "kg")

        if knn_candidates is not None:

            if isinstance(knn_candidates, pd.DataFrame):
                knn_ids = knn_candidates["movie_idx"].tolist()
            else:
                knn_ids = list(knn_candidates)
            knn_ids = [
                movie_id for movie_id in knn_ids
                if int(movie_id) not in seen_ids
            ]

            add_source(knn_ids, "knn")

        if not candidate_sources:
            return pd.DataFrame()

        # Combine candidate sources using reciprocal rank fusion.
        # This avoids directly comparing FAISS and KG score scales.
        source_df = pd.DataFrame(candidate_sources)

        source_df["rrf_score"] = 1.0 / (60 + source_df["rank"])

        combined = (
            source_df.groupby("movie_idx")
            .agg(
                rrf_score=("rrf_score", "sum"),
                candidate_source=(
                    "source",
                    lambda x: ", ".join(sorted(set(x)))
                )
            )
            .reset_index()
            .sort_values("rrf_score", ascending=False)
            .head(candidate_k)
        )

        # Retrieve metadata for the combined candidates
        candidates = combined.merge(
            self.movies,
            on="movie_idx",
            how="inner"
        )

        if candidates.empty:
            return candidates

        candidates = candidates.reset_index(drop=True)

        print("Starting evidence construction...")
        # Build one evidence block per user-candidate pair
        evidence_blocks = self._build_evidence(
            user_idx,
            candidates
        )
        print("Evidence construction completed!")
        # Rerank using the complete evidence
        print("Starting LLM reranking...")
        result = self.rerank(
            user_query=user_query,
            candidates=candidates,
            top_k=top_k,
            evidence_blocks=evidence_blocks
        )
        # print("LLM reranking completed!")
        return result
