import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from src.eval.metrics import build_genres, build_novelty, score_lists, summarize
from src.eval.run_coldstart import truncate
from src.kg.build_graph import KnowledgeGraph
from src.kg.retrieval import KGRetriever, load_params
from src.kg.sentence_transformer import SemanticNeighbors
from src.models.rag import MovieRAG

BASE_URL = "https://openai.rc.asu.edu/v1"
POOL_K = 100
TOP_K = 10


class EvalRAG(MovieRAG):

    def _embed_texts(self, texts):
        for attempt in range(4):
            try:
                resp = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda x: x.index)], dtype="float32")
        faiss.normalize_L2(vecs)
        return vecs


def embed_movies(rag, workers, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / "movie_embeddings.npz"
    n = len(rag.documents)
    if final.exists():
        d = np.load(final)
        if d["embeddings"].shape[0] == n and d["movie_idx"].tolist() == rag.movies["movie_idx"].tolist() \
                and str(d["model"]) == rag.embedding_model:
            print(f"loaded cached movie embeddings {d['embeddings'].shape}")
            return d["embeddings"]
        print("cached embeddings don't match the metadata / model - recomputing")

    bs, chunk_docs = rag.batch_size, rag.batch_size * 50
    parts = []
    for start in range(0, n, chunk_docs):
        f = cache_dir / f"chunk_{start}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        docs = rag.documents[start:start + chunk_docs]
        batches = [docs[i:i + bs] for i in range(0, len(docs), bs)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vecs = list(tqdm(ex.map(rag._embed_texts, batches), total=len(batches),
                             desc=f"embedding movies {start:,}-{start + len(docs):,}"))
        arr = np.vstack(vecs).astype("float32")
        np.save(f, arr)
        parts.append(arr)
    emb = np.vstack(parts)
    np.savez(final, embeddings=emb, movie_idx=rag.movies["movie_idx"].to_numpy(), model=rag.embedding_model)
    for f in cache_dir.glob("chunk_*.npy"):
        f.unlink()
    print(f"saved {final}  {emb.shape}")
    return emb


def build_pool(rag, kg_retriever, knn_lists, user, seen_full, candidate_k=POOL_K):
    _, _, user_query = rag._get_user_history(user)
    rag_df = rag.retrieve(query=user_query, candidate_k=candidate_k, exclude_movie_ids=seen_full)
    kg_ids = kg_retriever.recommend(user_idx=user, seen=seen_full, k=candidate_k)
    knn_ids = [m for m in knn_lists.get(user, []) if m not in seen_full]

    rows = []
    for source, ids in (("rag", rag_df["movie_idx"].tolist() if not rag_df.empty else []),
                        ("kg", kg_ids), ("knn", knn_ids)):
        rows += [{"movie_idx": int(m), "source": source, "rank": r} for r, m in enumerate(ids, start=1)]
    src = pd.DataFrame(rows)
    src["rrf_score"] = 1.0 / (60 + src["rank"])
    combined = (src.groupby("movie_idx")
                .agg(rrf_score=("rrf_score", "sum"),
                     candidate_source=("source", lambda x: ", ".join(sorted(set(x)))))
                .reset_index().sort_values("rrf_score", ascending=False, kind="stable").head(candidate_k))
    cands = combined.merge(rag.movies, on="movie_idx", how="inner").reset_index(drop=True)
    return user_query, cands


def make_prompt(rag, user_query, candidates, evidence_blocks):
    lines = []
    for i, (_, movie) in enumerate(candidates.iterrows()):
        if evidence_blocks is not None:
            details = evidence_blocks[i]
        else:
            parts = []
            for col in ["title", "genres", "overview", "director", "cast"]:
                if col in movie.index:
                    value = rag._value_to_text(movie[col])
                    if value:
                        parts.append(f"{col}: {value[:500]}")
            details = " | ".join(parts)
        lines.append(f"Candidate {i}:\n{details}")

    return f"""
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
{chr(10).join(lines)}
"""


def call_llm(client, model, prompt, n_candidates, retries):
    last_err, raw, usage = None, None, None
    t0 = time.time()
    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                reasoning_effort="none",
                messages=[
                    {"role": "system",
                     "content": "You are a movie recommendation ranking assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=3000,
            )
            raw = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            usage = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                     "completion_tokens": getattr(usage, "completion_tokens", None)} if usage else None
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object in reply")
            ranked = json.loads(raw[start:end])["ranked_ids"]
            valid = []
            for i in ranked:
                if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n_candidates and i not in valid:
                    valid.append(i)
            if not valid:
                raise ValueError("no valid candidate ids in reply")
            return {"status": "ok", "valid_ids": valid, "raw": raw, "usage": usage,
                    "latency": time.time() - t0, "attempts": attempt}
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(2 * attempt)
    return {"status": "failed", "valid_ids": [], "raw": raw, "usage": usage, "error": last_err,
            "latency": time.time() - t0, "attempts": retries + 1}


def load_baseline_rows(base, ns_tag, users):
    folder = Path(base) if ns_tag == "full" else Path(base) / f"cold_{ns_tag}"
    names = {"als": "ALS", "knn": "Item-KNN", "kg": "KG retriever", "popularity": "Popularity"}
    rows = {}
    for key, label in names.items():
        f = folder / f"per_user_{key}.csv"
        if f.exists():
            df = pd.read_csv(f)
            rows[label] = summarize(df[df["user_idx"].isin(users)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-n", type=int, default=0, help="cold start: keep only the first N ratings")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="history,full")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--users-file", default="data/eval/eval_users.json")
    ap.add_argument("--base", default="results/test", help="folder written by run_systems / run_coldstart")
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--kg", default="data/kg/kg.pkl")
    ap.add_argument("--kg-params", default="tests/results/kg_best_params.json")
    ap.add_argument("--semantic", default="data/kg/plot_embeddings.npz")
    ap.add_argument("--embed-cache", default="data/rag")
    ap.add_argument("--chat-model", default="qwen35-27b")
    ap.add_argument("--embed-model", default="qwen3-embedding-4b")
    args = ap.parse_args()

    tag = f"first{args.history_n}" if args.history_n else "full"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    proc = Path(args.proc)

    key = os.environ.get("ASU_API_KEY") or getpass("Enter your ASU API key: ")
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=300.0, max_retries=1)

    metadata = pd.read_parquet(proc / "metadata.parquet")
    rag = EvalRAG(client, embedding_model=args.embed_model, chat_model=args.chat_model)
    rag.prepare_documents(metadata)
    rag.embeddings = embed_movies(rag, args.workers, Path(args.embed_cache))
    rag.index = faiss.IndexFlatIP(rag.embeddings.shape[1])
    rag.index.add(rag.embeddings)
    if args.embed_only:
        return


    cfg = json.loads(Path(args.users_file).read_text())
    users = cfg["llm"][: args.limit] if args.limit else cfg["llm"]
    train = pd.read_parquet(proc / "train.parquet", columns=["user_idx", "movie_idx", "rating", "timestamp"])
    split = pd.read_parquet(proc / "test.parquet", columns=["user_idx", "movie_idx", "rating"])
    movies = pd.read_parquet(proc / "movies.parquet", columns=["movie_idx", "genres"])

    rel_all = split[split["rating"] >= cfg["relevance_threshold"]].groupby("user_idx")["movie_idx"].apply(set)
    relevant = {int(u): rel_all[u] for u in users if u in rel_all.index}
    users = sorted(relevant)
    seen_full = train[train["user_idx"].isin(users)].groupby("user_idx")["movie_idx"].apply(set).to_dict()
    train_u = train[train["user_idx"].isin(users)]
    if args.history_n:
        train_u = truncate(train_u, users, args.history_n)
    print(f"[{tag}] {len(users)} users")


    kg = KnowledgeGraph.load(args.kg)
    kg_retriever = KGRetriever(kg, **load_params(args.kg_params)).fit(train_u, pop_source=train)
    semantic = SemanticNeighbors.load(args.semantic).fit(train_u)
    rag.train, rag.kg_retriever, rag.semantic_neighbors = train_u.copy(), kg_retriever, semantic


    knn_f = Path(args.base) / ("" if tag == "full" else f"cold_{tag}") / "top100_knn.parquet"
    knn_df = pd.read_parquet(knn_f)
    knn_lists = knn_df.sort_values(["user_idx", "rank"]).groupby("user_idx")["movie_idx"].apply(list).to_dict()


    pool_f, ev_dir = out / "pools.parquet", out / "evidence"
    ev_dir.mkdir(exist_ok=True)
    state = {}

    def prep(u):
        user_query, cands = build_pool(rag, kg_retriever, knn_lists, u, seen_full[u])
        ev = rag._build_evidence(u, cands)
        return u, user_query, cands, ev

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for u, uq, cands, ev in tqdm(ex.map(prep, users), total=len(users), desc="building pools + evidence"):
            state[u] = {"query": uq, "cands": cands, "evidence": ev}
            (ev_dir / f"{u}.json").write_text(json.dumps({"movie_idx": cands["movie_idx"].tolist(), "evidence": ev}))
    pd.concat([s["cands"][["movie_idx", "rrf_score", "candidate_source"]].assign(
        user_idx=u, rrf_rank=range(1, len(s["cands"]) + 1)) for u, s in state.items()]
    ).to_parquet(pool_f, index=False)


    variants = [v.strip() for v in args.variants.split(",")]
    tasks = []
    for v in variants:
        (out / f"cache_{v}").mkdir(exist_ok=True)
        tasks += [(v, u) for u in users]

    def run_task(task):
        v, u = task
        f = out / f"cache_{v}" / f"{u}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if rec["status"] == "ok":
                return rec
        s = state[u]
        evidence = s["evidence"] if v == "full" else None
        prompt = make_prompt(rag, s["query"], s["cands"], evidence)
        res = call_llm(client, args.chat_model, prompt, len(s["cands"]), args.retries)
        rec = {"user_idx": u, "variant": v, "model": args.chat_model, "prompt_chars": len(prompt),
               "pool_movie_idx": s["cands"]["movie_idx"].tolist(), **res}
        f.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(tqdm(ex.map(run_task, tasks), total=len(tasks), desc="LLM calls"))


    genres = build_genres(movies)
    novelty = build_novelty(train, len(genres))
    lists = {"pool_rrf": {u: state[u]["cands"]["movie_idx"].tolist() for u in users}}
    stats = {}
    for v in variants:
        lists[f"llm_{v}"] = {}
        vr = [r for r in recs if r["variant"] == v]
        for r in vr:
            pool = r["pool_movie_idx"]
            order = r["valid_ids"] + [i for i in range(len(pool)) if i not in set(r["valid_ids"])]
            lists[f"llm_{v}"][r["user_idx"]] = [pool[i] for i in order]
        stats[v] = {"ok": sum(r["status"] == "ok" for r in vr), "failed": sum(r["status"] != "ok" for r in vr),
                    "mean_latency_s": float(np.mean([r["latency"] for r in vr])),
                    "mean_prompt_tokens": float(np.mean([(r.get("usage") or {}).get("prompt_tokens") or np.nan
                                                         for r in vr]))}

    rows = load_baseline_rows(args.base, tag, set(users))
    labels = {"pool_rrf": "Pool (RRF order, no LLM)", "llm_history": "Pool + LLM (history)",
              "llm_full": "Pool + LLM (full evidence)"}
    for name, lst in lists.items():
        per_user = score_lists(lst, relevant, genres, novelty)
        per_user.to_csv(out / f"per_user_{name}.csv", index=False)
        pd.DataFrame([(u, r + 1, m) for u, l in lst.items() for r, m in enumerate(l[:POOL_K])],
                     columns=["user_idx", "rank", "movie_idx"]).to_parquet(out / f"top100_{name}.parquet", index=False)
        rows[labels[name]] = summarize(per_user)

    df = pd.DataFrame(rows).T
    df["n_users"] = df["n_users"].astype(int)
    text = df.round(4).to_markdown()
    (out / "summary.md").write_text(text + "\n\nLLM call status: " + json.dumps(stats, indent=1) + "\n")
    print("\n" + text)
    print("\nLLM call status:", json.dumps(stats, indent=1))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
