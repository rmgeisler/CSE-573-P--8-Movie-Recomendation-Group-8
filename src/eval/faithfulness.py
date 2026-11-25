import argparse
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

BASE_URL = "https://openai.rc.asu.edu/v1"
TOP_K = 10


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\(\d{4}\)", " ", s)
    s = re.sub(r"^(.*?),\s*(the|a|an)\s*$", r"\2 \1", s.strip())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_phrase(haystack, needle):
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def structured_entities(ev):
    out = set()
    md = ev.get("candidate_metadata", {})
    for col in ("title", "director", "directors", "cast", "genres", "keywords"):
        for part in re.split(r"[|,]", str(md.get(col, ""))):
            if part.strip():
                out.add(norm(part))
    for p in ev.get("kg_paths", []):
        out.add(norm(p.get("entity_name", "")))
        for m in p.get("seed_movies", []):
            out.add(norm(m.get("title", "")))
    for n in ev.get("semantic_neighbors", []):
        out.add(norm(n.get("title", "")))
    out.discard("")
    return out


def check_entity(entity, structured, text_norm):
    e = norm(entity)
    if len(e) < 3:
        return False, False
    in_struct = e in structured or any(contains_phrase(s, e) for s in structured)
    in_text = in_struct or contains_phrase(text_norm, e)
    return in_struct, in_text


def make_prompt(evidence_dicts):
    blocks = []
    for k, ev in enumerate(evidence_dicts, start=1):
        ev = {key: v for key, v in ev.items() if key not in ("candidate_id", "movie_idx")}
        blocks.append(f"Recommendation {k}:\n{json.dumps(ev, default=str)}")
    return f"""
You are explaining movie recommendations to a user.

Below are {len(evidence_dicts)} recommended movies. For each one you get its retrieved evidence: its metadata,
knowledge-graph paths that connect it to movies the user liked, and plot-similar movies the user liked.

Write ONE short explanation (at most 35 words) for each recommendation saying why it was recommended.

Rules:
- Use ONLY facts that appear in that recommendation's own evidence. Do not use outside knowledge.
- Put EVERY director, actor, genre, keyword and movie title you mention in [[double brackets]], for example:
  "Recommended because it shares the genre [[Comedy]] with [[Get Shorty]] and stars [[John Travolta]]."
- Do not mention any person, genre, keyword or movie that is not in that recommendation's evidence.

Return ONLY valid JSON in this format:
{{"explanations": [{{"rank": 1, "explanation": "..."}}, {{"rank": 2, "explanation": "..."}}]}}
Include exactly {len(evidence_dicts)} explanations, ranks 1 to {len(evidence_dicts)}.

{chr(10).join(blocks)}
"""


def call_llm(client, model, prompt, n, retries):
    last_err, raw = None, None
    t0 = time.time()
    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(
                model=model, reasoning_effort="none",
                messages=[{"role": "system", "content": "You write grounded explanations. Return valid JSON only."},
                          {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=3000)
            raw = (resp.choices[0].message.content or "").strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object in reply")
            items = json.loads(raw[start:end])["explanations"]
            by_rank = {}
            for it in items:
                if isinstance(it.get("rank"), int) and 1 <= it["rank"] <= n and isinstance(it.get("explanation"), str):
                    by_rank.setdefault(it["rank"], it["explanation"])
            if not by_rank:
                raise ValueError("no usable explanations in reply")
            return {"status": "ok", "explanations": by_rank, "raw": raw, "latency": time.time() - t0, "attempts": attempt}
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(2 * attempt)
    return {"status": "failed", "explanations": {}, "raw": raw, "error": last_err,
            "latency": time.time() - t0, "attempts": retries + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-dir", default="results/llm/full")
    ap.add_argument("--system", default="llm_full", help="which ranked list to explain (llm_full)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--chat-model", default="qwen35-27b")
    args = ap.parse_args()

    d = Path(args.llm_dir)
    out = d / "faithfulness"
    (out / "cache").mkdir(parents=True, exist_ok=True)

    top = pd.read_parquet(d / f"top100_{args.system}.parquet")
    ok_users = []
    for f in sorted((d / "cache_full").glob("*.json")):
        if json.loads(f.read_text())["status"] == "ok":
            ok_users.append(int(f.stem))
    users = [u for u in sorted(top["user_idx"].unique()) if u in set(ok_users)]
    if args.limit:
        users = users[: args.limit]
    print(f"explaining the top {TOP_K} for {len(users)} users")

    key = os.environ.get("ASU_API_KEY") or getpass("Enter your ASU API key: ")
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=300.0, max_retries=1)

    def load_evidence(u):
        pack = json.loads((d / "evidence" / f"{u}.json").read_text())
        pos = {m: i for i, m in enumerate(pack["movie_idx"])}
        ranked = top[top["user_idx"] == u].sort_values("rank")["movie_idx"].tolist()[:TOP_K]
        return ranked, [json.loads(pack["evidence"][pos[m]]) for m in ranked]

    def run_user(u):
        f = out / "cache" / f"{u}.json"
        ranked, evs = load_evidence(u)
        if f.exists():
            rec = json.loads(f.read_text())
            if rec["status"] == "ok":
                return u, ranked, evs, rec
        rec = call_llm(client, args.chat_model, make_prompt(evs), len(evs), args.retries)
        f.write_text(json.dumps(rec))
        return u, ranked, evs, rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(run_user, users), total=len(users), desc="explanation calls"))

    rows, n_failed_calls = [], 0
    for u, ranked, evs, rec in results:
        if rec["status"] != "ok":
            n_failed_calls += 1
        for k, (m, ev) in enumerate(zip(ranked, evs), start=1):
            text = rec["explanations"].get(str(k)) or rec["explanations"].get(k)
            if text is None:
                rows.append({"user_idx": u, "rank": k, "movie_idx": m, "title": ev["candidate_metadata"].get("title"),
                             "explanation": None, "n_entities": 0, "missing_structured": "", "missing_text": "",
                             "faithful_structured": np.nan, "faithful_text": np.nan, "has_explanation": False})
                continue
            entities = re.findall(r"\[\[(.+?)\]\]", text)
            structured = structured_entities(ev)
            text_norm = norm(json.dumps(ev, default=str))
            miss_s, miss_t = [], []
            for e in entities:
                in_s, in_t = check_entity(e, structured, text_norm)
                if not in_s:
                    miss_s.append(e)
                if not in_t:
                    miss_t.append(e)
            rows.append({"user_idx": u, "rank": k, "movie_idx": m, "title": ev["candidate_metadata"].get("title"),
                         "explanation": text, "n_entities": len(entities),
                         "missing_structured": "; ".join(miss_s), "missing_text": "; ".join(miss_t),
                         "faithful_structured": float(len(entities) > 0 and not miss_s),
                         "faithful_text": float(len(entities) > 0 and not miss_t),
                         "has_explanation": True})
    df = pd.DataFrame(rows)
    df.to_csv(out / "faithfulness.csv", index=False)

    ex_ = df[df["has_explanation"]]
    per_user = ex_.groupby("user_idx")["faithful_structured"].mean()
    summary = {
        "users": len(users),
        "explanation_calls_failed": n_failed_calls,
        "explanations_scored": int(len(ex_)),
        "explanations_missing": int((~df["has_explanation"]).sum()),
        "no_citation_at_all": int((ex_["n_entities"] == 0).sum()),
        "faithfulness_structured": float(ex_["faithful_structured"].mean()),
        "faithfulness_text": float(ex_["faithful_text"].mean()),
        "per_user_mean_structured": float(per_user.mean()),
        "users_with_all_10_faithful": int((per_user == 1.0).sum()),
        "target": 0.90,
    }
    bad = (ex_["missing_structured"].str.split("; ").explode().replace("", np.nan).dropna().value_counts().head(15))
    text = ["| measure | value |", "|---|---|"] + [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
                                                  for k, v in summary.items()]
    text += ["", "Most common cited-but-not-in-evidence entities:", ""] + [f"- {e}: {c}" for e, c in bad.items()]
    (out / "summary.md").write_text("\n".join(text) + "\n")
    print("\n" + "\n".join(text))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
