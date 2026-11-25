import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

SYSTEMS = {"als": "ALS", "knn": "Item-KNN", "kg": "KG retriever", "popularity": "Popularity"}
LLM_SYSTEMS = {"pool_rrf": "Pool (RRF order, no LLM)", "llm_history": "Pool + LLM (history)",
               "llm_full": "Pool + LLM (full evidence)"}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
RNG = np.random.default_rng(42)


def load(folder, names):
    out = {}
    for key in names:
        f = Path(folder) / f"per_user_{key}.csv"
        if f.exists():
            out[key] = pd.read_csv(f).set_index("user_idx")
    return out


def boot_ci(x, n_boot=10000, chunk=500):
    x = np.asarray(x, dtype=float)
    n = len(x)
    means = []
    for _ in range(0, n_boot, chunk):
        idx = RNG.integers(0, n, size=(chunk, n))
        means.append(x[idx].mean(axis=1))
    means = np.concatenate(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(st.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(ps):
    order = np.argsort(ps)
    m, adj, running = len(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b, metric):
    j = a[[metric]].join(b[[metric]], lsuffix="_a", rsuffix="_b", how="inner").dropna()
    return (j[f"{metric}_a"] - j[f"{metric}_b"]).to_numpy(), len(j)


def compare(a, b, metric, n_boot):
    d, n = paired(a, b, metric)
    lo, hi = boot_ci(d, n_boot)
    return {"n": n, "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi, "p": wilcoxon_p(d)}


def md(df, digits=4):
    return df.round(digits).to_markdown()


def sig(p):
    return "yes" if p < 0.05 else "no"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_bars(path, rows, title, n_users, n_boot):
    items = sorted(((k, v.mean(), boot_ci(v.to_numpy(), n_boot)) for k, v in rows.items()), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(items))
    means = np.array([m for _, m, _ in items])
    lo = np.array([c[0] for _, _, c in items])
    hi = np.array([c[1] for _, _, c in items])
    ax.barh(y, means, height=0.5, color=SERIES[0])
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _, _ in items], color=INK, fontsize=9.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("NDCG@10 (mean over users, 95% bootstrap CI)", color=INK2, fontsize=9)
    ax.set_title(f"{title}  (n = {n_users} users)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_cold(path, curve):
    xs = ["first 5\nratings", "first 10\nratings", "full\nhistory"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    style_axes(ax)
    for i, (label, ys) in enumerate(curve.items()):
        ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=7, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (2, ys[2]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_xlim(-0.2, 2.9)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("NDCG@10", color=INK2, fontsize=9)
    ax.set_title("NDCG@10 by length of the user's rating history", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/test")
    ap.add_argument("--llm-dir", default="results/llm/full")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--users-file", default="data/eval/eval_users.json")
    ap.add_argument("--out", default="results/stats")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.users_file).read_text())
    report = []


    base = load(args.base, SYSTEMS)
    llm = load(args.llm_dir, LLM_SYSTEMS)
    llm_users = set(next(iter(llm.values())).index)
    allsys = {**{SYSTEMS[k]: v[v.index.isin(llm_users)] for k, v in base.items()},
              **{LLM_SYSTEMS[k]: v for k, v in llm.items()}}
    users = sorted(set.intersection(*[set(v.index) for v in allsys.values()]))
    allsys = {k: v.loc[users] for k, v in allsys.items()}
    n_llm = len(users)

    metrics = ["ndcg@10", "precision@10", "hr@10", "mrr@10", "recall@10"]
    rows = []
    for name, df in allsys.items():
        row = {"system": name}
        for m in metrics:
            row[m] = df[m].mean()
        lo, hi = boot_ci(df["ndcg@10"].to_numpy(), args.boot)
        row["ndcg@10 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
        row["ild@10"], row["novelty@10"] = df["ild@10"].mean(), df["novelty@10"].mean()
        rows.append(row)
    tabA = pd.DataFrame(rows).set_index("system")
    report += [f"## A. All systems on the same {n_llm} users (the LLM sample)", "", md(tabA), ""]
    tabA.to_csv(out / "A_llm_sample_means.csv")


    als = allsys["ALS"]
    brow = []
    for name, df in allsys.items():
        if name == "ALS":
            continue
        for m in ("ndcg@10", "precision@10"):
            c = compare(df, als, m, args.boot)
            brow.append({"system": name, "metric": m, **c})
    tabB = pd.DataFrame(brow)
    for m in ("ndcg@10", "precision@10"):
        mask = tabB["metric"] == m
        tabB.loc[mask, "p_holm"] = holm(tabB.loc[mask, "p"].to_numpy())
    tabB["significant (Holm<0.05)"] = tabB["p_holm"].map(sig)
    report += [f"## B. Each system minus ALS, paired over {n_llm} users", "",
               "mean_diff < 0 means worse than ALS. p = Wilcoxon signed-rank; p_holm corrects for the number of systems compared.",
               "", md(tabB.set_index(["system", "metric"]), 5), ""]
    tabB.to_csv(out / "B_vs_als_llm_sample.csv", index=False)


    crow = []
    pairs = [("Pool + LLM (full evidence)", "Pool (RRF order, no LLM)"),
             ("Pool + LLM (history)", "Pool (RRF order, no LLM)"),
             ("Pool + LLM (full evidence)", "Pool + LLM (history)"),
             ("Pool (RRF order, no LLM)", "Item-KNN")]
    for a, b in pairs:
        for m in ("ndcg@10", "precision@10", "ild@10", "novelty@10"):
            crow.append({"A": a, "B": b, "metric": m, **compare(allsys[a], allsys[b], m, args.boot)})
    tabC = pd.DataFrame(crow)
    tabC["significant (p<0.05)"] = tabC["p"].map(sig)
    report += ["## C. Paired comparisons that isolate the LLM's contribution (A minus B)", "",
               md(tabC.set_index(["A", "B", "metric"]), 5), ""]
    tabC.to_csv(out / "C_llm_contribution.csv", index=False)


    big = {SYSTEMS[k]: v for k, v in base.items()}
    drow = []
    for name, df in big.items():
        if name == "ALS":
            continue
        for m in ("ndcg@10", "precision@10"):
            drow.append({"system": name, "metric": m, **compare(df, big["ALS"], m, args.boot)})
    tabD = pd.DataFrame(drow)
    tabD["significant (p<0.05)"] = tabD["p"].map(sig)
    report += [f"## D. Non-LLM systems minus ALS on the full test sample ({len(big['ALS']):,} users)", "",
               md(tabD.set_index(["system", "metric"]), 5), ""]
    tabD.to_csv(out / "D_vs_als_full_sample.csv", index=False)


    cold = {n: load(Path(args.base) / f"cold_first{n}", SYSTEMS) for n in (5, 10)}
    heavy = set(cfg["heavy"])
    erows, curve, gains = [], {}, []
    for key, label in SYSTEMS.items():
        if key not in base or any(key not in cold[n] for n in (5, 10)):
            continue
        curve[label] = [cold[5][key]["ndcg@10"].mean(), cold[10][key]["ndcg@10"].mean(), base[key]["ndcg@10"].mean()]
        erows.append({"system": label, "first 5": curve[label][0], "first 10": curve[label][1],
                      "full history": curve[label][2],
                      "heavy users (>50)": base[key].loc[base[key].index.isin(heavy), "ndcg@10"].mean(),
                      "kept at first 5": curve[label][0] / curve[label][2]})
    tabE = pd.DataFrame(erows).set_index("system")
    report += ["## E. Cold start: NDCG@10 by history length", "", md(tabE), ""]
    tabE.to_csv(out / "E_coldstart_ndcg.csv")

    if all(k in cold[5] for k in ("als", "kg")):
        for key in ("knn", "kg", "popularity"):
            for cond, a, b in [("first 5", cold[5][key], cold[5]["als"]), ("first 10", cold[10][key], cold[10]["als"]),
                               ("heavy (>50 ratings)", base[key][base[key].index.isin(heavy)],
                                base["als"][base["als"].index.isin(heavy)])]:
                c = compare(a, b, "ndcg@10", args.boot)
                gains.append({"system": SYSTEMS[key], "condition": cond, **c})
        tabF = pd.DataFrame(gains)
        tabF["significant (p<0.05)"] = tabF["p"].map(sig)
        report += ["## F. Gain over ALS (system minus ALS, NDCG@10) by user type", "",
                   "Positive = better than ALS. Success criterion 3 asks whether the gain is larger for new users than for heavy users.",
                   "", md(tabF.set_index(["system", "condition"]), 5), ""]
        tabF.to_csv(out / "F_coldstart_gain_vs_als.csv", index=False)


    test = pd.read_parquet(Path(args.proc) / "test.parquet", columns=["user_idx", "movie_idx", "rating"])
    rel = test[test["rating"] >= cfg["relevance_threshold"]].groupby("user_idx")["movie_idx"].apply(set).to_dict()
    pools = pd.read_parquet(Path(args.llm_dir) / "pools.parquet")
    grow = []
    for var in ("llm_history", "llm_full"):
        top = pd.read_parquet(Path(args.llm_dir) / f"top100_{var}.parquet")
        rho, overlap, pool_top_pos = [], [], []
        for u, g in top.groupby("user_idx"):
            p = pools[pools["user_idx"] == u].sort_values("rrf_rank")
            llm_rank = dict(zip(g["movie_idx"], g["rank"]))
            pr = p["movie_idx"].tolist()
            if len(pr) < 10:
                continue
            rho.append(st.spearmanr(range(1, len(pr) + 1), [llm_rank.get(m, len(pr)) for m in pr]).statistic)
            overlap.append(len(set(g.sort_values("rank")["movie_idx"].head(10)) & set(pr[:10])) / 10)
            pool_top_pos.append(np.mean([llm_rank.get(m, len(pr)) for m in pr[:10]]))
        grow.append({"system": LLM_SYSTEMS[var], "Spearman(pool rank, LLM rank)": np.mean(rho),
                     "share of LLM top-10 also in pool top-10": np.mean(overlap),
                     "mean LLM rank of the pool's top-10 items": np.mean(pool_top_pos)})
    tabG = pd.DataFrame(grow).set_index("system")
    report += ["## G1. How far did the LLM move things from the pool's order?", "", md(tabG), ""]


    def hits_by_source(lst_df, label):
        m = lst_df[lst_df["rank"] <= 10].merge(pools[["user_idx", "movie_idx", "candidate_source"]],
                                               on=["user_idx", "movie_idx"], how="left")
        m["hit"] = [mv in rel.get(u, ()) for u, mv in zip(m["user_idx"], m["movie_idx"])]
        res = {"list": label}
        for src in ("knn", "kg", "rag"):
            has = m["candidate_source"].fillna("").str.contains(src)
            res[f"{src}: share of top-10"] = has.mean()
            res[f"{src}: hit rate"] = m.loc[has, "hit"].mean() if has.any() else np.nan
        return res

    srows = [hits_by_source(pd.read_parquet(Path(args.llm_dir) / f"top100_{k}.parquet"), LLM_SYSTEMS[k])
             for k in ("pool_rrf", "llm_history", "llm_full")]
    tabG2 = pd.DataFrame(srows).set_index("list")
    report += ["## G2. Which candidate source fills the top 10, and how often is it a hit?", "",
               "A candidate can come from several sources, so shares can add up to more than 1.", "", md(tabG2), ""]
    pd.concat([tabG, tabG2], axis=1).to_csv(out / "G_diagnostics.csv")


    crit = []
    full, histo = allsys["Pool + LLM (full evidence)"], allsys["Pool + LLM (history)"]
    n_ = compare(full, als, "ndcg@10", args.boot)
    p_ = compare(full, als, "precision@10", args.boot)
    ok1 = n_["mean_diff"] > 0 and n_["p"] < 0.05 and p_["mean_diff"] > 0 and p_["p"] < 0.05
    crit.append(["1. Full pipeline beats ALS on NDCG@10 and P@10 (significant)", "PASS" if ok1 else "NOT MET",
                 f"NDCG@10 diff {n_['mean_diff']:+.4f} (p={n_['p']:.3g}); P@10 diff {p_['mean_diff']:+.4f} (p={p_['p']:.3g})"])
    nov = compare(full, als, "novelty@10", args.boot)
    ild = compare(full, als, "ild@10", args.boot)
    gain_div = (nov["mean_diff"] > 0 and nov["p"] < 0.05) or (ild["mean_diff"] > 0 and ild["p"] < 0.05)
    no_loss = n_["mean_diff"] >= 0
    crit.append(["2. Novelty or diversity improves without NDCG@10 falling below ALS",
                 "PASS" if (gain_div and no_loss) else "NOT MET",
                 f"novelty diff {nov['mean_diff']:+.3f} (p={nov['p']:.3g}); ILD diff {ild['mean_diff']:+.3f} "
                 f"(p={ild['p']:.3g}); NDCG@10 below ALS: {not no_loss}"])
    if gains:
        g = pd.DataFrame(gains)
        kg = g[g["system"] == "KG retriever"].set_index("condition")["mean_diff"]
        crit.append(["3. New users gain more over ALS than heavy users (KG retriever; full pipeline not run on cold start)",
                     "PASS (KG retriever only)" if kg["first 5"] > kg["heavy (>50 ratings)"] else "NOT MET",
                     f"KG minus ALS: first 5 {kg['first 5']:+.4f}, first 10 {kg['first 10']:+.4f}, heavy {kg['heavy (>50 ratings)']:+.4f}"])
    ff = Path(args.llm_dir) / "faithfulness" / "faithfulness.csv"
    if ff.exists():
        f = pd.read_csv(ff)
        f = f[f["has_explanation"]]
        v = f["faithful_structured"].mean()
        crit.append(["4. At least 90% of explanations are faithful", "PASS" if v >= 0.9 else "NOT MET",
                     f"{v:.1%} of {len(f)} explanations (structured check); text check {f['faithful_text'].mean():.1%}"])
    else:
        crit.append(["4. At least 90% of explanations are faithful", "NOT RUN", "run src.eval.faithfulness first"])
    tabH = pd.DataFrame(crit, columns=["success criterion", "verdict", "evidence"])
    report += ["## H. Success criteria from the proposal", "", tabH.to_markdown(index=False), ""]
    tabH.to_csv(out / "H_success_criteria.csv", index=False)


    fig_bars(out / "fig_ndcg_llm_sample.png", {k: v["ndcg@10"] for k, v in allsys.items()},
             "NDCG@10, all systems on the LLM sample", n_llm, args.boot)
    if curve:
        fig_cold(out / "fig_coldstart.png", curve)

    (out / "stats_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nSaved report, tables and charts to {out}/")


if __name__ == "__main__":
    main()
