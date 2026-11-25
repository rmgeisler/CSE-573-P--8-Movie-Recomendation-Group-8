## A. All systems on the same 100 users (the LLM sample)

| system                     |   ndcg@10 |   precision@10 |   hr@10 |   mrr@10 |   recall@10 | ndcg@10 95% CI   |   ild@10 |   novelty@10 |
|:---------------------------|----------:|---------------:|--------:|---------:|------------:|:-----------------|---------:|-------------:|
| ALS                        |    0.0943 |          0.042 |    0.32 |   0.1634 |      0.1182 | [0.0631, 0.1274] |   0.7737 |       2.9032 |
| Item-KNN                   |    0.0757 |          0.037 |    0.31 |   0.1074 |      0.1097 | [0.0475, 0.1086] |   0.7786 |       2.6729 |
| KG retriever               |    0.062  |          0.031 |    0.27 |   0.1067 |      0.0776 | [0.0378, 0.0890] |   0.7747 |       1.8389 |
| Popularity                 |    0.0559 |          0.03  |    0.24 |   0.1083 |      0.0646 | [0.0327, 0.0822] |   0.8054 |       1.6148 |
| Pool (RRF order, no LLM)   |    0.0645 |          0.029 |    0.26 |   0.1128 |      0.0694 | [0.0358, 0.0989] |   0.7495 |       2.2972 |
| Pool + LLM (history)       |    0.0362 |          0.022 |    0.19 |   0.0635 |      0.0437 | [0.0203, 0.0542] |   0.6117 |       3.4772 |
| Pool + LLM (full evidence) |    0.0343 |          0.021 |    0.16 |   0.053  |      0.0488 | [0.0175, 0.0541] |   0.6087 |       3.6446 |

## B. Each system minus ALS, paired over 100 users

mean_diff < 0 means worse than ALS. p = Wilcoxon signed-rank; p_holm corrects for the number of systems compared.

|                                                |   n |   mean_diff |   ci_low |   ci_high |       p |   p_holm | significant (Holm<0.05)   |
|:-----------------------------------------------|----:|------------:|---------:|----------:|--------:|---------:|:--------------------------|
| ('Item-KNN', 'ndcg@10')                        | 100 |    -0.01858 | -0.04975 |   0.01176 | 0.2228  |  0.2228  | no                        |
| ('Item-KNN', 'precision@10')                   | 100 |    -0.005   | -0.017   |   0.007   | 0.5104  |  0.5104  | no                        |
| ('KG retriever', 'ndcg@10')                    | 100 |    -0.0323  | -0.06318 |  -0.0027  | 0.06138 |  0.15496 | no                        |
| ('KG retriever', 'precision@10')               | 100 |    -0.011   | -0.025   |   0.002   | 0.11026 |  0.32878 | no                        |
| ('Popularity', 'ndcg@10')                      | 100 |    -0.03837 | -0.07535 |  -0.00258 | 0.03998 |  0.15496 | no                        |
| ('Popularity', 'precision@10')                 | 100 |    -0.012   | -0.027   |   0.003   | 0.10959 |  0.32878 | no                        |
| ('Pool (RRF order, no LLM)', 'ndcg@10')        | 100 |    -0.02978 | -0.06851 |   0.01307 | 0.03874 |  0.15496 | no                        |
| ('Pool (RRF order, no LLM)', 'precision@10')   | 100 |    -0.013   | -0.027   |   0.001   | 0.06893 |  0.27571 | no                        |
| ('Pool + LLM (history)', 'ndcg@10')            | 100 |    -0.05808 | -0.09086 |  -0.02699 | 0.00074 |  0.0037  | yes                       |
| ('Pool + LLM (history)', 'precision@10')       | 100 |    -0.02    | -0.034   |  -0.006   | 0.00682 |  0.03989 | yes                       |
| ('Pool + LLM (full evidence)', 'ndcg@10')      | 100 |    -0.05994 | -0.09084 |  -0.02953 | 0.00035 |  0.00212 | yes                       |
| ('Pool + LLM (full evidence)', 'precision@10') | 100 |    -0.021   | -0.035   |  -0.007   | 0.00665 |  0.03989 | yes                       |

## C. Paired comparisons that isolate the LLM's contribution (A minus B)

|                                                                            |   n |   mean_diff |   ci_low |   ci_high |       p | significant (p<0.05)   |
|:---------------------------------------------------------------------------|----:|------------:|---------:|----------:|--------:|:-----------------------|
| ('Pool + LLM (full evidence)', 'Pool (RRF order, no LLM)', 'ndcg@10')      | 100 |    -0.03016 | -0.06705 |   0.00245 | 0.19848 | no                     |
| ('Pool + LLM (full evidence)', 'Pool (RRF order, no LLM)', 'precision@10') | 100 |    -0.008   | -0.021   |   0.004   | 0.22689 | no                     |
| ('Pool + LLM (full evidence)', 'Pool (RRF order, no LLM)', 'ild@10')       | 100 |    -0.14082 | -0.16898 |  -0.11198 | 0       | yes                    |
| ('Pool + LLM (full evidence)', 'Pool (RRF order, no LLM)', 'novelty@10')   | 100 |     1.34744 |  1.15291 |   1.56023 | 0       | yes                    |
| ('Pool + LLM (history)', 'Pool (RRF order, no LLM)', 'ndcg@10')            | 100 |    -0.02831 | -0.06299 |   0.00214 | 0.13528 | no                     |
| ('Pool + LLM (history)', 'Pool (RRF order, no LLM)', 'precision@10')       | 100 |    -0.007   | -0.019   |   0.005   | 0.25153 | no                     |
| ('Pool + LLM (history)', 'Pool (RRF order, no LLM)', 'ild@10')             | 100 |    -0.1378  | -0.16766 |  -0.10825 | 0       | yes                    |
| ('Pool + LLM (history)', 'Pool (RRF order, no LLM)', 'novelty@10')         | 100 |     1.18006 |  0.99152 |   1.37616 | 0       | yes                    |
| ('Pool + LLM (full evidence)', 'Pool + LLM (history)', 'ndcg@10')          | 100 |    -0.00185 | -0.01683 |   0.01327 | 0.66395 | no                     |
| ('Pool + LLM (full evidence)', 'Pool + LLM (history)', 'precision@10')     | 100 |    -0.001   | -0.009   |   0.007   | 0.80837 | no                     |
| ('Pool + LLM (full evidence)', 'Pool + LLM (history)', 'ild@10')           | 100 |    -0.00302 | -0.02059 |   0.01472 | 0.63869 | no                     |
| ('Pool + LLM (full evidence)', 'Pool + LLM (history)', 'novelty@10')       | 100 |     0.16739 |  0.02622 |   0.31663 | 0.04072 | yes                    |
| ('Pool (RRF order, no LLM)', 'Item-KNN', 'ndcg@10')                        | 100 |    -0.0112  | -0.04602 |   0.02299 | 0.36767 | no                     |
| ('Pool (RRF order, no LLM)', 'Item-KNN', 'precision@10')                   | 100 |    -0.008   | -0.02    |   0.003   | 0.20482 | no                     |
| ('Pool (RRF order, no LLM)', 'Item-KNN', 'ild@10')                         | 100 |    -0.02916 | -0.04571 |  -0.0119  | 3e-05   | yes                    |
| ('Pool (RRF order, no LLM)', 'Item-KNN', 'novelty@10')                     | 100 |    -0.37577 | -0.49191 |  -0.26715 | 0       | yes                    |

## D. Non-LLM systems minus ALS on the full test sample (5,000 users)

|                                  |    n |   mean_diff |   ci_low |   ci_high |   p | significant (p<0.05)   |
|:---------------------------------|-----:|------------:|---------:|----------:|----:|:-----------------------|
| ('Item-KNN', 'ndcg@10')          | 5000 |    -0.01665 | -0.02061 |  -0.01282 |   0 | yes                    |
| ('Item-KNN', 'precision@10')     | 5000 |    -0.00992 | -0.01184 |  -0.00802 |   0 | yes                    |
| ('KG retriever', 'ndcg@10')      | 5000 |    -0.02398 | -0.0282  |  -0.01976 |   0 | yes                    |
| ('KG retriever', 'precision@10') | 5000 |    -0.0129  | -0.01522 |  -0.01058 |   0 | yes                    |
| ('Popularity', 'ndcg@10')        | 5000 |    -0.02761 | -0.03238 |  -0.02286 |   0 | yes                    |
| ('Popularity', 'precision@10')   | 5000 |    -0.01658 | -0.01898 |  -0.01416 |   0 | yes                    |

## E. Cold start: NDCG@10 by history length

| system       |   first 5 |   first 10 |   full history |   heavy users (>50) |   kept at first 5 |
|:-------------|----------:|-----------:|---------------:|--------------------:|------------------:|
| ALS          |    0.0296 |     0.0345 |         0.0785 |              0.0765 |            0.3771 |
| Item-KNN     |    0.0334 |     0.0343 |         0.0618 |              0.0615 |            0.54   |
| KG retriever |    0.0443 |     0.0449 |         0.0545 |              0.0583 |            0.8125 |
| Popularity   |    0.0509 |     0.0509 |         0.0509 |              0.0492 |            0.9997 |

## F. Gain over ALS (system minus ALS, NDCG@10) by user type

Positive = better than ALS. Success criterion 3 asks whether the gain is larger for new users than for heavy users.

|                                         |    n |   mean_diff |   ci_low |   ci_high |       p | significant (p<0.05)   |
|:----------------------------------------|-----:|------------:|---------:|----------:|--------:|:-----------------------|
| ('Item-KNN', 'first 5')                 | 5000 |     0.0038  |  0.00098 |   0.00663 | 0.02421 | yes                    |
| ('Item-KNN', 'first 10')                | 5000 |    -0.00013 | -0.003   |   0.00285 | 0.28457 | no                     |
| ('Item-KNN', 'heavy (>50 ratings)')     | 2834 |    -0.01508 | -0.01939 |  -0.01088 | 0       | yes                    |
| ('KG retriever', 'first 5')             | 5000 |     0.01469 |  0.01129 |   0.01813 | 0       | yes                    |
| ('KG retriever', 'first 10')            | 5000 |     0.01038 |  0.00683 |   0.01392 | 0       | yes                    |
| ('KG retriever', 'heavy (>50 ratings)') | 2834 |    -0.01821 | -0.02312 |  -0.0132  | 0       | yes                    |
| ('Popularity', 'first 5')               | 5000 |     0.02126 |  0.0174  |   0.02528 | 0       | yes                    |
| ('Popularity', 'first 10')              | 5000 |     0.0164  |  0.01239 |   0.02056 | 0       | yes                    |
| ('Popularity', 'heavy (>50 ratings)')   | 2834 |    -0.02732 | -0.03253 |  -0.02211 | 0       | yes                    |

## G1. How far did the LLM move things from the pool's order?

| system                     |   Spearman(pool rank, LLM rank) |   share of LLM top-10 also in pool top-10 |   mean LLM rank of the pool's top-10 items |
|:---------------------------|--------------------------------:|------------------------------------------:|-------------------------------------------:|
| Pool + LLM (history)       |                          0.2364 |                                     0.236 |                                     45.105 |
| Pool + LLM (full evidence) |                          0.3322 |                                     0.262 |                                     37.064 |

## G2. Which candidate source fills the top 10, and how often is it a hit?

A candidate can come from several sources, so shares can add up to more than 1.

| list                       |   knn: share of top-10 |   knn: hit rate |   kg: share of top-10 |   kg: hit rate |   rag: share of top-10 |   rag: hit rate |
|:---------------------------|-----------------------:|----------------:|----------------------:|---------------:|-----------------------:|----------------:|
| Pool (RRF order, no LLM)   |                  0.92  |          0.0315 |                 0.964 |         0.0301 |                  0.708 |          0.0282 |
| Pool + LLM (history)       |                  0.608 |          0.028  |                 0.723 |         0.0249 |                  0.623 |          0.0209 |
| Pool + LLM (full evidence) |                  0.584 |          0.0257 |                 0.706 |         0.0241 |                  0.67  |          0.0179 |

## H. Success criteria from the proposal

| success criterion                                                                                    | verdict                  | evidence                                                                                |
|:-----------------------------------------------------------------------------------------------------|:-------------------------|:----------------------------------------------------------------------------------------|
| 1. Full pipeline beats ALS on NDCG@10 and P@10 (significant)                                         | NOT MET                  | NDCG@10 diff -0.0599 (p=0.000353); P@10 diff -0.0210 (p=0.00665)                        |
| 2. Novelty or diversity improves without NDCG@10 falling below ALS                                   | NOT MET                  | novelty diff +0.741 (p=5.92e-10); ILD diff -0.165 (p=9.19e-15); NDCG@10 below ALS: True |
| 3. New users gain more over ALS than heavy users (KG retriever; full pipeline not run on cold start) | PASS (KG retriever only) | KG minus ALS: first 5 +0.0147, first 10 +0.0104, heavy -0.0182                          |
| 4. At least 90% of explanations are faithful                                                         | PASS                     | 99.6% of 500 explanations (structured check); text check 99.6%                          |
