# KG Retriever Tuning Results

Validation split, 5,000 users (seed 42). Coordinate search; selection metric = Recall@100, full history.

## Best configuration

```json
{
  "pop_alpha": 0.75,
  "smoothing": 0.1,
  "normalize_candidates": false,
  "backfill": false,
  "rel_weights": {
    "DIRECTED_BY": 1.0,
    "STARS": 1.2,
    "HAS_KEYWORD": 0.5,
    "HAS_GENRE": 0.3
  }
}
```

| History | Recall@100 | Precision@100 | NDCG@10 |
|---|---|---|---|
| Starting settings (full) | 0.2795 | 0.0174 | 0.0658 |
| Full | 0.2978 | 0.0177 | 0.0661 |
| First 10 ratings | 0.2366 | 0.0118 | 0.0310 |
| First 5 ratings | 0.2323 | 0.0116 | 0.0300 |

## All trials

```
               stage  pop_alpha  smoothing  normalize_candidates  backfill  DIRECTED_BY  STARS  HAS_KEYWORD  HAS_GENRE  recall@100  precision@100  ndcg@10  objective
            backfill       0.75       0.10                 False      True          1.0    1.2         0.50       0.30    0.297816       0.017654 0.066091   0.297816
           HAS_GENRE       0.75       0.10                 False     False          1.0    1.2         0.50       0.30    0.297816       0.017654 0.066091   0.297816
               STARS       0.75       0.10                 False     False          1.0    1.2         0.50       0.15    0.296370       0.017752 0.065855   0.296370
normalize_candidates       0.75       0.10                 False     False          1.0    0.6         0.50       0.15    0.296190       0.017656 0.066605   0.296190
           smoothing       0.75       0.10                  True     False          1.0    0.6         0.50       0.15    0.296092       0.017688 0.067010   0.296092
           smoothing       0.75       0.05                  True     False          1.0    0.6         0.50       0.15    0.296033       0.017792 0.066110   0.296033
         HAS_KEYWORD       0.75       0.10                 False     False          1.0    1.2         0.25       0.15    0.295335       0.017736 0.065855   0.295335
         DIRECTED_BY       0.75       0.10                 False     False          0.5    0.6         0.50       0.15    0.295262       0.017550 0.065885   0.295262
           HAS_GENRE       0.75       0.10                 False     False          1.0    1.2         0.50       0.60    0.295242       0.017306 0.066424   0.295242
         DIRECTED_BY       0.75       0.10                 False     False          2.0    0.6         0.50       0.15    0.294983       0.017670 0.065942   0.294983
           smoothing       0.75       0.20                  True     False          1.0    0.6         0.50       0.15    0.294188       0.017554 0.067722   0.294188
           HAS_GENRE       0.75       0.10                 False     False          1.0    1.2         0.50       0.05    0.293920       0.017664 0.065176   0.293920
               STARS       0.75       0.10                 False     False          1.0    0.3         0.50       0.15    0.293760       0.017500 0.065567   0.293760
         HAS_KEYWORD       0.75       0.10                 False     False          1.0    1.2         1.00       0.15    0.293519       0.017522 0.065662   0.293519
           smoothing       0.75       0.02                  True     False          1.0    0.6         0.50       0.15    0.291291       0.017716 0.066106   0.291291
           smoothing       0.75       0.50                  True     False          1.0    0.6         0.50       0.15    0.284389       0.016968 0.066097   0.284389
           smoothing       0.75       0.00                  True     False          1.0    0.6         0.50       0.15    0.279455       0.017424 0.065773   0.279455
```
