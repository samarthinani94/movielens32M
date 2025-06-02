# movielens32M

A movie recommender system based on MovieLens data.

---

## `ratings.csv`

### Sparsity
- **Sparsity**: `0.001886`

### Rating Distribution
| Rating | Count     |
|--------|-----------|
| 4.0    | 8,367,654 |
| 3.0    | 6,054,990 |
| 5.0    | 4,596,577 |
| 3.5    | 4,290,105 |
| 4.5    | 2,974,000 |
| 2.0    | 2,028,622 |
| 2.5    | 1,685,386 |
| 1.0    | 946,675   |
| 1.5    | 531,063   |
| 0.5    | 525,132   |

### Observations
- **48%** of items have less than 5 ratings.
- **57%** of items have less than 5 ratings if considering data from **2019-01** onwards.
- Each user has at least **20 ratings**.

### Processing Plan
- Apply **5-core processing**.
- Use data from **2019 onwards** to account for changes in media consumption patterns due to **COVID-19**. The focus is on modeling the most recent patterns.

---

## `tags.csv`

### Features
- Use the **top 5 tags** and include their percentages as features.
- Generate **10 features** based on tags.

### Observations
- **17 tags** contain missing values (`isna`).

---

## `movies.csv`

### Metadata
- Includes **movie release year**, which can be used as metadata.


## matrix factorization + MSE - ndcg = 0.052
