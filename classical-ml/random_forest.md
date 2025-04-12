
# 🌲 Random Forest Explained

Random Forest is a popular ensemble learning method used for classification and regression tasks. It builds multiple decision trees and combines their results for improved performance and robustness.

---

## ✅ What Is a Random Forest?

A Random Forest is an **ensemble of decision trees**, where:

- Each tree is trained on a **bootstrap sample** of the data (sampling with replacement)
- At each split in the tree, only a **random subset of features** is considered
- Predictions are made by **majority voting** (classification) or **averaging** (regression)

This process reduces variance and avoids overfitting compared to individual decision trees.

---

## 🧠 Why Is It “Random”?

- **Random Data**: Each tree sees a different random sample (bagging = bootstrap aggregating)
- **Random Features**: Each split considers a random subset of features

This randomness decorrelates the trees and increases generalization.

---

## 📈 Key Benefits

| Feature | Description |
|--------|-------------|
| Non-linear modeling | Can capture complex feature relationships |
| Robust to outliers | Each tree is only weakly sensitive to extreme values |
| Handles missing values | By surrogate splits or internal strategies |
| Feature importance | Provides ranked list of useful features |
| Low variance overall | Averaging reduces overfitting risk |

---

## ⚠️ Limitations

- Slower to train with large datasets (many trees)
- Can be hard to interpret
- May overfit if trees are too deep and too many features are used
- Not as regularized or sequentially optimized as boosting (e.g., XGBoost)

---

## 🛠️ Key Hyperparameters

- `n_estimators`: Number of trees
- `max_depth`: Maximum depth of each tree
- `max_features`: Number of features to consider at each split
- `min_samples_split`: Minimum number of samples to split a node
- `bootstrap`: Whether to use bootstrapping

---

## 🧪 Example Use Case (Credit Risk)

> We used Random Forest to build a baseline model for credit risk scoring.  
> It handled noisy variables well and provided useful feature rankings.  
> However, it was later replaced by XGBoost for better precision and recall under imbalanced conditions.

---

## ✅ Summary

Random Forest is a powerful, non-linear model that reduces overfitting by averaging across many decision trees trained on random samples. It is robust and often used as a baseline or benchmark model in real-world tabular ML tasks.



Pros:

- Non-linear modeling

- Handles categorical and missing data better than logistic regression

- Less sensitive to outliers

Cons:

- Less precise than boosting (higher bias)

- Trees are trained independently (bagging), so learning is not sequential

- No built-in regularization (can overfit with too many trees)

---

## 🌲 Random Forest Hyperparameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `n_estimators` | Number of trees in the forest | More trees → better performance, but slower |
| `max_depth` | Max depth of each tree | Prevents overfitting if limited |
| `min_samples_split` | Minimum samples to split a node | Higher → more conservative |
| `min_samples_leaf` | Minimum samples in a leaf node | Prevents small noisy leaves |
| `max_features` | Number of features to consider when splitting | Lower = more randomness |
| `bootstrap` | Whether to use bootstrapped samples | If False, uses entire dataset |
| `criterion` | Split metric (`gini`, `entropy`) | Depends on task, often default is fine |
