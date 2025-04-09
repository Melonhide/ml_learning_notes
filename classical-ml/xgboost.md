# XGBoost (Extreme Gradient Boosting)

XGBoost is a powerful and scalable implementation of gradient boosting algorithms, widely used in machine learning competitions and industry applications—especially effective on tabular data.

---

## 📌 Why XGBoost?

### ✅ Handles Missing Values Automatically

One of XGBoost's powerful features is its ability to handle missing values **natively without any imputation**.

Instead of filling missing values with statistics (mean, median, etc.), XGBoost learns how to deal with them during tree construction:

- During each split, if some data points have missing values for the splitting feature,
- XGBoost tries **both left and right splits** for these missing samples,
- It computes the loss for both cases and chooses the direction (left or right) that minimizes the loss,
- This direction is stored as the **default direction** for missing values at that split node.

💡 When the model sees a missing value at prediction time, it simply follows the learned default path.

This logic is applied **at every split and at every tree**, so missing value handling is learned dynamically and recursively as the model grows.


### ✅ Built-in Regularization

XGBoost includes built-in regularization to control model complexity and prevent overfitting — a key improvement over traditional Gradient Boosting methods.

#### 🧮 Objective Function with Regularization

Each boosting iteration optimizes the following objective:

$$
\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

Where:

- $L(y_i, \hat{y}_i)$ is the loss function (e.g., log loss)
- $\Omega(f_k)$ is the regularization term for tree $k$

#### 📐 Regularization Term Definition

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
$$

- $T$: number of leaves in the tree  
- $w_j$: score/output of leaf $j$  
- $\gamma$: penalty on the number of leaves (structural regularization)  
- $\lambda$: L2 regularization term (like Ridge regression)

#### 🎯 Why it Matters

| Goal | How XGBoost Achieves It |
|------|--------------------------|
| Avoid overly complex trees | Penalize large $T$ via $\gamma$ |
| Prevent large leaf weights | Add L2 penalty via $\lambda$ |
| Encourage simplicity and robustness | Regularization terms are used during each split evaluation |

#### 🔧 Related Hyperparameters

- `gamma`: controls the minimum loss reduction required to make a split
- `lambda`: L2 regularization term on weights
- `alpha`: L1 regularization term (optional, like Lasso)

#### ✅ Summary

> XGBoost incorporates both structural and weight regularization into its objective function. This prevents overfitting by penalizing trees that are too complex or have extreme predictions, making the model more robust and generalizable.

- Built-in parallelization during training
- Strong performance on structured/tabular data
- Widely supported and production-ready

---

## ⚙️ How It Works

XGBoost builds an additive ensemble of decision trees, where each new tree is trained to correct the mistakes (residuals) of the previous trees. Unlike traditional gradient boosting which fits residuals directly, XGBoost uses both the **first- and second-order derivatives (gradient and hessian)** of the loss function to guide the optimization.

