# XGBoost (Extreme Gradient Boosting)

XGBoost is a powerful and scalable implementation of gradient boosting algorithms, widely used in machine learning competitions and industry applications—especially effective on tabular data.

---

## 📌 Why XGBoost?

- Handles missing data natively
- Regularization to avoid overfitting (L1 & L2)
- Built-in parallelization during training
- Strong performance on structured/tabular data
- Widely supported and production-ready

---

## ⚙️ How It Works

XGBoost builds an additive ensemble of decision trees, where each new tree is trained to correct the mistakes (residuals) of the previous trees. Unlike traditional gradient boosting which fits residuals directly, XGBoost uses both the **first- and second-order derivatives (gradient and hessian)** of the loss function to guide the optimization.

---

### 🧩 Step 1: Initialization

We begin with a base prediction for all training samples. For binary classification, this is usually set to:

$$
\hat{y}^{(0)} = 0.5
$$

---

### 🌳 Step 2: Train the First Tree

We use the binary log loss function:

$$
L(y_i, \hat{y}_i) = -[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

We compute the gradient and hessian for each sample:

$$
g_i = \frac{\partial L}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

We train the first decision tree to fit the negative gradient $-g_i$ using $(g_i, h_i)$ pairs for split finding.

---

### 🔁 Step 3: Update Predictions

Once the first tree $f_1(x)$ is trained, update the prediction:

$$
\hat{y}^{(1)} = \hat{y}^{(0)} + \eta f_1(x)
$$

where $\eta$ is the learning rate.

---

### 🔁 Step 4+: Train More Trees

At each iteration $t$:

1. Compute $g_i^{(t)}, h_i^{(t)}$ based on $\hat{y}^{(t-1)}$
2. Train a new tree $f_t(x)$ to minimize:

$$
Obj_t \approx \sum_i \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)
$$

3. Update the prediction:

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)
$$

---

### 🧮 Objective Function

The total objective function being optimized is:

$$
Obj = \sum_i L(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
$$

The regularization term:

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2
$$

- $T$: number of leaves
- $w_j$: score/output of leaf $j$
- $\gamma$: penalty on number of leaves
- $\lambda$: L2 regularization weight

---

### 📌 Final Prediction

Final output of the model:

$$
\hat{y} = \sum_{t=1}^T \eta f_t(x)
$$

For classification, $\hat{y}$ is typically passed through a sigmoid function.

---

*Note: This explanation assumes binary classification for clarity. The same principles apply to regression and multi-class problems with adjusted loss functions.*