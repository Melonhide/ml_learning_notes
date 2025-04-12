Logistic Regression assumes a linear relationship between the input features and the log-odds of the positive class. It is simple, fast, and interpretable, but struggles when there are complex non-linear patterns or interactions among features. In our case, the borrower risk features had non-linear thresholds (e.g., sudden risk increase when LTV > 0.8), which made tree-based models like XGBoost more suitable.

Pros:

- Interpretable

- Fast training

- Works well with linear separability

Cons (for our case):

- Assumes linear relationships between features and outcome

- Poor at modeling interactions and non-linear patterns

- Requires careful preprocessing (e.g., missing value imputation, scaling)

