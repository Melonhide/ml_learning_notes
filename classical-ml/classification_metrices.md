
# ✅ Classification Metrics Explained

In any classification task—especially risk prediction—understanding evaluation metrics like **false positives**, **false negatives**, **recall**, and **precision** is critical.

---

## 📊 Confusion Matrix (Binary Classification)

|                         | **Predicted Positive** | **Predicted Negative** |
|-------------------------|------------------------|------------------------|
| **Actual Positive**     | ✅ True Positive (TP)   | ❌ False Negative (FN) |
| **Actual Negative**     | ❌ False Positive (FP)  | ✅ True Negative (TN)   |

---

## ❗ What Do These Mean?

| Metric | Meaning | Example (Loan Default Prediction) |
|--------|---------|----------------------------------|
| **True Positive (TP)** | Model correctly predicts a default | Defaulted borrower is flagged |
| **False Positive (FP)** | Model incorrectly predicts a default | Safe borrower wrongly flagged |
| **False Negative (FN)** | Model fails to detect a defaulter | Defaulter not flagged (risky!) |
| **True Negative (TN)** | Model correctly predicts non-default | Safe borrower not flagged |

---

## 🧠 Key Metrics

### ✅ Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

> Of all borrowers flagged as risky, how many actually default?

- High precision → fewer **false alarms** (FP)
- Important when false positives are expensive

---

### ✅ Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

> Of all borrowers who actually defaulted, how many did we catch?

- High recall → fewer **missed defaulters** (FN)
- Important when false negatives are costly

---

### ✅ F1 Score

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

> Harmonic mean of precision and recall. Balanced view of performance.

---

### ✅ Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

⚠️ Not reliable for **imbalanced datasets** (e.g., only 5% default rate).

---

## 🎯 Choosing Metrics Based on Business Context

| Scenario | Focus Metric |
|----------|--------------|
| Avoiding financial loss (missed defaulters) | **Recall** |
| Avoiding wasted effort (false alarms) | **Precision** |
| Balanced performance | **F1 Score** |

---

## 💡 Threshold Tuning

By adjusting the classification threshold, you can control the trade-off between precision and recall.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

This helps optimize model for business objectives.

---

## ✅ Summary

- **False Positive** = flagging a good customer (over-warning)
- **False Negative** = missing a bad customer (under-warning)
- **Precision** = how many predictions were actually correct
- **Recall** = how many actual defaulters we caught
- Choose your metrics depending on **what your business cares about most**
