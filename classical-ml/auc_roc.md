
# 📈 AUC and ROC Curve Explained

In binary classification, especially with imbalanced data, **AUC-ROC** is a critical metric to evaluate a model's performance beyond simple accuracy.

---

## ✅ What is ROC Curve?

The **Receiver Operating Characteristic (ROC) curve** plots:

- **True Positive Rate (TPR)** = Recall = TP / (TP + FN)
- **False Positive Rate (FPR)** = FP / (FP + TN)

It shows how the model's sensitivity and fall-out change as the classification threshold varies.

---

## 📊 What is AUC?

**AUC = Area Under the ROC Curve**  
It represents the model’s ability to distinguish between the positive and negative class.

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: No better than random guessing
- **AUC < 0.5**: Worse than random

> AUC is threshold-independent and ideal for evaluating **ranking quality** of predicted probabilities.

---

## 📌 Why AUC over Accuracy?

| Metric     | Issue with Imbalanced Data        |
|------------|----------------------------------|
| Accuracy   | Can be misleading (e.g., 95% accurate but predicts all negatives) |
| AUC        | Captures model's ability to rank positive > negative |

---

## 💡 Practical Notes

- AUC is good for comparing models before threshold selection
- It tells you: "Given a randomly chosen positive and negative sample, how likely is the model to rank the positive one higher?"
- For highly imbalanced tasks, **Precision-Recall AUC** might be more informative

---

## ✅ Summary

- **ROC Curve** shows model's performance across all thresholds
- **AUC** quantifies it as a single number
- High AUC → Model does a good job separating classes
- Especially useful when dealing with **imbalanced datasets**
