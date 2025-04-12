
# 🧠 LSTM for Anomaly Detection – Interview FAQ

This note summarizes common interview questions and answers related to LSTM-based time series anomaly detection, especially as applied to early-stage credit risk modeling.

---

## ❓ 1. Why LSTM over CNN or traditional methods like ARIMA?

- LSTM captures long-term dependencies using memory cells and gates.
- ARIMA assumes linearity and stationarity, which rarely hold in borrower behavior.
- CNN works well on local patterns but lacks long-term temporal modeling.
- LSTM handles irregular sequences like repayment patterns, delayed behaviors, etc.

---

## ❓ 2. How does LSTM work internally?

LSTM uses three gates to control information flow:

- **Input Gate** – decides how much new info to write
- **Forget Gate** – decides how much past info to keep
- **Output Gate** – controls the output from memory

These allow LSTM to remember long-term patterns and avoid vanishing gradients.

---

## ❓ 3. Why use an Autoencoder instead of a classifier?

- Early-stage anomalies are hard to label → use unsupervised learning
- Autoencoder learns to reconstruct “normal” sequences
- If reconstruction error is high, input is likely abnormal

---

## ❓ 4. How do you construct the input sequences?

- Use a sliding window (e.g. 6 or 12 months) of borrower features
- Shape: (T, n_features) per input
- Normalization: MinMax or z-score

---

## ❓ 5. How is the anomaly threshold set?

- Train only on normal data
- Compute reconstruction error distribution
- Use 95th percentile or IQR-based threshold
- Tune threshold using post-hoc validation on known defaulters

---

## ❓ 6. What labels were used for training?

- None! Model is trained only on “healthy” borrowers
- It is a fully unsupervised setup for anomaly detection

---

## ❓ 7. How do you handle model drift or aging?

- Set up monthly retraining pipeline
- Update model and threshold every month using rolling window
- Validate reconstruction on recent normal samples

---

## ❓ 8. Did you try other models (GRU / CNN / Transformer)?

- GRU tried for lighter setup; performance was similar
- CNN lacks long-range memory
- Transformer requires more data and engineering; could be future extension

---

## ❓ 9. How did you deploy the model?

- Trained offline on GPU
- Exported with ONNX / TensorFlow Lite
- Deployed in inference engine with latency < 200ms

---

## ❓ 10. How do you evaluate precision / recall without labels?

- Use historical defaulters to see if anomaly score triggered earlier
- Measure “advance warning rate”
- Plot anomaly score distributions to see separation

---

## ✅ Summary

- LSTM excels in time-dependent sequence modeling
- Autoencoder is suitable for unsupervised anomaly detection
- A clear thresholding strategy and validation process are key
