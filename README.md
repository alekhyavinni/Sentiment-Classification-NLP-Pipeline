# Sentiment-Classification-NLP-Pipeline
Built a Python-based supervised learning model to classify restaurant reviews using feature engineering, scikit-learn classifiers, and automated evaluation for safe, human-aligned text classification.

---

## 📌 Project Overview

- **Objective**: To build a text classification model that accurately predicts sentiment labels (Negative, Neutral, Positive) using preprocessed review data.
- **Approach**:
  - Text preprocessing and tokenization using `spaCy`
  - Embedding representation via `Torchtext`
  - LSTM-based deep learning model in `PyTorch`
  - Model evaluation using F1 Score, Accuracy, and Confusion Matrix

---

## 🧰 Tools & Technologies

| Category             | Stack Used                                                                 |
|----------------------|----------------------------------------------------------------------------|
| Language             | Python 3.10                                                                 |
| Libraries            | `PyTorch`, `Torchtext`, `spaCy`, `sklearn`, `matplotlib`, `seaborn`        |
| Model Architecture   | LSTM (Long Short-Term Memory)                                               |
| Evaluation Metrics   | Accuracy, Macro-Averaged F1 Score, Normalized Confusion Matrix             |
| Dataset              | Restaurant review dataset with 3-class sentiment annotations               |

---

## 🧪 Output
![image](https://github.com/user-attachments/assets/10138b1a-d59d-48d0-b6ed-5c2ccac15b1f)

---

## 🔍 Key Features

* End-to-end NLP pipeline: text cleaning, tokenization, vocabulary creation
* Dynamic padding and batch generation using `BucketIterator`
* Bidirectional LSTM for contextual sentiment understanding
* Real-time performance reporting with per-class metrics
* Easy to adapt for other sentiment-labeled datasets

---
## 🚀 How to Run

```bash
python assignment2.py
```

Make sure you have the IMDb dataset loaded and `spaCy` English tokenizer installed:

```bash
python -m spacy download en_core_web_sm
```

---

## 👩‍💻 Author

Alekhya Erikipati – *Master’s in Artificial Intelligence | NLP & ML Enthusiast*

---

## 📎 License

This project is for academic and educational purposes only. Commercial use prohibited without permission.

---

