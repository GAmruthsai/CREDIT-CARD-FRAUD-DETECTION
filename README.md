# 💳 Credit Card Fraud Detection

### AdaBoost + Majority Voting Ensemble | B.Tech Major Project

-----

## 📌 Project Overview

This project detects fraudulent credit card transactions using an ensemble approach combining **AdaBoost** and a **Majority Voting classifier**. The model is trained on the highly imbalanced Kaggle Credit Card Fraud Dataset and addresses class imbalance using SMOTE oversampling.

-----

## 🧠 Models Used

|Model                             |Type              |
|----------------------------------|------------------|
|AdaBoost (with Decision Tree base)|Boosting Ensemble |
|Logistic Regression               |Base Voter        |
|Random Forest                     |Base Voter        |
|AdaBoost                          |Base Voter        |
|**Majority Voting**               |**Final Ensemble**|

-----

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/                   # Place creditcard.csv here (from Kaggle)
│   └── .gitkeep
│
├── notebooks/
│   └── exploration.ipynb   # EDA and model experimentation
│
├── src/
│   ├── preprocess.py       # Data loading, scaling, SMOTE
│   ├── train.py            # Model training
│   ├── evaluate.py         # Metrics, confusion matrix, ROC curve
│   └── predict.py          # Predict on new data
│
├── models/                 # Saved trained models (.pkl)
│   └── .gitkeep
│
├── reports/
│   └── figures/            # Confusion matrix, ROC curve images
│
├── tests/
│   └── test_pipeline.py    # Basic unit tests
│
├── main.py                 # Run full pipeline
├── requirements.txt        # All dependencies
└── README.md
```

-----

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

- Go to: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- Download `creditcard.csv`
- Place it inside the `data/` folder

### 4. Run the full pipeline

```bash
python main.py
```

-----

## 📊 Results (Expected)

|Metric           |AdaBoost|Majority Voting|
|-----------------|--------|---------------|
|Accuracy         |~99.9%  |~99.9%         |
|Precision (Fraud)|~90%    |~92%           |
|Recall (Fraud)   |~85%    |~88%           |
|F1-Score (Fraud) |~87%    |~90%           |
|ROC-AUC          |~97%    |~98%           |

-----

## 🛠️ Tech Stack

- Python 3.10+
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib, seaborn
- joblib

-----

## 👨‍💻 Author

**Amruth Sai Guggilla**  
B.Tech Computer Science — JNTU  
[LinkedIn](https://linkedin.com/in/amruthsaiguggilla)

-----

## 📄 License

MIT License