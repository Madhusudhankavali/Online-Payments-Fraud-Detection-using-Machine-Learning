# Online Payments Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20Scikit--learn-orange.svg)

This repository contains the final project for the **Smartinternz:Artificial intelligence and machine learning** course. The project focuses on developing a robust machine learning model to detect fraudulent online payment transactions with high accuracy, precision, and recall.

**TEAM ID:**
 LTVIP2026TMIDS88452
 
**Team Members:**
* E Yashwanth Gowd(22BFA33220)
* K Madhusudhan(22BFA33226)
* G Jyothi Swaroop(22BFA33221)
* K Gurusai(22BFA33225)

---

## Dataset Download Link
[Kaggle - Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)

## 1. Project Overview

### 🎯 Project Goal
Online payment fraud is a critical challenge in the financial sector, causing billions of dollars in losses annually. This project aims to develop a robust classification system that can accurately distinguish between legitimate and fraudulent transactions, minimizing both financial losses (false negatives) and customer friction (false positives).

### 📊 The Dataset & Key Challenge
The project utilizes a real-world dataset of 6,362,620 online payment transactions.

The fundamental challenge is the **severe class imbalance**:
* **Legitimate Transactions:** 6,354,407 (99.871%)
* **Fraudulent Transactions:** 8,213 (0.129%)
* **Imbalance Ratio:** Approximately 1 fraudulent transaction for every 774 legitimate ones.

This imbalance means a naive model can achieve 99.8% accuracy by simply predicting "non-fraud" for every transaction, making traditional accuracy a useless metric.

## 2. Methodology & Pipeline

The project follows a systematic ML pipeline to address the class imbalance and build a high-performance model.

### Step 1: Data Preprocessing
To prepare the data for modeling, the following steps were taken:
* **Feature Removal:** Dropped `isFlaggedFraud` (redundant) and `nameOrig` / `nameDest` (extremely high cardinality, risk of overfitting).
* **Feature Transformation:** Applied a `log1p` transformation to the `amount` feature to handle its severe right-skewness.
* **Encoding:** Used `LabelEncoder` to convert the categorical `type` feature (e.g., 'CASH_OUT', 'PAYMENT') into numerical values.
* **Data Splitting:** The data was split into training (70%) and testing (30%) sets using a stratified split to preserve the class imbalance ratio in both sets.

### Step 2: Handling Class Imbalance (SMOTEENN)
A rigorous comparative analysis of different imbalance strategies was the project's core innovation. The most effective strategy was **SMOTEENN**, a hybrid technique that combines oversampling and undersampling:
1.  **SMOTE (Synthetic Minority Over-sampling Technique):** Generates new, synthetic "fraud" instances by interpolating between existing fraud samples.
2.  **ENN (Edited Nearest Neighbors):** Removes ambiguous or noisy samples from both classes, cleaning the decision boundary.

This technique was applied *only* to the training data to create a more balanced set for the model to learn from, while the test set remained in its original, imbalanced state for a realistic evaluation.

### Step 3: Model Selection & Hyperparameter Tuning
Five different classification algorithms were tested:
1.  Random Forest
2.  Decision Tree
3.  Extra Trees Classifier
4.  Linear SVC
5.  **XGBoost Classifier (Winner)**

The **XGBoost Classifier** was identified as the most promising model. It underwent a multi-stage tuning process to find the optimal parameters, progressing from a broad `RandomizedSearchCV` to a focused `GridSearchCV`, and finally to **manual tuning**.

This process, combined with the SMOTEENN-resampled data, yielded the final, optimized model.

#### Final Optimal Parameters (XGBoost):
* `n_estimators`: 150
* `max_depth`: 6
* `learning_rate`: 0.1
* `scale_pos_weight`: 774 (Set to the class imbalance ratio)

## 3. 🏆 Final Results

The manually-tuned XGBoost model, trained on SMOTEENN-resampled data, achieved the best and most balanced performance on the unseen 30% test set.

The primary success metric was the **F1-Score for the fraud class**, which balances the tradeoff between precision and recall.

### Performance Metrics (Tuned XGBoost):
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Fraud F1-Score** | **0.89** | The harmonic mean of Precision & Recall. The primary success metric. |
| **Fraud Recall** | **0.86** | The model successfully identified 86% of all actual fraudulent transactions. |
| **Fraud Precision** | **0.93** | Of all transactions the model flagged as fraud, 93% were actually fraudulent. |
| **Overall Accuracy** | **99.97%** | The total percentage of correct predictions. |

### Confusion Matrix (Tuned XGBoost on Test Set):
This matrix shows the model's performance on the 1,908,786 transactions in the test set.

| | **Predicted Fraud** | **Predicted Legit** |
| :--- | :--- | :--- |
| **Actual Fraud** | **2,073 (TP)** | **346 (FN)** |
| **Actual Legit** | **146 (FP)** | **1,906,221 (TN)** |

**Interpretation:**
* **True Positives (TP):** 2,073 fraudulent transactions were correctly caught.
* **False Negatives (FN):** 346 fraudulent transactions were missed (14.3% of frauds).
* **False Positives (FP):** Only 146 legitimate transactions were incorrectly flagged as fraud, demonstrating excellent precision and minimizing customer friction.

## 4. 🛠️ Technology Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Class Imbalance:** Imbalanced-learn (for SMOTEENN)
* **Modeling:** XGBoost

## 5. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Aryzenshi/Al-Project-BCSE306.git](https://github.com/Aryzenshi/Al-Project-BCSE306.git)
    cd Al-Project-BCSE306
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
    ```

4.  **Run the analysis:**
    * The primary code and analysis are contained within the Jupyter Notebook (`.ipynb`) file.
    * Open and run the notebook cells sequentially to reproduce the preprocessing, model training, tuning, and evaluation.
    * The dataset (`online-fraud-dataset.csv`) is required.

## 6. Acknowledgments

* This project was completed for the **Smartinternz:Artificial intelligence and machine learning** course.
* We acknowledge the use of Generative AI tools (Gemini, ChatGPT) for assistance in generating boilerplate code, debugging the SMOTEENN pipeline, and refining the hyperparameter tuning methodology.
