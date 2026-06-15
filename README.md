# creditcardfrauddetection
# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning project that detects fraudulent credit card transactions using **Logistic Regression**. The model is trained on a balanced dataset created from the highly imbalanced Credit Card Fraud Detection dataset and allows users to predict whether a transaction is legitimate or fraudulent.

---

## 📌 Features

- Data preprocessing using Pandas
- Handles imbalanced data through undersampling
- Feature scaling using StandardScaler
- Logistic Regression classification model
- Model evaluation using accuracy score
- Real-time transaction prediction through user input
- Fraud probability estimation

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## 📂 Dataset

The dataset contains anonymized credit card transactions made by European cardholders.

### Attributes

| Feature Type | Description |
|-------------|-------------|
| Time | Seconds elapsed between each transaction and the first transaction |
| V1 - V28 | PCA transformed features |
| Amount | Transaction amount |
| Class | Target variable (0 = Legitimate, 1 = Fraudulent) |

### Dataset Statistics

- Total Transactions: 284,807
- Legitimate Transactions: 284,315
- Fraudulent Transactions: 492

---

## ⚙️ Project Workflow

### 1. Load Dataset

```python
data = pd.read_csv("creditcard.csv")
```

### 2. Separate Fraud and Legitimate Transactions

```python
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]
```

### 3. Balance Dataset

Since fraud cases are extremely rare, random undersampling is used.

```python
legit_sample = legit.sample(n=len(fraud), random_state=42)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

### 4. Feature and Target Separation

```python
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

### 5. Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

### 6. Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 7. Train Logistic Regression Model

```python
model = LogisticRegression(max_iter=5000)

model.fit(X_train, Y_train)
```

### 8. Model Evaluation

```python
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Testing Accuracy:", accuracy_score(Y_test, test_pred))
```

### 9. User Transaction Prediction

Users can manually enter transaction details.

```text
(Time, V1, V2, ..., V28, Amount)
```

The model predicts:

- Legitimate Transaction
- Fraudulent Transaction

along with confidence probabilities.

---

## 📊 Sample Output

```text
Training Accuracy: 95.3%
Testing Accuracy: 92.8%

Prediction Result

Transaction is LEGITIMATE

Legitimate Probability: 0.9874
Fraud Probability: 0.0126
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

### Navigate to Project Folder

```bash
cd credit-card-fraud-detection
```

### Install Required Libraries

```bash
pip install pandas numpy scikit-learn
```

---

## ▶️ Run Project

```bash
python fraud_detection.py
```

Enter transaction values when prompted.

---

## 📈 Future Improvements

- Random Forest Classifier
- XGBoost Classifier
- SMOTE for advanced balancing
- Flask Web Application
- Streamlit Dashboard
- Real-time API Integration
- Model Deployment on Cloud

---

## 📋 Results

| Metric | Score |
|----------|----------|
| Algorithm | Logistic Regression |
| Data Scaling | StandardScaler |
| Data Balancing | Random Undersampling |
| Classification Type | Binary Classification |
| Prediction Output | Fraud / Legitimate |

---

## 🎯 Learning Outcomes

- Data preprocessing and cleaning
- Handling imbalanced datasets
- Feature scaling techniques
- Logistic Regression implementation
- Model evaluation and prediction
- Machine Learning project deployment basics

---
