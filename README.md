# Fraud Detection Analysis 🔍💰

A comprehensive machine learning project for detecting fraudulent financial transactions using multiple classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow.svg)

## 📋 Problem Statement

In this project, we analyze a dataset containing financial transaction metrics including transaction type, amount, account balances, and more. The goal is to develop a predictive model capable of accurately identifying **fraudulent transactions**. Given the severe financial implications of missing a fraudulent transaction, our primary emphasis is on ensuring high **recall for the fraud class**.

## 🎯 Objectives

- **Explore the Dataset**: Uncover patterns, distributions, and relationships within the transaction data
- **Conduct Extensive EDA**: Dive deep into bivariate relationships against the target variable
- **Data Preprocessing**: Handle missing values, outliers, encode categorical variables, and transform skewed features
- **Model Building**: Implement and tune multiple classification models (Decision Tree, Random Forest, Logistic Regression, SVM)
- **Model Evaluation**: Compare models using precision, recall, F1-score, and accuracy metrics
- **Business Intelligence**: Create interactive Power BI dashboards for fraud pattern analysis

## 📊 Dataset Description

| Variable | Description |
|----------|-------------|
| `step` | Time unit (1 step = 1 hour) |
| `type` | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount in local currency |
| `nameOrig` | Customer who initiated the transaction |
| `oldbalanceOrg` | Initial balance of origin account |
| `newbalanceOrig` | New balance of origin account after transaction |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` | Initial balance of destination account |
| `newbalanceDest` | New balance of destination account after transaction |
| `isFraud` | Target variable (1 = Fraud, 0 = Non-Fraud) |

## 📈 Key Findings

- **Dataset Size**: 11,142 transactions
- **Fraud Rate**: 10.25% (1,142 fraudulent transactions)
- **Critical Insight**: Fraud occurs exclusively in TRANSFER and CASH_OUT transaction types
- **Best Model**: Decision Tree with **100% recall** for fraud detection

## 🏆 Model Performance Comparison

| Model | Recall (Fraud) | Precision (Fraud) | F1-Score | Accuracy |
|-------|----------------|-------------------|----------|----------|
| **Decision Tree** | **1.00** | 0.97 | 0.98 | 1.00 |
| Random Forest | 0.99 | 1.00 | 1.00 | 1.00 |
| SVM | 0.99 | 1.00 | 0.99 | 1.00 |
| Logistic Regression | 0.96 | 0.98 | 0.97 | 0.99 |

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-Learn** - Machine learning models
- **SciPy** - Box-Cox transformations
- **Power BI** - Business intelligence dashboards

## 📁 Project Structure

```
fraud-detection-analysis/
|
|-- README.md                           # Project documentation
|-- Fraud_Detection_Analysis.ipynb     # Main Jupyter notebook
|-- fraud_data_for_powerbi.csv         # Processed dataset for Power BI
|-- Fraud_Analysis_Dataset.csv         # Original dataset
|
|-- visualizations/                     # Generated plots
|   |-- 01_continuous_distribution.png
|   |-- 02_categorical_distribution.png
|   |-- 03_numerical_vs_target.png
|   |-- 04_categorical_vs_target.png
|   |-- 05_correlation_heatmap.png
|   |-- 06_boxcox_transformation.png
|   |-- 07_dt_confusion_matrix.png
|   |-- 08_rf_confusion_matrix.png
|   |-- 09_lr_confusion_matrix.png
|   |-- 10_svm_confusion_matrix.png
|   |-- 11_model_comparison_recall.png
|   |-- 12_comprehensive_comparison.png
|
|-- powerbi/                            # Power BI files
    |-- Fraud_Detection_Dashboard.pbix
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the Analysis

1. Clone the repository:
```bash
git clone https://github.com/Codewithaiyan/fraud-detection-analysis.git
cd fraud-detection-analysis
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Fraud_Detection_Analysis.ipynb
```

3. Or run in Google Colab:
   - Upload the notebook to Google Colab
   - Upload the dataset
   - Run all cells

## 📊 Power BI Dashboard

The project includes an interactive Power BI dashboard featuring:
- KPI Cards (Total Transactions, Fraud Cases, Fraud Rate)
- Fraud Distribution Donut Chart
- Transaction Type Analysis
- Time Series Fraud Trends
- Interactive Slicers for filtering

## 🔍 Analysis Workflow

1. **Data Loading & Overview**
   - Import dataset
   - Check data types and structure
   - Summary statistics

2. **Exploratory Data Analysis (EDA)**
   - Univariate analysis (distributions)
   - Bivariate analysis (features vs target)
   - Correlation analysis

3. **Data Preprocessing**
   - Remove irrelevant features (customer IDs)
   - Handle missing values
   - Treat outliers using Box-Cox transformation
   - One-hot encode categorical variables
   - Feature scaling for distance-based models

4. **Model Building & Tuning**
   - Train-test split (80-20, stratified)
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation with StratifiedKFold
   - Optimize for recall (fraud class)

5. **Model Evaluation**
   - Classification reports
   - Confusion matrices
   - Compare models on multiple metrics

## 📝 Key Insights

- **Time Pattern**: Fraudulent transactions occur predominantly in early time steps
- **Transaction Type**: Only TRANSFER and CASH_OUT types contain fraud
- **Balance Behavior**: Fraudulent transactions typically drain the entire origin account balance
- **Amount Pattern**: Fraud amounts vary widely from small to very large transactions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## 👤 Author

**Your Name**
- GitHub: [@Codewithaiyan](https://github.com/Codewithaiyan)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
