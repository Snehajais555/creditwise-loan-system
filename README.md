# CreditWise Loan System

## Problem Statement
SecureTrust Bank, a mid-sized financial company, offers personal and home loans across urban and rural India. Loan applications are currently verified manually — loan officers check income proof, employment details, and credit history by hand. This process is slow, inconsistent, and prone to two costly errors: good customers getting rejected, and high-risk customers getting approved.

This project builds an ML-based loan approval system that predicts whether an application should be **Approved or Rejected**, using historical applicant data, to support faster and more consistent decisions before final human verification.

## Dataset
Each row represents a loan applicant with 19 features covering:
- **Personal:** Age, Marital Status, Dependents, Education Level, Gender
- **Financial:** Applicant/Coapplicant Income, Savings, DTI Ratio, Existing Loans
- **Credit:** Credit Score, Collateral Value
- **Loan details:** Loan Amount, Loan Term, Loan Purpose, Property Area, Employer Category

**Target:** `Loan_Approved` (1 = Approved, 0 = Rejected)

## Approach
1. **Preprocessing:** Missing values handled with `SimpleImputer` (mean for numerical columns, most-frequent for categorical columns)
2. **EDA:** Checked class balance, income/credit score distributions, boxplots comparing approved vs. rejected applicants, and a correlation heatmap to identify the strongest predictors
3. **Encoding:** `LabelEncoder` for ordinal-like fields, `OneHotEncoder` for nominal categorical fields (Employment Status, Marital Status, Loan Purpose, Property Area, Employer Category, Gender)
4. **Scaling:** `StandardScaler` applied to features before training
5. **Models trained:** Logistic Regression, KNN, Naive Bayes
6. **Feature engineering:** Added squared terms for DTI Ratio and Credit Score, log-transformed Applicant Income to reduce skew — then retrained all models

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 86.5% | 0.78 | 0.77 | 0.78 |
| **Logistic Regression (after feature engineering)** | **88.0%** | 0.78 | 0.84 | 0.81 |
| Naive Bayes | 86.5% | 0.80 | 0.74 | 0.77 |
| KNN | 76–78.5% | 0.63–0.67 | 0.52–0.57 | 0.57–0.62 |

Logistic Regression with engineered features performed best overall, with the highest recall — important here since failing to catch a genuinely risky applicant is more costly than an overly cautious rejection.

## Tech Stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib
