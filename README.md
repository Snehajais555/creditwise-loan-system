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

## Version 2: Pipeline Approach

`Credit_Wise_Pipe.ipynb` rebuilds the project with scikit-learn `Pipeline` and `ColumnTransformer`, and compares 7 models instead of 3.

**What changed from v1**
- Imputation, scaling and encoding now live inside a pipeline and are learned from the training data only, so no information leaks from the test set.
- Rows with a missing target (5%) are dropped instead of being filled like features.
- Models are chosen with 5-fold cross-validation on the training set, and the test set is used once for the final model.
- Feature engineering kept: squared Credit Score and DTI Ratio, log of Applicant Income.

**Model comparison (5-fold CV, F1)**

| Model | CV F1 |
|---|---|
| Gradient Boosting | 0.930 |
| Decision Tree | 0.917 |
| AdaBoost | 0.912 |
| Random Forest | 0.892 |
| Logistic Regression | 0.773 |
| Naive Bayes | 0.766 |
| KNN | 0.737 |

Tree-based models clearly beat linear and distance-based models here, which suggests approval depends on thresholds (for example credit score and DTI cut-offs) more than on a straight-line relationship.

**Final model: Gradient Boosting (test set, 190 rows)**

| | Precision | Recall | F1 |
|---|---|---|---|
| Rejected (0) | 0.98 | 0.95 | 0.96 |
| Approved (1) | 0.89 | 0.97 | 0.93 |

Accuracy: **95%**. Confusion matrix: 122 correct rejections, 59 correct approvals, 7 wrongly approved, 2 wrongly rejected.
## Tech Stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib
