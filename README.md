# customer-churn-prediction-gradient-boosting

## Overview
Customer churn is one of the most important business challenges in subscription-based industries. When customers leave, companies lose recurring revenue, increase customer acquisition pressure, and often expose deeper issues in pricing, service quality, support, or overall customer experience.

This project develops an end-to-end machine learning workflow for predicting telecom customer churn using the **Telco Customer Churn** dataset. The objective is not only to build a strong predictive model, but also to identify the customer, service, contract, and billing patterns most strongly associated with churn risk.

The final solution combines:

- rigorous data cleaning and preprocessing
- exploratory data analysis
- baseline model benchmarking
- XGBoost model development
- hyperparameter tuning
- threshold optimization
- feature importance interpretation
- business-oriented retention recommendations

---

## Business Objective
The goal of this project is to predict whether a customer is likely to churn and to support **proactive retention strategy**.

From a business perspective, the final model can be used in two ways:

1. **Ranked churn-risk scoring**  
   Prioritize customers by churn probability for targeted retention campaigns.

2. **Threshold-based intervention rule**  
   Use an optimized classification threshold to identify a broader set of likely churners for proactive outreach.

---

## Dataset
**Dataset:** Telco Customer Churn  
**Source:** Kaggle  
**Target variable:** `Churn`

Each row represents a telecom customer, and the features describe:

- customer demographics
- account and contract structure
- subscribed services
- support and security options
- billing and payment behavior
- monthly and total charges

After cleaning, the working dataset contains:

- **7,032 rows**
- **20 columns**

### Data Cleaning Highlights
Key preprocessing steps included:

- detection of hidden missing values in `TotalCharges`
- conversion of `TotalCharges` from object to numeric
- removal of 11 incomplete rows
- removal of `customerID`
- binary encoding of the target variable:
  - `1` = churn
  - `0` = no churn

---

## Project Structure

```text
customer-churn-prediction-gradient-boosting/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── figures/
│   ├── target_distribution.png
│   ├── churn_rate_by_contract.png
│   ├── churn_rate_by_payment_method.png
│   ├── churn_rate_by_tech_support.png
│   ├── churn_rate_by_online_security.png
│   ├── baseline_model_cv_comparison.png
│   ├── baseline_model_test_comparison.png
│   ├── baseline_model_roc_curves.png
│   ├── baseline_model_pr_curves.png
│   ├── xgboost_cv_comparison.png
│   ├── xgboost_test_comparison.png
│   ├── xgboost_vs_logistic_regression.png
│   ├── xgboost_hyperparameter_search_top_configs.png
│   ├── xgboost_untuned_vs_tuned.png
│   ├── tuned_xgboost_vs_logistic_regression.png
│   ├── threshold_tradeoff_precision_recall_f1.png
│   ├── default_vs_optimized_threshold.png
│   ├── tuned_xgboost_top_feature_importance.png
│   ├── tuned_xgboost_aggregated_feature_importance.png
│   └── final_model_comparison.png
├── notebooks/
│   └── customer-churn-prediction-gradient-boosting.ipynb
├── README.md
├── requirements.txt
└── .gitignore
Methodology
1. Exploratory Data Analysis

The EDA stage focused on:

target imbalance analysis
numerical feature comparison by churn group
churn-rate analysis across categorical segments
encoded feature association overview
2. Baseline Model Benchmarking

The following baseline models were benchmarked:

Logistic Regression
Decision Tree
Random Forest

Among them, Logistic Regression emerged as the strongest baseline.

3. Advanced Model Development

The main advanced model in the project is XGBoost, chosen because gradient boosting is particularly effective on structured tabular data and can capture nonlinear interactions between features.

4. Hyperparameter Tuning

The XGBoost model was optimized with RandomizedSearchCV using stratified 5-fold cross-validation and ROC-AUC as the tuning objective.

5. Threshold Optimization

Because churn prediction is a decision-sensitive business task, the model was evaluated across multiple classification thresholds.

The final selected operating threshold was:

0.30

This produced a more recall-oriented and retention-focused intervention rule than the default threshold of 0.50.

6. Model Interpretability

Feature importance analysis was used to identify the main churn drivers and translate model behavior into business insight.

Model Performance
Best Baseline — Logistic Regression
Accuracy: 0.80
Precision: 0.65
Recall: 0.57
F1-score: 0.61
ROC-AUC: 0.84
Average Precision: 0.62
Final Advanced Model — Tuned XGBoost (Default Threshold = 0.50)
Accuracy: 0.81
Precision: 0.65
Recall: 0.56
F1-score: 0.60
ROC-AUC: 0.84
Average Precision: 0.66
Final Selected Operating Point — Tuned XGBoost (Optimized Threshold = 0.30)
Accuracy: 0.75
Precision: 0.52
Recall: 0.78
F1-score: 0.63
ROC-AUC: 0.84
Average Precision: 0.66
Final Model Selection

The final selected solution is:

Tuned XGBoost with an optimized threshold of 0.30

This model was chosen because it:

remains highly competitive with the strongest baseline
achieves stronger ranking-oriented churn performance
responds successfully to hyperparameter tuning
becomes substantially more useful for retention targeting after threshold optimization
is supported by interpretable and business-consistent feature importance
Key Findings
Strongest churn drivers

The most important variables in the final tuned XGBoost model were:

Contract
InternetService
OnlineSecurity
TechSupport
PaperlessBilling
PaymentMethod
tenure
Most important transformed features

At the transformed-feature level, the strongest signals included:

Contract_Month-to-month
InternetService_Fiber optic
OnlineSecurity_No
TechSupport_No
PaperlessBilling_Yes
PaymentMethod_Electronic check
Strategic insight

Churn in this dataset is strongly associated with:

weak contractual commitment
fragile early customer relationships
support/security gaps
billing and payment behavior
internet service profile
Business Recommendations
1. Prioritize month-to-month customers

This is the strongest churn-related segment in the final model. These customers should be the top priority for proactive retention efforts.

2. Investigate fiber optic churn

Fiber optic customers appear more churn-prone than other internet-service groups. This may indicate pricing dissatisfaction, service-quality issues, or unmet value expectations.

3. Use OnlineSecurity and TechSupport as retention levers

Customers without these services are consistently more likely to churn. Support and security add-ons may function as retention stabilizers.

4. Improve billing stability

Payment behavior, especially electronic check usage, appears closely associated with churn risk. Encouraging migration to automatic payment methods may reduce churn exposure.

5. Focus on early-tenure customers

Customers with shorter tenure are especially vulnerable, suggesting that onboarding and early-lifecycle intervention are critical.

Visual Highlights
Target distribution

Churn rate by contract

Churn rate by payment method

Churn rate by tech support

Baseline model comparison

Untuned vs tuned XGBoost

Default vs optimized threshold

Aggregated feature importance

Final model comparison

Tools and Libraries

This project uses:

Python
pandas
NumPy
Matplotlib
Seaborn
scikit-learn
XGBoost
Jupyter / Google Colab
Reproducibility

To reproduce the project locally:

git clone <your-repository-url>
cd customer-churn-prediction-gradient-boosting
pip install -r requirements.txt

Then open the notebook in:

Jupyter Notebook, or
Google Colab
Limitations

Although the final model is strong and interpretable, several limitations remain:

the dataset does not include richer behavioral or service-interaction history
the model captures associations rather than causal effects
threshold choice should ideally be aligned with actual business intervention cost
model performance may shift over time as pricing, services, and customer behavior evolve
Future Improvements

Possible next steps include:

explicit ROI-based threshold selection
segment-specific thresholds
SHAP-based local interpretability
deployment as a prediction API
model monitoring and drift tracking
integration into a retention scoring dashboard
Final Takeaway

This project shows that telecom customer churn can be modeled effectively with modern machine learning and translated into practical retention strategy.

The final result — Tuned XGBoost with a threshold of 0.30 — provides a strong churn-risk framework that is both analytically rigorous and operationally useful.

It demonstrates not only predictive modeling capability, but also the ability to connect machine learning outputs to real business decision-making.
