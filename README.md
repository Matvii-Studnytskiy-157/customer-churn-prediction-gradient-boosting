# Customer Churn Prediction with Gradient Boosting

> An end-to-end machine learning project for predicting telecom customer churn using **XGBoost**, **hyperparameter tuning**, **threshold optimization**, and **business-oriented model interpretation**.

This project builds a complete churn prediction workflow on the **Telco Customer Churn** dataset, with the goal of identifying customers at risk of leaving and converting model output into actionable retention strategy.

The final solution combines:
- rigorous data cleaning and preprocessing
- exploratory data analysis
- baseline model benchmarking
- advanced gradient boosting
- hyperparameter tuning
- threshold optimization
- feature importance analysis
- business-focused recommendations

## Business Context

Customer churn is one of the most important business problems in subscription-based industries. When customers leave, companies lose recurring revenue, increase customer acquisition pressure, and often reveal underlying issues in pricing, service quality, support, onboarding, or overall product value.

In telecom environments, churn prediction is especially valuable because it allows the business to move from **reactive churn response** to **proactive customer retention**.

This project addresses the following question:

> **Can customer churn be predicted accurately enough to support targeted and practical retention intervention?**

## Project Objective

The objective of this project is to develop a machine learning system that predicts whether a telecom customer is likely to churn and to translate those predictions into practical retention insight.

More specifically, the project aims to:

- identify the strongest factors associated with churn risk
- compare baseline and advanced classification models
- optimize the final model for retention-oriented decision-making
- interpret the final model in business terms
- propose actionable customer-retention strategies

The final outcome is not just a classifier, but a **business-oriented churn prediction framework**.

## Dataset

**Dataset:** Telco Customer Churn  
**Source:** Kaggle  
**Target variable:** `Churn`

Each row represents a telecom customer, and the feature space includes information related to:

- demographics
- account structure
- subscribed services
- support and security options
- billing and payment behavior
- monthly and total charges

### Dataset Size
- **Original dataset:** 7,043 rows × 21 columns
- **Cleaned dataset:** 7,032 rows × 20 columns

### Data Cleaning Highlights
The preprocessing stage included:

- detection of hidden missing values in `TotalCharges`
- conversion of `TotalCharges` from text to numeric
- removal of 11 incomplete observations
- removal of the identifier column `customerID`
- binary encoding of the target variable:
  - `1` = churn
  - `0` = no churn

These steps produced a clean, model-ready dataset suitable for both interpretable analysis and predictive modeling.

## Repository Structure

```
customer-churn-prediction-gradient-boosting/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── figures/
│   ├── target_distribution.png
│   ├── churn_rate_by_contract.png
│   ├── churn_rate_by_payment_method.png
│   ├── churn_rate_by_tech_support.png
│   ├── churn_rate_by_online_security.png
│   ├── baseline_model_test_comparison.png
│   ├── baseline_model_roc_curves.png
│   ├── baseline_model_pr_curves.png
│   ├── xgboost_test_comparison.png
│   ├── xgboost_untuned_vs_tuned.png
│   ├── tuned_xgboost_vs_logistic_regression.png
│   ├── threshold_tradeoff_precision_recall_f1.png
│   ├── default_vs_optimized_threshold.png
│   ├── tuned_xgboost_aggregated_feature_importance.png
│   └── final_model_comparison.png
├── notebooks/
│   └── customer-churn-prediction-gradient-boosting.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

The project follows a structured applied machine learning workflow designed to move from raw data to business-ready churn recommendations.

### 1. Data Cleaning and Preprocessing
- hidden missing-value detection
- numeric type correction
- identifier removal
- target encoding

### 2. Exploratory Data Analysis
- target imbalance analysis
- numerical feature comparison by churn group
- churn-rate analysis across customer segments
- encoded feature association overview

### 3. Baseline Model Benchmarking
The following baseline models were compared:
- Logistic Regression
- Decision Tree
- Random Forest

### 4. Advanced Model Development
- XGBoost model construction
- direct comparison against the strongest baseline

### 5. Hyperparameter Tuning
- RandomizedSearchCV
- stratified 5-fold cross-validation
- ROC-AUC optimization

### 6. Threshold Optimization
- precision-recall trade-off analysis
- F1-based threshold selection
- business-oriented operating point selection

### 7. Model Interpretability
- transformed-feature importance
- aggregated original-feature importance
- business-driver interpretation

### 8. Business Translation
- retention segment identification
- action-oriented recommendation design

This workflow ensures that the final model is not only predictive, but also interpretable and operationally useful.

## Modeling Results

### Best Baseline — Logistic Regression
- **Accuracy:** 0.80
- **Precision:** 0.65
- **Recall:** 0.57
- **F1-score:** 0.61
- **ROC-AUC:** 0.84
- **Average Precision:** 0.62

### Tuned XGBoost — Default Threshold (0.50)
- **Accuracy:** 0.81
- **Precision:** 0.65
- **Recall:** 0.56
- **F1-score:** 0.60
- **ROC-AUC:** 0.84
- **Average Precision:** 0.66

### Tuned XGBoost — Optimized Threshold (0.30)
- **Accuracy:** 0.75
- **Precision:** 0.52
- **Recall:** 0.78
- **F1-score:** 0.63
- **ROC-AUC:** 0.84
- **Average Precision:** 0.66

### Performance Interpretation
The final results show that:

- **Logistic Regression** is a very strong baseline and difficult to outperform in default-threshold classification
- **Tuned XGBoost** matches the strongest baseline on standard metrics while improving ranking-oriented churn performance
- **Threshold optimization** makes the final model substantially more effective for proactive retention targeting

This means the final solution is strongest not because it maximizes accuracy alone, but because it produces a better balance between **ranking quality**, **churn capture**, and **practical business usefulness**.

## Final Model Selection

### Final Selected Model
**Tuned XGBoost with an optimized classification threshold of 0.30**

### Why this model was selected
The final model was chosen because it:

- remains highly competitive with the strongest baseline in standard classification metrics
- outperforms the best baseline in **Average Precision**
- improves meaningfully after hyperparameter tuning
- becomes more effective for retention intervention after threshold optimization
- remains interpretable and aligned with the business structure of churn

### Practical interpretation
This final solution supports two complementary use cases:

1. **Ranked churn-risk scoring**  
   Use tuned XGBoost probabilities to prioritize customers by churn risk.

2. **Threshold-based intervention**  
   Use the optimized threshold of **0.30** when the goal is to identify a broader set of likely churners for proactive outreach.

In other words, the selected model is not only the strongest advanced model in the project — it is also the most operationally useful one.

## Key Findings

The final model and the exploratory analysis point to a consistent set of churn-related patterns.

### Most important original features
- **Contract**
- **InternetService**
- **OnlineSecurity**
- **TechSupport**
- **PaperlessBilling**
- **PaymentMethod**
- **tenure**

### Most important transformed features
- `Contract_Month-to-month`
- `InternetService_Fiber optic`
- `OnlineSecurity_No`
- `TechSupport_No`
- `PaperlessBilling_Yes`
- `PaymentMethod_Electronic check`

### Main analytical takeaway
Customer churn in this dataset is strongly associated with:

- weak contractual commitment
- service and support gaps
- riskier billing and payment behavior
- fragile early customer relationships
- certain internet-service configurations

The strongest signal by far is **contract structure**, especially the presence of a **month-to-month contract**, which dominates the final feature importance profile.

## Business Recommendations

The final model suggests several high-priority retention actions.

### 1. Prioritize month-to-month customers
Contract structure is the strongest churn-related signal in the project. Customers on month-to-month plans should be the first target for proactive retention campaigns, loyalty incentives, and contract-conversion offers.

### 2. Investigate the fiber optic segment
Fiber optic customers show elevated churn risk despite belonging to a premium service group. This may reflect dissatisfaction with price, service reliability, or perceived value.

### 3. Use support and security services as retention levers
Customers without **OnlineSecurity** or **TechSupport** are consistently more churn-prone. These add-ons may function not only as services, but also as retention stabilizers.

### 4. Improve billing stability
Payment behavior, particularly **electronic check**, appears strongly associated with churn risk. Encouraging more stable payment methods may reduce customer fragility.

### 5. Strengthen early-tenure intervention
Customers with shorter tenure are especially vulnerable, suggesting that onboarding and early-stage engagement are critical retention windows.

### Strategic implication
Retention effort should not be spread uniformly across the customer base. Instead, it should be concentrated on clearly identifiable high-risk segments revealed by the final model.

## Visual Highlights

The figures below summarize the most important analytical and modeling findings from the project.

### Target distribution
![Target Distribution](figures/target_distribution.png)

### Churn rate by contract
![Churn Rate by Contract](figures/churn_rate_by_contract.png)

### Churn rate by payment method
![Churn Rate by Payment Method](figures/churn_rate_by_payment_method.png)

### Churn rate by tech support
![Churn Rate by Tech Support](figures/churn_rate_by_tech_support.png)

### Baseline model comparison
![Baseline Model Comparison](figures/baseline_model_test_comparison.png)

### Untuned vs tuned XGBoost
![Untuned vs Tuned XGBoost](figures/xgboost_untuned_vs_tuned.png)

### Default vs optimized threshold
![Default vs Optimized Threshold](figures/default_vs_optimized_threshold.png)

### Aggregated feature importance
![Aggregated Feature Importance](figures/tuned_xgboost_aggregated_feature_importance.png)

### Final model comparison
![Final Model Comparison](figures/final_model_comparison.png)

## Tools and Tech Stack

This project was built with the following tools and libraries:

- **Python**
- **pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **XGBoost**
- **Jupyter Notebook / Google Colab**

These tools were used for data cleaning, visualization, preprocessing, model development, hyperparameter tuning, and threshold-based evaluation.

## Limitations

Although the final solution is strong and interpretable, several limitations remain:

- the dataset does not include richer behavioral or service-interaction history
- the model captures statistical associations rather than causal relationships
- threshold selection is not yet tied to explicit intervention cost or retention ROI
- churn behavior may evolve over time as services, pricing, or customer expectations change

## Future Improvements

Natural next steps for extending this project include:

- ROI-based threshold selection
- segment-specific intervention thresholds
- SHAP-based local interpretability
- deployment as a scoring API
- retention dashboard integration
- model monitoring and drift detection

These extensions would move the project closer to a production-ready churn management system.

## Final Takeaway

This project shows that telecom customer churn can be modeled effectively with modern machine learning and translated into practical retention strategy.

The final solution — **Tuned XGBoost with an optimized threshold of 0.30** — provides a churn prediction framework that is:

- analytically rigorous
- operationally useful
- business-interpretable
- portfolio-ready

Beyond predictive modeling alone, this project demonstrates the ability to:

- structure an end-to-end machine learning workflow
- compare baseline and advanced models rigorously
- optimize decision thresholds for practical use
- interpret model behavior in business terms
- translate technical results into actionable recommendations

In short, the project is designed not just to predict churn, but to support smarter retention decisions.

## Executive Summary

- **Problem:** Predict telecom customer churn and support proactive retention
- **Best baseline:** Logistic Regression
- **Best advanced model:** Tuned XGBoost
- **Final threshold:** 0.30
- **Main churn drivers:** Contract, InternetService, OnlineSecurity, TechSupport, PaperlessBilling, PaymentMethod, tenure
- **Best use case:** Retention targeting and churn-risk prioritization
