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

