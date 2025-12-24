# Approaches-for-Psychological-Risk-Stratification
DASS-21 Depression Risk Prediction using Machine Learning

# DASS-21 Depression Risk Prediction using Machine Learning

A machine learning pipeline for predicting psychological risk levels (depression) based on the DASS-21 (Depression, Anxiety, and Stress Scale) survey instrument. This project compares five classification algorithms and implements best practices for handling psychological survey data.

## Overview

### Problem Statement
Mental health disorders affect over 280 million people worldwide. Early identification of individuals at risk for depression remains challenging due to limited access to mental health professionals. This project explores whether machine learning can predict depression risk from personality traits and demographic variables without directly asking about depression symptoms.

### Objective
Build and compare machine learning models that classify individuals into:
- Class 0: Lower psychological risk (Normal/Mild depression)
- Class 1: Higher psychological risk (Moderate to Severe depression)

### Key Innovation
We explicitly prevent construct-level data leakage by excluding all DASS items (depression, anxiety, stress) from the feature set, ensuring that predictions are based on genuinely independent variables like personality and demographics.

## Dataset

Source: Open Psychometrics (https://openpsychometrics.org/)

- Total Samples: 39,775
- Features: 172 (raw) → 38 (selected)
- Target Classes: 2 (Binary)
- Class Distribution: 46.2% Low Risk / 53.8% High Risk

### Features Used

| Category | Features | Description |
|----------|----------|-------------|
| Personality (TIPI) | 10 items | Big Five traits: Extraversion, Agreeableness, Conscientiousness, Emotional Stability, Openness |
| Vocabulary (VCL) | 16 items | Attention/validity check items |
| Demographics | 8+ variables | Age, gender, education, country, urbanicity, religion, race, marital status, family size |

### Features Excluded (to prevent leakage)
- All 21 DASS items (Q1A-Q21A)
- Derived scores (depression_score, anxiety_score, stress_score)
- Response timing metadata

## Methodology

### Steps

1. Data Loading: Load DASS-21 dataset with flexible separator handling
2. Target Creation: Binary classification using median split (cutoff = 34.0)
3. Preprocessing: Missing value imputation, outlier detection and capping, categorical encoding
4. Feature Selection: Remove DASS items to prevent construct leakage
5. Train/Test Split: 80/20 stratified split
6. Class Balancing: SMOTE oversampling on training set
7. Feature Scaling: StandardScaler (z-score normalization)
8. Model Training: GridSearchCV with 5-fold cross-validation
9. Evaluation: Multiple metrics on held-out test set

## Models

Five classification algorithms were evaluated:

| Model | Description |
|-------|-------------|
| Decision Tree | Rule-based classifier with interpretable splits |
| Naive Bayes | Probabilistic classifier assuming feature independence |
| Logistic Regression | Linear model with probability outputs |
| Random Forest | Ensemble of decision trees |
| MLP (Neural Network) | Multi-layer perceptron for complex patterns |

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Decision Tree | 0.887 | 0.893 | 0.897 | 0.895 | 0.958 |
| Naive Bayes | 0.876 | 0.870 | 0.870 | 0.870 | 0.952 |
| Logistic Regression | 0.904 | 0.915 | 0.906 | 0.911 | 0.971 |
| Random Forest | 0.905 | 0.911 | 0.912 | 0.912 | 0.970 |
| MLP (Neural Network) | 0.876 | 0.885 | 0.884 | 0.884 | 0.949 |

### Best Model: Logistic Regression
- Accuracy: 90.4%
- ROC-AUC: 0.971
- F1-Score: 0.911


## Key Findings

1. Construct-Level Leakage Matters: Including anxiety and stress items as predictors resulted in near-perfect accuracy (100%), which was misleading. After removing all DASS items, realistic performance (~90%) was achieved.

2. Linear Models Perform Well: Logistic Regression matched or exceeded complex models, suggesting the relationship between personality and depression risk is approximately linear.

3. Personality Predicts Depression Risk: Big Five personality traits, especially Emotional Stability and Extraversion, showed strong predictive power for depression risk.

4. Model Choice Flexibility: All models achieved AUC > 0.94, indicating the predictive signal can be captured by various algorithms.

## Limitations

- Cross-sectional data: Cannot establish causal relationships
- Online convenience sample: May not generalize to clinical populations
- Self-report bias: Social desirability and recall issues
- Binary classification: Oversimplifies the continuous nature of depression

## Future Work

- Longitudinal validation for predicting future depression onset
- External validation on clinical samples
- Multi-class prediction (Normal, Mild, Moderate, Severe)
- Feature importance analysis and model interpretability

## References

- Lovibond, S.H. & Lovibond, P.F. (1995). Manual for the Depression Anxiety Stress Scales
- Gosling, S.D., et al. (2003). A very brief measure of the Big-Five personality domains
- Chawla, N.V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique


