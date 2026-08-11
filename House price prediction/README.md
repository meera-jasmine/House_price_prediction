# House Prices Prediction using Machine Learning

Predicting residential house sale prices using property characteristics such as overall quality, living area, garage capacity, basement area, neighborhood, and other housing attributes.

## Problem Statement

The objective of this project is to develop a machine learning regression model that predicts the selling price of a house based on its available property features.

The project focuses on:

1. Understanding and analyzing the housing dataset.
2. Cleaning and preprocessing the data.
3. Exploring relationships between house characteristics and sale prices.
4. Building and comparing multiple regression models.
5. Evaluating models using MAE, RMSE, and R² Score.
6. Applying hyperparameter tuning using GridSearchCV.
7. Selecting the best-performing model for house price prediction.

## Project Type

**Regression** — The target variable `SalePrice` is a continuous numerical value representing the selling price of a house.

## Contents

- **`House Prices Prediction.ipynb`** — Complete notebook containing data loading, data inspection, missing-value handling, exploratory data analysis, categorical encoding, train-test splitting, feature scaling, model training, evaluation, model comparison, and hyperparameter tuning.
- **`MODEL_COMPARISON_REPORT.md`** — Detailed comparison of the five regression models and explanation of the final model selection.
- **`CHALLENGES_REPORT.md`** — Data preprocessing, modeling, evaluation, and hyperparameter-tuning challenges encountered during the project.

## Dataset

The dataset contains **1,460 house records and 81 original columns**.

The features describe different aspects of residential properties, including:

- Property size and lot characteristics
- Overall quality and condition
- Construction and remodeling information
- Neighborhood and location-related attributes
- Basement characteristics
- Garage characteristics
- Rooms and bathrooms
- Exterior and interior features
- Sale information

### Target Variable

**`SalePrice`** — The final selling price of the house and the continuous target variable used for regression.

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Inspected shape, columns, data types, descriptive statistics, and missing values.
- Identified columns with missing values.
- Removed high-missing-value columns such as `Alley`, `MasVnrType`, `FireplaceQu`, `PoolQC`, `Fence`, and `MiscFeature`.
- Filled missing numerical values in `LotFrontage`, `MasVnrArea`, and `GarageYrBlt` using median imputation.
- Filled remaining categorical missing values using the mode.
- Converted categorical variables into numerical features using one-hot encoding with `drop_first=True`.
- Separated `SalePrice` as the target variable.
- Split the data into training and testing sets using an 80:20 ratio.
- Applied `StandardScaler` for the Linear Regression model.

## Exploratory Data Analysis

The notebook includes visual analysis of important relationships in the dataset, including:

- Neighborhood vs Sale Price
- Overall Quality vs Sale Price
- Ground Living Area vs Sale Price
- Garage Capacity vs Sale Price
- Top features correlated with Sale Price
- Distribution of SalePrice and important numerical variables
- Outlier analysis
- Actual vs Predicted plots
- Residual plots
- Model R² comparison

## Key Findings

- **Overall Quality (`OverallQual`)** showed a strong positive relationship with `SalePrice`, with a correlation of approximately **0.79**.
- **Ground Living Area (`GrLivArea`)** showed a strong positive relationship with house prices.
- **Neighborhood** was an important factor in the variation of house sale prices.
- Houses with greater **garage capacity (`GarageCars`)** generally showed higher sale prices.
- The `SalePrice` distribution was right-skewed, with a smaller number of high-priced properties.
- Ensemble tree-based models performed substantially better than the baseline Linear Regression model.
- XGBoost achieved the best test-set performance among the five models evaluated.

## Machine Learning Models

Five regression algorithms were trained and evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**

### Evaluation Metrics

- **MAE (Mean Absolute Error)** — Average absolute difference between actual and predicted prices. Lower is better.
- **RMSE (Root Mean Squared Error)** — Penalizes larger prediction errors more strongly. Lower is better.
- **R² Score** — Measures the proportion of target variance explained by the model. Higher is better.

## Model Performance Summary

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| **XGBoost** | **17,054.11** | **26,417.55** | **0.9090** |
| Random Forest | 17,519.71 | 28,647.75 | 0.8930 |
| Gradient Boosting | 18,275.50 | 28,741.94 | 0.8923 |
| Decision Tree | 27,088.63 | 41,079.41 | 0.7800 |
| Linear Regression | 20,384.19 | 51,992.05 | 0.6476 |

## Hyperparameter Tuning

GridSearchCV with **5-fold cross-validation** was used to validate model configurations.

### Random Forest

The search evaluated:

- `n_estimators`: 100, 200
- `max_depth`: 10, 20, None
- `min_samples_split`: 2, 5

Best parameters recorded in the notebook:

```text
n_estimators = 100
max_depth = None
min_samples_split = 2
```

Best cross-validation R²:

**0.8426**

Test-set R² after tuning:

**0.8899**

The tuned Random Forest did not improve upon the original Random Forest test-set R² of 0.8930.

### XGBoost

GridSearchCV was also applied to XGBoost using:

- `n_estimators`: 100, 200
- `learning_rate`: 0.05, 0.1
- `max_depth`: 3, 5
- `subsample`: 0.8, 1.0

The notebook documents the selected configuration as:

```text
n_estimators = 200
learning_rate = 0.1
max_depth = 3
subsample = 1.0
```

The notebook reports a cross-validation R² of approximately **0.8608** and a test-set R² of approximately **0.909** for the optimized XGBoost model.

## Final Model

**XGBoost Regressor** was selected as the final model.

It achieved:

- **R² Score:** 0.9090
- **MAE:** 17,054.11
- **RMSE:** 26,417.55

Based on the evaluation metrics documented in the notebook, XGBoost provided the strongest predictive performance among the evaluated models.

## Challenges

Important project challenges and the techniques used to address them are documented separately in:

**`CHALLENGES_REPORT.md`**

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn**

## Conclusion

This project developed a machine learning solution for predicting residential house sale prices.

The workflow covered data understanding, missing-value treatment, categorical encoding, exploratory data analysis, feature preparation, train-test splitting, model training, model evaluation, model comparison, and hyperparameter tuning.

Among the five evaluated regression models, **XGBoost achieved the highest test-set R² Score of approximately 0.909 and the lowest MAE and RMSE**, making it the selected final model for house price prediction.
