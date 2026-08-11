# Model Comparison Report — House Prices Prediction

## Objective

Compare the performance of five machine learning regression models trained to predict residential house sale prices and identify the best-performing model using MAE, RMSE, and R² Score.

## Dataset and Target

- **Rows:** 1,460
- **Original columns:** 81
- **Target:** `SalePrice`
- **Problem type:** Regression

The target is a continuous numerical house sale price, so regression algorithms were used.

## Models Evaluated

| Model | Type |
|---|---|
| Linear Regression | Linear regression baseline |
| Decision Tree Regressor | Single tree-based model |
| Random Forest Regressor | Bagging ensemble |
| Gradient Boosting Regressor | Boosting ensemble |
| XGBoost Regressor | Gradient-boosting ensemble |

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted house prices.

**Lower MAE is better.**

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater penalty to large errors.

**Lower RMSE is better.**

### R² Score

Measures how much of the variation in `SalePrice` is explained by the model.

**Higher R² is better.**

## Results Summary

| Rank | Model | MAE | RMSE | R² Score |
|---:|---|---:|---:|---:|
| 1 | **XGBoost** | **17,054.11** | **26,417.55** | **0.9090** |
| 2 | Random Forest | 17,519.71 | 28,647.75 | 0.8930 |
| 3 | Gradient Boosting | 18,275.50 | 28,741.94 | 0.8923 |
| 4 | Decision Tree | 27,088.63 | 41,079.41 | 0.7800 |
| 5 | Linear Regression | 20,384.19 | 51,992.05 | 0.6476 |

## Model-wise Analysis

### 1. Linear Regression

- **MAE:** 20,384.19
- **RMSE:** 51,992.05
- **R²:** 0.6476

Linear Regression provided the baseline performance. Its R² score was substantially lower than the tree-based ensemble models, indicating that the housing dataset contains complex and non-linear relationships that a simple linear model cannot fully capture.

### 2. Decision Tree Regressor

- **MAE:** 27,088.63
- **RMSE:** 41,079.41
- **R²:** 0.7800

The Decision Tree performed better than Linear Regression in terms of R² and RMSE but had the highest MAE among the evaluated models. A single decision tree can capture non-linear relationships but can be less stable than ensemble methods.

### 3. Random Forest Regressor

- **MAE:** 17,519.71
- **RMSE:** 28,647.75
- **R²:** 0.8930

Random Forest showed strong predictive performance. Combining multiple decision trees improved generalization and substantially reduced error compared with the single Decision Tree.

### 4. Gradient Boosting Regressor

- **MAE:** 18,275.50
- **RMSE:** 28,741.94
- **R²:** 0.8923

Gradient Boosting performed almost as well as Random Forest. Its R² score was approximately 0.892, showing that sequential boosting was effective at learning the complex relationships in the housing data.

### 5. XGBoost Regressor

- **MAE:** 17,054.11
- **RMSE:** 26,417.55
- **R²:** 0.9090

XGBoost achieved the best performance across the evaluated metrics. It produced the highest R² and the lowest MAE and RMSE in the notebook's model comparison.

## Hyperparameter Tuning

### Random Forest

GridSearchCV was performed with 5-fold cross-validation.

Search space:

```text
n_estimators: [100, 200]
max_depth: [10, 20, None]
min_samples_split: [2, 5]
```

Best parameters recorded:

```text
n_estimators = 100
max_depth = None
min_samples_split = 2
```

Best cross-validation R²:

**0.8426**

Test-set R² after tuning:

**0.8899**

The tuned model did not improve on the original Random Forest test-set R² of 0.8930.

### XGBoost

GridSearchCV was also performed with 5-fold cross-validation.

Search space:

```text
n_estimators: [100, 200]
learning_rate: [0.05, 0.1]
max_depth: [3, 5]
subsample: [0.8, 1.0]
```

The notebook documents the selected configuration as:

```text
n_estimators = 200
learning_rate = 0.1
max_depth = 3
subsample = 1.0
```

The notebook reports:

- Cross-validation R²: approximately **0.8608**
- Test-set R²: approximately **0.909**

## Final Recommendation

**XGBoost Regressor** is selected as the final model because it achieved:

- Highest R²: **0.9090**
- Lowest MAE: **17,054.11**
- Lowest RMSE: **26,417.55**

The model is therefore the strongest performer among the five evaluated algorithms in this project.

## Interpretation

The results show that ensemble tree-based methods substantially outperform the Linear Regression baseline. This suggests that house prices are influenced by non-linear relationships and interactions among property characteristics.

Random Forest and Gradient Boosting both achieved R² values close to 0.89, while XGBoost improved the result to approximately 0.909.

## Limitations

- The model performance is based on a single 80:20 train-test split with `random_state=42`.
- The dataset represents historical house sales, so predictions may not directly reflect current market prices.
- The model should be validated on new or external data before real-world deployment.
- Hyperparameter tuning results should be interpreted together with test-set performance rather than cross-validation score alone.
