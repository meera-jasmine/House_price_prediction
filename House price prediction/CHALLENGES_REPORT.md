# Challenges Report — House Prices Prediction

## Project Overview

This report documents the major data preparation, exploratory analysis, modeling, and evaluation challenges encountered while developing the house price prediction model.

---

## 1. Missing Values Across Multiple Features

### Challenge

The original dataset contains missing values in several columns.

The notebook identified missing values in features including:

- `LotFrontage`
- `Alley`
- `MasVnrType`
- `MasVnrArea`
- `BsmtQual`
- `BsmtCond`
- `BsmtExposure`
- `BsmtFinType1`
- `BsmtFinType2`
- `Electrical`
- `FireplaceQu`
- `GarageType`
- `GarageYrBlt`
- `GarageFinish`
- `GarageQual`
- `GarageCond`

### Technique Used

Columns with substantial missing information were removed:

```text
Alley
MasVnrType
FireplaceQu
PoolQC
Fence
MiscFeature
```

Numerical missing values in:

```text
LotFrontage
MasVnrArea
GarageYrBlt
```

were filled using the median.

Remaining categorical missing values were filled using the mode.

### Reasoning

Removing highly incomplete features reduces the amount of unreliable information, while median and mode imputation preserves the remaining observations without unnecessarily deleting rows.

---

## 2. High Dimensionality from Categorical Variables

### Challenge

The dataset contains many categorical features such as:

- `Neighborhood`
- `HouseStyle`
- `Foundation`
- `Exterior1st`
- `KitchenQual`
- `GarageType`
- `SaleCondition`

Machine learning algorithms require numerical representations of these categorical variables.

### Technique Used

One-hot encoding was applied using:

```python
pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)
```

### Reasoning

One-hot encoding converts categorical values into numerical indicator columns so that regression algorithms can process them.

`drop_first=True` removes one category from each encoded group to reduce redundant dummy variables.

---

## 3. Different Feature Scales

### Challenge

Housing features exist on very different numerical scales.

For example:

- `OverallQual` has a relatively small range.
- `LotArea` can contain much larger values.
- `SalePrice` is measured in substantially larger monetary units.

### Technique Used

`StandardScaler` was applied to the training and test features for the Linear Regression model.

```text
Training data → fit_transform()
Testing data → transform()
```

### Reasoning

Scaling puts numerical features on a comparable scale and is particularly useful for algorithms sensitive to feature magnitude.

The tree-based models were trained on the unscaled features because tree-based algorithms do not require standard feature scaling in the same way.

---

## 4. Non-Linear Relationships

### Challenge

The relationship between property characteristics and `SalePrice` is not purely linear.

EDA showed strong relationships between sale price and variables such as:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `Neighborhood`

### Technique Used

Multiple model families were compared, including:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### Reasoning

Tree-based ensemble methods can capture non-linear relationships and interactions between features that a simple linear model may not represent effectively.

This was reflected in the results: Linear Regression achieved an R² of approximately 0.648, while XGBoost achieved approximately 0.909.

---

## 5. Outliers in Housing Data

### Challenge

The notebook used box plots to inspect variables such as:

- `SalePrice`
- `LotArea`
- `GrLivArea`
- `GarageArea`

Housing datasets naturally contain expensive properties and unusually large lots or living areas.

### Technique Used

Outliers were visually inspected rather than automatically deleting every extreme observation.

### Reasoning

An extreme value is not automatically an error. In a real housing market, high-value and large properties can be legitimate observations.

Therefore, indiscriminate outlier removal could remove meaningful market information.

---

## 6. Model Selection

### Challenge

Different algorithms produced substantially different results.

The project therefore required a systematic comparison rather than selecting a model based only on familiarity.

### Technique Used

Five regression models were evaluated using:

- MAE
- RMSE
- R² Score

### Result

XGBoost achieved the strongest performance:

```text
MAE  = 17,054.11
RMSE = 26,417.55
R²   = 0.9090
```

### Reasoning

Using multiple evaluation metrics gives a more complete view of prediction quality than relying on one metric alone.

---

## 7. Hyperparameter Tuning

### Challenge

Default model parameters may not always provide the best generalization performance.

### Technique Used

GridSearchCV with 5-fold cross-validation was applied to Random Forest and XGBoost.

For Random Forest, the search covered:

```text
n_estimators
max_depth
min_samples_split
```

For XGBoost, the search covered:

```text
n_estimators
learning_rate
max_depth
subsample
```

### Result

Random Forest tuning produced a test-set R² of approximately **0.8899**, which was slightly below the original Random Forest result of **0.8930**.

The notebook reports the optimized XGBoost configuration with a test-set R² of approximately **0.909**.

### Reasoning

Hyperparameter tuning was used to systematically search parameter combinations and validate whether model performance could be improved.

---

## 8. Preventing Misinterpretation of Model Performance

### Challenge

A high R² value does not automatically mean that a model is ready for real-world deployment.

### Technique Used

The project evaluated the models using multiple metrics and also inspected:

- Actual vs Predicted plots
- Residual plots
- Model comparison charts

### Reasoning

Different metrics reveal different aspects of model behavior. Residual and prediction plots also provide visual evidence about model errors.

---

## 9. Final Model Selection

### Challenge

Random Forest, Gradient Boosting, and XGBoost all produced strong results, making the final selection dependent on quantitative evidence.

### Decision

XGBoost was selected because it achieved the:

- Highest R² Score
- Lowest MAE
- Lowest RMSE

among the evaluated models.

### Final Result

**XGBoost Regressor — R² = 0.9090**

---

## Lessons Learned

1. Missing values should be analyzed before deciding whether to remove or impute them.
2. Categorical variables need appropriate numerical encoding before model training.
3. Feature scaling is important for scale-sensitive algorithms such as Linear Regression.
4. Tree-based ensembles can capture complex non-linear relationships effectively.
5. Model selection should be based on consistent evaluation metrics.
6. Hyperparameter tuning should be validated using cross-validation and a separate test set.
7. Outliers should be investigated before being removed because extreme observations may represent valid real-world cases.
8. A strong test-set result should still be validated on new data before real-world deployment.
