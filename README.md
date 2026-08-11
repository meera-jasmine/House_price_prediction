# House Prices Prediction using Machine Learning

Predicting the selling price of a house using property characteristics such as overall quality, living area, lot area, garage capacity, basement area, neighborhood, and other housing features to support accurate property price estimation.

## Problem Statement

The objective of this project is to develop a machine learning model that can predict the selling price of a house based on its various characteristics.

The project focuses on:

1. Understanding and analyzing the factors that influence house prices.
2. Performing data cleaning and preprocessing on the housing dataset.
3. Conducting exploratory data analysis to identify important patterns and relationships.
4. Building and comparing multiple machine learning regression models.
5. Optimizing selected models using hyperparameter tuning.
6. Identifying the best-performing model for house price prediction.

## Project Type

**Regression** — The target variable `SalePrice` is a continuous numerical value representing the selling price of a house.

## Contents

* **`House Prices Prediction.ipynb`** — Complete analysis notebook containing data loading, data understanding, data cleaning, exploratory data analysis, feature encoding, train-test splitting, feature scaling, regression model training, model evaluation, model comparison, and hyperparameter tuning.
* **Model Comparison** — Comparison of Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost using MAE, MSE, RMSE, and R² Score.
* **Hyperparameter Tuning** — GridSearchCV-based optimization performed for Random Forest and XGBoost models.
* **EDA and Visualizations** — Analysis of relationships between house characteristics and SalePrice using box plots, scatter plots, histograms, correlation analysis, and residual plots.

## Dataset

The dataset contains **1,460 house records with 81 columns** representing different characteristics of residential properties.

The features include information related to:

* Property size and area
* Overall house quality
* Number of rooms and facilities
* Garage capacity
* Basement area
* Neighborhood
* Lot characteristics
* Construction and sale information

### Target Variable

**`SalePrice`** — The final selling price of the house and the continuous target variable used for regression.

## Data Preprocessing

The following preprocessing techniques were performed:

* Inspected dataset shape, columns, data types, and descriptive statistics.
* Identified missing values across the dataset.
* Removed columns containing a very high percentage of missing values, including `Alley`, `PoolQC`, `Fence`, `MiscFeature`, `FireplaceQu`, and `MasVnrType`.
* Filled missing numerical values in `LotFrontage`, `MasVnrArea`, and `GarageYrBlt` using median imputation.
* Handled remaining categorical missing values.
* Converted categorical variables into numerical representations using one-hot encoding.
* Separated independent features from the target variable `SalePrice`.
* Split the dataset into training and testing sets using an 80:20 ratio.
* Applied standard scaling for the Linear Regression model.

## Exploratory Data Analysis

Several visualizations were performed to understand the relationship between housing features and sale prices.

### Key EDA Analyses

* Neighborhood vs Sale Price
* Overall Quality vs Sale Price
* Ground Living Area vs Sale Price
* Garage Capacity vs Sale Price
* Correlation of numerical variables with SalePrice
* Distribution of important numerical features
* Outlier analysis
* SalePrice distribution
* Actual vs Predicted price plots
* Residual analysis

## Key Findings

* **Overall Quality (`OverallQual`)** showed the strongest positive relationship with `SalePrice`, with a correlation of approximately **0.79**.
* **Ground Living Area (`GrLivArea`)** showed a strong positive relationship with house prices, indicating that larger living areas generally correspond to higher selling prices.
* **Neighborhood** had a significant influence on house prices, with premium neighborhoods showing higher median sale prices.
* Houses with larger **garage capacity (`GarageCars`)** generally had higher median selling prices.
* The `SalePrice` distribution was right-skewed, indicating that most houses were sold within the lower-to-middle price range while a smaller number of properties had substantially higher prices.
* Outliers were identified in variables such as `SalePrice` and `LotArea`. These observations were retained because they can represent genuine high-value properties rather than data-entry errors.

## Machine Learning Models

Five regression algorithms were implemented and compared:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Model Performance Summary

| Model                       |  R² Score | Performance                     |
| --------------------------- | --------: | ------------------------------- |
| Linear Regression           |     0.648 | Baseline model                  |
| Decision Tree Regressor     |      0.78 | Improved over Linear Regression |
| Random Forest Regressor     |     0.893 | Strong performance              |
| Gradient Boosting Regressor |     ~0.89 | Strong performance              |
| **XGBoost Regressor**       | **0.909** | **Best performing model**       |

## Hyperparameter Tuning

GridSearchCV with **5-fold cross-validation** was used to optimize the Random Forest and XGBoost models.

### Random Forest

The tuning process evaluated combinations of:

* `n_estimators`
* `max_depth`
* `min_samples_split`

The optimized Random Forest achieved an R² Score of approximately **0.89**.

### XGBoost

The tuning process evaluated:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `subsample`

The optimized XGBoost model achieved an R² Score of approximately **0.909** on the test dataset.

## Final Model

**XGBoost Regressor** was selected as the final model because it achieved the highest R² Score among the evaluated models.

### Final Performance

* **R² Score:** ~0.909
* **MAE:** ~17,054
* **RMSE:** ~26,418

The model demonstrated the strongest overall predictive performance among the models tested.

## Challenges Faced

### 1. Handling Missing Values

The dataset contained missing values across multiple columns. Columns with extremely high missing-value percentages were removed, while numerical missing values in relevant columns were handled using median imputation.

### 2. Categorical Variables

The dataset contained several categorical variables that could not be directly used by machine learning algorithms. One-hot encoding was therefore applied to convert categorical features into numerical form.

### 3. Outliers

Several extreme observations were identified, particularly in `SalePrice` and `LotArea`. These were retained because they can represent legitimate premium properties.

### 4. Model Selection

Multiple regression algorithms were evaluated because different models can capture different types of relationships between housing characteristics and price.

### 5. Hyperparameter Optimization

GridSearchCV with 5-fold cross-validation was used to find better parameter combinations for Random Forest and XGBoost.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · SciPy**

## Conclusion

This project developed a machine learning solution for predicting house selling prices using property characteristics.

The workflow included data understanding, missing-value handling, categorical encoding, exploratory data analysis, feature preparation, model training, model evaluation, model comparison, and hyperparameter tuning.

Among the five regression models evaluated, **XGBoost Regressor achieved the best performance with an R² Score of approximately 0.909**, making it the final selected model for house price prediction.

The analysis also showed that factors such as **Overall Quality, Ground Living Area, Garage Capacity, Neighborhood, and Basement Area** have important relationships with house prices.
