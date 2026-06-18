# Employee Attrition Analysis

This project analyzes employee attrition using Python, combining exploratory data analysis, statistical methods, feature selection strategies, machine learning models and model interpretability techniques.

The main goal is to identify patterns associated with employee turnover and compare predictive models capable of estimating the probability of attrition.

## Project Overview

Employee attrition is an important organizational issue because it can affect productivity, hiring costs, team stability and long-term workforce planning.

This project follows a structured data science workflow, starting with exploratory and statistical analysis, followed by variable selection, model comparison, cross-validation, hyperparameter tuning, threshold optimization and model interpretation.

The project is still under development. The final model refinement, final interpretation of the results and business-oriented insights are not yet completed.

## Objectives

* Explore employee-related variables associated with attrition.
* Analyze categorical, quantitative and mixed variable relationships.
* Reduce redundancy in variable selection using association, correlation and multicollinearity analysis.
* Build different variable sets for predictive modeling.
* Compare multiple classification algorithms.
* Evaluate model stability using cross-validation.
* Optimize selected model combinations through hyperparameter tuning.
* Test different classification thresholds.
* Interpret model behavior using Odds Ratios, feature importance and SHAP values.

## Dataset

The cleaned dataset contains 1,470 employee records and 43 columns.

The target variable is employee attrition, represented in two formats:

* `Attrition`: categorical target variable with `Yes` and `No` values.
* `AttritionFlag`: binary target variable, where `1` represents attrition and `0` represents no attrition.

The cleaned dataset has no missing values and no duplicated rows.

The target distribution is imbalanced:

* No attrition: 1,233 employees
* Attrition: 237 employees
* Attrition rate: approximately 16.1%

The dataset includes variables related to:

* Demographic information, such as age and gender.
* Job-related characteristics, such as department, job role, job level and business travel.
* Compensation and financial variables, such as monthly income, daily rate, hourly rate and stock options.
* Career and tenure variables, such as years at company, years in current role, years since last promotion and total working years.
* Work conditions, such as overtime, distance from home and work-life balance.
* Satisfaction and performance-related variables, such as job satisfaction, environment satisfaction, job involvement, relationship satisfaction and performance rating.

In addition to the original variables, the cleaned dataset includes engineered categorical variables such as:

* `AgeGroup`
* `TenureGroup`
* `DistanceGroup`
* `CareerStage`
* `RoleStabilityGroup`
* `PromotionDelayGroup`
* `IncomeGroup`
* `EducationLevel`
* `StockOption`
* `JobLevelGroup`
* `SatisfactionLevel`
* `PerformanceRatingLevel`
* `EnvironmentSatisfactionLevel`
* `JobInvolvementLevel`
* `RelationshipSatisfactionLevel`
* `WorkLifeBalanceLevel`

Some columns, such as identifiers or constant variables, may be retained in the cleaned dataset but are not necessarily used as predictive features in the modeling stage.

## Data Source

The dataset used in this project is a public IBM HR Analytics dataset. It is used for educational and analytical purposes to explore employee attrition patterns and build predictive models.

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Methodology

### 1. Data Preparation

The dataset was cleaned and transformed to support both exploratory analysis and predictive modeling.

Several numerical variables were also converted into categorical groups to allow different analytical perspectives and to support categorical, quantitative and mixed modeling strategies.

### 2. Exploratory Data Analysis

Exploratory analysis was conducted to understand the distribution of variables and their relationship with employee attrition.

The analysis considered categorical variables, quantitative variables and engineered groups.

### 3. Association, Correlation and Multicollinearity Analysis

Association, correlation and multicollinearity analyses were used mainly to support variable selection and reduce redundancy between predictors.

For categorical variables, association measures such as Chi-square tests and Cramer's V were used to evaluate relationships with the target variable and identify possible redundancy between categorical predictors.

For quantitative variables, correlation analysis was used to evaluate linear or monotonic relationships between predictors and with the target variable.

For categorical, quantitative and mixed variable sets, multicollinearity analysis was used to identify highly redundant predictors. This helped avoid including multiple variables carrying very similar information in the same model.

Therefore, these analyses were not used only to describe the data, but also to guide the construction of more consistent and less redundant model variable sets.

### 4. Model Variable Sets

Different groups of variables were created and tested to compare how categorical, quantitative and mixed predictors performed in the prediction of employee attrition.

This allowed the project to evaluate whether model performance improved when using specific groups of predictors or broader combinations of variables.

### 5. Machine Learning Models

The project compares several classification models, including:

* Logistic Regression
* Balanced Logistic Regression
* Decision Tree
* Balanced Decision Tree
* Random Forest
* Balanced Random Forest
* Gradient Boosting
* Balanced Gradient Boosting
* XGBoost
* Balanced XGBoost

Balanced versions of the models were tested because the target variable is imbalanced.

### 6. Model Evaluation

The models were evaluated using several classification metrics:

* Accuracy
* Precision
* Recall
* F1-score
* AUC

Since attrition is an imbalanced classification problem, metrics such as Recall, Precision and F1-score are especially important for evaluating model performance beyond overall accuracy.

### 7. Cross-Validation

Cross-validation was used to evaluate model stability and reduce dependence on a single train-test split.

This step helped identify which model and variable-set combinations performed more consistently.

### 8. Hyperparameter Tuning

The best-performing model combinations were selected for hyperparameter tuning.

Randomized search was used to test different parameter combinations and improve model performance.

### 9. Threshold Optimization

Different classification thresholds were tested to evaluate how the decision threshold affects model performance.

This is especially relevant in attrition prediction because the default threshold of 0.50 may not provide the best balance between precision and recall.

### 10. Model Interpretability

Model interpretation was performed using different techniques depending on the model type:

* Odds Ratios for Logistic Regression models.
* Feature importance for tree-based models.
* SHAP values to analyze global and local feature contributions.

These techniques help explain which variables contribute most to attrition prediction and how they influence model outputs.

## Technologies Used

* Python
* pandas
* numpy
* matplotlib
* scipy
* statsmodels
* scikit-learn
* xgboost
* shap
* Jupyter Notebook
* uv
* Ruff

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── clean/
├── notebooks/
│   ├── eda/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_categorical_analysis.ipynb
│   │   ├── 03_quantitative_analysis.ipynb
│   │   └── 04_mixed_analysis.ipynb
│   └── modeling/
│       ├── 01_log_regression.ipynb
│       ├── 02_decision_tree.ipynb
│       ├── 03_random_forest.ipynb
│       ├── 04_gradient_boosting.ipynb
│       ├── 05_xgboost.ipynb
│       └── 06_model_comparison.ipynb
├── src/
│   └── attrition_analysis/
│       ├── associations_test.py
│       ├── data_selection.py
│       ├── logistic_models_utils.py
│       ├── models_utils.py
│       ├── multicollinearity.py
│       └── statistics_utils.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Install the dependencies using `uv`:

```bash
uv sync
```

Run the notebooks using VS Code or Jupyter Notebook.

To check code quality with Ruff:

```bash
uvx ruff check src
```

To check formatting without applying changes:

```bash
uvx ruff format --check src
```

## Current Status

Completed stages:

* Data cleaning
* Exploratory data analysis
* Association analysis
* Correlation analysis
* Multicollinearity analysis
* Variable set construction
* Model comparison
* Cross-validation
* Hyperparameter tuning
* Threshold optimization
* Model interpretation using Odds Ratios, feature importance and SHAP values

In progress:

* Final model refinement
* Final selection of the best predictive approach
* Consolidated interpretation of results
* Business-oriented insight summary
* Final conclusions and recommendations

## Next Steps

The next steps of the project include refining the final model selection, consolidating the main predictive findings and translating the results into business-oriented insights.

The final analysis should explain not only which model performs best, but also which factors appear to be most relevant for employee attrition and how these findings can support decision-making.

## Author

Developed by Nathan Sperinde.
