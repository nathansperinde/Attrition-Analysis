# Employee Attrition Analysis

This project analyzes employee attrition using Python, combining exploratory data analysis, statistical methods, feature selection strategies, Logistic Regression modeling and model interpretability techniques.

The main goal is to identify patterns associated with employee turnover and evaluate Logistic Regression approaches capable of estimating the probability of attrition.

## Project Overview

Employee attrition is an important organizational issue because it can affect productivity, hiring costs, team stability and long-term workforce planning.

This project follows a structured data science workflow, starting with exploratory and statistical analysis, followed by variable selection, Logistic Regression model comparison, cross-validation, hyperparameter tuning, threshold optimization and model interpretation.

The project is still under development. The final model refinement, final interpretation of the results and business-oriented insights are not yet completed.

## Objectives

* Explore employee-related variables associated with attrition.
* Analyze categorical, quantitative and mixed variable relationships.
* Reduce redundancy in variable selection using association, correlation and multicollinearity analysis.
* Build different variable sets for predictive modeling.
* Compare Logistic Regression approaches using different variable sets.
* Evaluate model stability using cross-validation.
* Optimize selected Logistic Regression combinations through hyperparameter tuning.
* Test different classification thresholds.
* Interpret model behavior using Logistic Regression coefficients and Odds Ratios.

## Data Source

The dataset used in this project is a public IBM HR Analytics dataset. It is used for educational and analytical purposes to explore employee attrition patterns and build predictive models.

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

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

## Variable Scales

Some variables in the IBM HR Employee Attrition dataset are encoded as numerical values, but they represent predefined ordinal categories. For this reason, these variables were mapped into descriptive labels during the data preparation stage.

| Variable | Scale |
|---|---|
| Education | 1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctor |
| EnvironmentSatisfaction | 1 = Low, 2 = Medium, 3 = High, 4 = Very High |
| JobInvolvement | 1 = Low, 2 = Medium, 3 = High, 4 = Very High |
| JobSatisfaction | 1 = Low, 2 = Medium, 3 = High, 4 = Very High |
| PerformanceRating | 1 = Low, 2 = Good, 3 = Excellent, 4 = Outstanding |
| RelationshipSatisfaction | 1 = Low, 2 = Medium, 3 = High, 4 = Very High |
| WorkLifeBalance | 1 = Bad, 2 = Good, 3 = Better, 4 = Best |

These mappings were applied to improve interpretability in the exploratory data analysis, statistical tests, model interpretation and business insights.

In addition to these predefined scales, some continuous variables were grouped into categorical ranges, such as age groups, tenure groups, income groups, distance from home groups and career stage groups.

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

This allowed the project to evaluate whether Logistic Regression performance improved when using specific groups of predictors or broader combinations of variables.

### 5. Logistic Regression Modeling

The modeling stage focuses exclusively on Logistic Regression because it is an interpretable algorithm and aligns well with the objective of explaining employee attrition risk factors.

The project compares Logistic Regression configurations such as:

* Standard Logistic Regression.
* Balanced Logistic Regression using class weighting.
* Logistic Regression models trained with different variable sets.

Balanced Logistic Regression was tested because the target variable is imbalanced. In an attrition analysis context, it is especially important to reduce false negatives, since predicting that an employee has no attrition risk when they actually leave may be more costly than incorrectly flagging an employee as at risk.

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

This step helped identify which Logistic Regression and variable-set combinations performed more consistently.

### 8. Hyperparameter Tuning

The best-performing Logistic Regression combinations were selected for hyperparameter tuning.

Randomized search was used to test different parameter combinations and improve model performance.

### 9. Threshold Optimization

Different classification thresholds were tested to evaluate how the decision threshold affects model performance.

This is especially relevant in attrition prediction because the default threshold of 0.50 may not provide the best balance between precision and recall.

### 10. Model Interpretability

Model interpretation was performed using Logistic Regression coefficients and Odds Ratios.

These techniques help explain which variables contribute most to attrition prediction and whether each variable increases or decreases the estimated odds of employee attrition.

## Technologies Used

* Python
* pandas
* numpy
* matplotlib
* scipy
* statsmodels
* scikit-learn
* Jupyter Notebook
* uv
* Ruff

## Project Structure

```text
.
├── data/
│   ├── clean/
│   │   └── Employee-Attrition_Clean.csv
│   └── raw/
│       └── HR-Employee-Attrition.csv
├── notebooks/
│   ├── eda/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_categorical_analysis.ipynb
│   │   ├── 03_quantitative_analysis.ipynb
│   │   └── 04_mixed_analysis.ipynb
│   └── modeling/
│       ├── 01_log_regression.ipynb
│       └── 02_models_comparison.ipynb
├── src/
│   └── attrition_analysis/
│       ├── __init__.py
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

## Current Status

Completed stages:

* Data cleaning
* Exploratory data analysis
* Association analysis
* Correlation analysis
* Multicollinearity analysis
* Variable set construction
* Logistic Regression model comparison
* Cross-validation
* Hyperparameter tuning
* Threshold optimization
* Model interpretation using coefficients and Odds Ratios

In progress:

* Final model refinement
* Final selection of the best Logistic Regression approach
* Consolidated interpretation of results
* Business-oriented insight summary
* Final conclusions and recommendations

## Next Steps

The next steps of the project include refining the final Logistic Regression model selection, consolidating the main predictive findings and translating the results into business-oriented insights.

The final analysis should explain not only which Logistic Regression configuration performs best, but also which factors appear to be most relevant for employee attrition and how these findings can support decision-making.

## Author

Developed by Nathan Sperinde.
