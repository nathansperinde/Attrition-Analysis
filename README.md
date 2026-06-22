# Employee Attrition Analysis

This project analyzes employee attrition using Python, combining exploratory data analysis, statistical methods, feature selection strategies, Logistic Regression modeling, threshold optimization and model interpretability techniques.

The main goal is to identify patterns associated with employee turnover and evaluate interpretable Logistic Regression models capable of supporting employee attrition analysis from both predictive and business-oriented perspectives.

## Project Overview

Employee attrition is an important organizational issue because it can affect productivity, hiring costs, team stability and long-term workforce planning.

This project follows a structured data science workflow, starting with data preparation and exploratory analysis, followed by statistical tests, variable selection, Logistic Regression modeling, cross-validation, hyperparameter tuning, threshold optimization, model interpretation and business-oriented conclusions.

The final analysis focuses not only on predictive performance, but also on interpretability. Since employee attrition is a sensitive human resources topic, the models are treated as decision-support tools rather than automatic decision-making systems.

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
* Translate model results into business-oriented insights and practical recommendations.

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

### 11. Executive Summary and Business Interpretation

The final stage consolidated the main model results, limitations, business implications and final conclusions.

The analysis was translated into practical insights related to workload, business travel, job involvement, satisfaction, compensation, benefits, career level, commuting distance and work-life balance.

## Final Results

The results suggest that employee attrition is associated with multiple organizational, job-related and individual-context dimensions.

The main factors identified in the analysis include:

* workload and overtime;
* frequent business travel;
* low job involvement;
* low job satisfaction and low environment satisfaction;
* limited access to benefits, such as stock options;
* lower income;
* lower job level;
* poorer work-life balance;
* greater distance from home.

These results should be interpreted as associations identified in this dataset, not as direct causal effects.

The project identified two practical modeling strategies:

### Balanced and Interpretable Strategy

The first strategy uses `Modelo 2 — Nível Hierárquico e Benefícios` with balanced Logistic Regression and a classification threshold of `0.71`.

This strategy is recommended when the objective is to obtain a more stable, explainable and balanced decision-support tool.

### Recall-Oriented Strategy

The second strategy uses `Modelo 3 — Faixa Salarial` with balanced Logistic Regression and a classification threshold of `0.58`.

This strategy is recommended when the priority is to reduce false negatives and identify more employees who may be at risk of leaving, even if this increases the number of false positives.

This approach may be useful in contexts where failing to identify a potential leaver is considered more costly than incorrectly flagging an employee as at risk.

## Business Implications

From a business perspective, the model may support retention strategies by helping organizations identify groups of employees who appear to be more vulnerable to attrition.

The results may be useful for guiding:

* workload monitoring;
* employee engagement initiatives;
* satisfaction tracking;
* benefits and compensation review;
* career development;
* internal mobility;
* flexible work policies;
* work-life balance initiatives.

The model should be used as a decision-support tool, not as an automatic decision-making system. Any practical application should include human evaluation, contextual analysis and ethical consideration.

## Limitations

This analysis has some limitations that should be considered when interpreting the results.

First, the dataset is imbalanced, which makes the identification of attrition cases more challenging.

Second, several variables are correlated with each other, especially variables related to job level, income, career stage, tenure and role stability. For this reason, some predictors should not be interpreted in isolation.

Third, the results should be interpreted as associations, not causal effects. The model can identify patterns related to attrition, but it cannot prove that a specific factor directly causes employees to leave.

Fourth, some variables are specific to the structure of this dataset. For example, business travel, stock options and distance from home should be interpreted as contextual indicators rather than universal predictors of turnover.

Finally, turnover is a complex and context-dependent phenomenon. The relevance of each factor may vary according to industry, role, labour market conditions, organizational culture and individual circumstances.

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
│   ├── modeling/
│   │   ├── 01_log_regression.ipynb
│   │   └── 02_models_comparison.ipynb
│   └── results/
│       └── 01_executive_summary.ipynb
├── outputs/
│   └── tables/...
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

A suggested execution order is:

```text
notebooks/eda/01_data_preparation.ipynb
notebooks/eda/02_categorical_analysis.ipynb
notebooks/eda/03_quantitative_analysis.ipynb
notebooks/eda/04_mixed_analysis.ipynb
notebooks/modeling/01_log_regression.ipynb
notebooks/modeling/02_models_comparison.ipynb
notebooks/results/01_executive_summary.ipynb
```

## Project Status

The project is completed.

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
* Final model strategy selection
* Business implications
* Limitations
* Final conclusions

## Final Conclusion

This project suggests that Logistic Regression can provide a useful balance between predictive performance and interpretability in employee attrition analysis.

The results indicate that attrition should be understood as a multidimensional phenomenon, associated with workload, job attitudes, satisfaction, benefits, compensation, career level, commuting distance and work-life balance.

Although a broader multidimensional model achieved strong predictive performance, it was kept only as a control model because it was more complex and less suitable for business interpretation.

Overall, the project supports the use of Logistic Regression as a practical and interpretable approach for employee attrition analysis, especially when the goal is to support organizational decision-making rather than automate decisions about employees.

## Possible Future Improvements

Future work could include:

* Testing the model on more recent or organization-specific HR datasets.
* Comparing the results with other interpretable machine learning methods.
* Including external variables related to labour market conditions or organizational context.
* Developing a dashboard to communicate attrition risk patterns to decision-makers.

## Author

Developed by Nathan Sperinde.