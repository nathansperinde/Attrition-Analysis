import pandas as pd
import numpy as np

from .data_selection import (
    CATEGORICAL_MODEL_VARS,
    QUANTITATIVE_MODEL_VARS,
    MIXED_MODEL_VARS
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    ParameterGrid
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


categorical_models_dict_c = {
    model_name: {
        "numeric_vars": [],
        "categorical_vars": categorical_vars
    }
    for model_name, categorical_vars in CATEGORICAL_MODEL_VARS.items()
}


quantitative_models_dict_c = {
    model_name: {
        "numeric_vars": numeric_vars,
        "categorical_vars": []
    }
    for model_name, numeric_vars in QUANTITATIVE_MODEL_VARS.items()
}


mixed_models_dict_c = MIXED_MODEL_VARS


estimators_dict = {
    
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=10000,
            solver="liblinear"
        )
    },
    
    "Logistic Regression Balanced": {
        "estimator": LogisticRegression(
            max_iter=10000,
            solver="liblinear",
            class_weight="balanced"
        )
    }
}


param_distributions_dict = {

    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"]
    },

    "Logistic Regression Balanced": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"]
    }
}


def split_train_test_df(
    df,
    target="AttritionFlag",
    test_size=0.30,
    random_state=42
):
    
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )
    
    return df_train.copy(), df_test.copy()


def prepare_model_data(
    df,
    numeric_vars,
    categorical_vars,
    target="AttritionFlag"
):
    
    selected_vars = numeric_vars + categorical_vars
    
    df_model = df[selected_vars + [target]].dropna().copy()
    
    X = pd.get_dummies(
        df_model[selected_vars],
        columns=categorical_vars,
        drop_first=True
    )
    
    X = X.astype(float)
    y = df_model[target]
    
    return X, y


def prepare_train_test_model_data(
    df_train,
    df_test,
    numeric_vars,
    categorical_vars,
    target="AttritionFlag"
):
    
    selected_vars = numeric_vars + categorical_vars
    
    df_train_model = df_train[selected_vars + [target]].dropna().copy()
    df_test_model = df_test[selected_vars + [target]].dropna().copy()
    
    X_train = pd.get_dummies(
        df_train_model[selected_vars],
        columns=categorical_vars,
        drop_first=True
    )
    
    X_test = pd.get_dummies(
        df_test_model[selected_vars],
        columns=categorical_vars,
        drop_first=True
    )
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    y_train = df_train_model[target]
    y_test = df_test_model[target]
    
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline(
    estimator,
    numeric_vars,
    x_columns,
    scale_numeric=True
):
    
    steps = []
    
    numeric_cols_existing = [
        col for col in numeric_vars
        if col in x_columns
    ]
    
    if scale_numeric and len(numeric_cols_existing) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric_scaler", StandardScaler(), numeric_cols_existing)
            ],
            remainder="passthrough"
        )
        
        steps.append(("preprocessor", preprocessor))
    
    steps.append(("classifier", clone(estimator)))
    
    return Pipeline(steps=steps)


def take_rows(data, indices):
    
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    
    return data[indices]


def get_pipeline_feature_names(fitted_pipeline, x_columns):
    
    if "preprocessor" not in fitted_pipeline.named_steps:
        return list(x_columns)
    
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    
    numeric_cols = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric_scaler":
            numeric_cols = list(columns)
    
    remainder_cols = [
        col for col in x_columns
        if col not in numeric_cols
    ]
    
    return numeric_cols + remainder_cols


def get_logistic_interpretation(fitted_pipeline, x_columns):
    
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = get_pipeline_feature_names(fitted_pipeline, x_columns)
    
    coefficients = classifier.coef_[0]
    
    if len(feature_names) != len(coefficients):
        feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
    
    interpretation_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Odds_Ratio": np.exp(coefficients)
        })
        .sort_values(by="Odds_Ratio", ascending=False)
        .reset_index(drop=True)
    )
    
    return interpretation_df


def calculate_classification_metrics(y_true, y_pred, y_prob):
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }


def run_logistic_cross_validation(
    df,
    models_dict,
    estimators_dict=estimators_dict,
    target="AttritionFlag",
    n_splits=10,
    random_state=42,
    scale_numeric=True
):

    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    cv_results = []
    
    for variable_set_name, model_info in models_dict.items():
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        
        X, y = prepare_model_data(
            df=df,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            target=target
        )
        
        for model_name, model_config in estimators_dict.items():
            
            scores = {
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1": [],
                "AUC": []
            }
            
            for train_index, valid_index in cv.split(X, y):
                
                X_train = X.iloc[train_index].copy()
                X_valid = X.iloc[valid_index].copy()
                y_train = y.iloc[train_index]
                y_valid = y.iloc[valid_index]
                
                pipeline = build_logistic_pipeline(
                    estimator=model_config["estimator"],
                    numeric_vars=numeric_vars,
                    x_columns=X.columns,
                    scale_numeric=scale_numeric
                )
                
                pipeline.fit(X_train, y_train)
                
                y_prob = pipeline.predict_proba(X_valid)[:, 1]
                y_pred = (y_prob >= 0.50).astype(int)
                
                scores["Accuracy"].append(
                    accuracy_score(y_valid, y_pred)
                )
                
                scores["Precision"].append(
                    precision_score(y_valid, y_pred, zero_division=0)
                )
                
                scores["Recall"].append(
                    recall_score(y_valid, y_pred, zero_division=0)
                )
                
                scores["F1"].append(
                    f1_score(y_valid, y_pred, zero_division=0)
                )
                
                scores["AUC"].append(
                    roc_auc_score(y_valid, y_prob)
                )
            
            cv_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                
                "Accuracy_Mean": round(np.mean(scores["Accuracy"]), 3),
                "Accuracy_Std": round(np.std(scores["Accuracy"]), 3),
                
                "Precision_Mean": round(np.mean(scores["Precision"]), 3),
                "Precision_Std": round(np.std(scores["Precision"]), 3),
                
                "Recall_Mean": round(np.mean(scores["Recall"]), 3),
                "Recall_Std": round(np.std(scores["Recall"]), 3),
                
                "F1_Mean": round(np.mean(scores["F1"]), 3),
                "F1_Std": round(np.std(scores["F1"]), 3),
                
                "AUC_Mean": round(np.mean(scores["AUC"]), 3),
                "AUC_Std": round(np.std(scores["AUC"]), 3),
                
                "N_Numeric_Variables": len(numeric_vars),
                "N_Categorical_Variables": len(categorical_vars),
                "N_Features_After_Dummies": X.shape[1]
            })
    
    return pd.DataFrame(cv_results)


def run_logistic_gap_analysis(
    df,
    models_dict,
    estimators_dict=estimators_dict,
    target="AttritionFlag",
    n_splits=10,
    random_state=42,
    scale_numeric=True
):
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    gap_results = []
    
    for variable_set_name, model_info in models_dict.items():
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        
        X, y = prepare_model_data(
            df=df,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            target=target
        )
        
        for model_name, model_config in estimators_dict.items():
            
            train_f1_scores = []
            valid_f1_scores = []
            
            train_recall_scores = []
            valid_recall_scores = []
            
            train_auc_scores = []
            valid_auc_scores = []
            
            for train_index, valid_index in cv.split(X, y):
                
                X_train = X.iloc[train_index].copy()
                X_valid = X.iloc[valid_index].copy()
                y_train = y.iloc[train_index]
                y_valid = y.iloc[valid_index]
                
                pipeline = build_logistic_pipeline(
                    estimator=model_config["estimator"],
                    numeric_vars=numeric_vars,
                    x_columns=X.columns,
                    scale_numeric=scale_numeric
                )
                
                pipeline.fit(X_train, y_train)
                
                y_train_prob = pipeline.predict_proba(X_train)[:, 1]
                y_train_pred = (y_train_prob >= 0.50).astype(int)
                
                y_valid_prob = pipeline.predict_proba(X_valid)[:, 1]
                y_valid_pred = (y_valid_prob >= 0.50).astype(int)
                
                train_f1_scores.append(
                    f1_score(y_train, y_train_pred, zero_division=0)
                )
                
                valid_f1_scores.append(
                    f1_score(y_valid, y_valid_pred, zero_division=0)
                )
                
                train_recall_scores.append(
                    recall_score(y_train, y_train_pred, zero_division=0)
                )
                
                valid_recall_scores.append(
                    recall_score(y_valid, y_valid_pred, zero_division=0)
                )
                
                train_auc_scores.append(
                    roc_auc_score(y_train, y_train_prob)
                )
                
                valid_auc_scores.append(
                    roc_auc_score(y_valid, y_valid_prob)
                )
            
            train_f1_mean = np.mean(train_f1_scores)
            valid_f1_mean = np.mean(valid_f1_scores)
            f1_gap = train_f1_mean - valid_f1_mean
            
            train_recall_mean = np.mean(train_recall_scores)
            valid_recall_mean = np.mean(valid_recall_scores)
            recall_gap = train_recall_mean - valid_recall_mean
            
            train_auc_mean = np.mean(train_auc_scores)
            valid_auc_mean = np.mean(valid_auc_scores)
            auc_gap = train_auc_mean - valid_auc_mean
            
            if f1_gap > 0.15:
                gap_diagnosis = "Possible overfitting"
            elif train_f1_mean < 0.40 and valid_f1_mean < 0.40:
                gap_diagnosis = "Possible underfitting"
            elif f1_gap <= 0.05:
                gap_diagnosis = "Stable generalization"
            else:
                gap_diagnosis = "Moderate gap"
            
            gap_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                
                "Train_F1_Mean": round(train_f1_mean, 3),
                "CV_F1_Mean": round(valid_f1_mean, 3),
                "F1_Gap": round(f1_gap, 3),
                
                "Train_Recall_Mean": round(train_recall_mean, 3),
                "CV_Recall_Mean": round(valid_recall_mean, 3),
                "Recall_Gap": round(recall_gap, 3),
                
                "Train_AUC_Mean": round(train_auc_mean, 3),
                "CV_AUC_Mean": round(valid_auc_mean, 3),
                "AUC_Gap": round(auc_gap, 3),
                
                "Gap_Diagnosis": gap_diagnosis,
                
                "N_Numeric_Variables": len(numeric_vars),
                "N_Categorical_Variables": len(categorical_vars),
                "N_Features_After_Dummies": X.shape[1]
            })
    
    return pd.DataFrame(gap_results)


def run_logistic_model_comparison(
    df,
    models_dict,
    estimators_dict=estimators_dict,
    target="AttritionFlag",
    thresholds=None,
    test_size=0.30,
    random_state=42,
    scale_numeric=True
):
    
    if thresholds is None:
        thresholds = np.arange(0.20, 0.651, 0.025)
    
    general_results = []
    threshold_results = []
    confusion_results = {}
    trained_models = {}
    interpretation_results = {}
    
    for variable_set_name, model_info in models_dict.items():
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        
        X, y = prepare_model_data(
            df=df,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            target=target
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        confusion_results[variable_set_name] = {}
        trained_models[variable_set_name] = {}
        interpretation_results[variable_set_name] = {}
        
        for model_name, model_config in estimators_dict.items():
            
            pipeline = build_logistic_pipeline(
                estimator=model_config["estimator"],
                numeric_vars=numeric_vars,
                x_columns=X_train.columns,
                scale_numeric=scale_numeric
            )
            
            pipeline.fit(X_train, y_train)
            
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred_50 = (y_prob >= 0.50).astype(int)
            
            metrics_50 = calculate_classification_metrics(
                y_true=y_test,
                y_pred=y_pred_50,
                y_prob=y_prob
            )
            
            general_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                "Threshold": 0.50,
                
                "Accuracy": round(metrics_50["Accuracy"], 3),
                "Precision": round(metrics_50["Precision"], 3),
                "Recall": round(metrics_50["Recall"], 3),
                "F1-score": round(metrics_50["F1-score"], 3),
                "AUC": round(metrics_50["AUC"], 3),
                
                "N_Numeric_Variables": len(numeric_vars),
                "N_Categorical_Variables": len(categorical_vars),
                "N_Features_After_Dummies": X.shape[1]
            })
            
            for threshold in thresholds:
                
                threshold = round(threshold, 3)
                y_pred_threshold = (y_prob >= threshold).astype(int)
                
                metrics_threshold = calculate_classification_metrics(
                    y_true=y_test,
                    y_pred=y_pred_threshold,
                    y_prob=y_prob
                )
                
                threshold_results.append({
                    "Variable_Set": variable_set_name,
                    "Model": model_name,
                    "Threshold": threshold,
                    
                    "Accuracy": round(metrics_threshold["Accuracy"], 3),
                    "Precision": round(metrics_threshold["Precision"], 3),
                    "Recall": round(metrics_threshold["Recall"], 3),
                    "F1-score": round(metrics_threshold["F1-score"], 3),
                    "AUC": round(metrics_threshold["AUC"], 3)
                })
            
            confusion_results[variable_set_name][model_name] = confusion_matrix(
                y_test,
                y_pred_50
            )
            
            trained_models[variable_set_name][model_name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "model": pipeline,
                "y_prob": y_prob,
                "numeric_vars": numeric_vars,
                "categorical_vars": categorical_vars
            }
            
            interpretation_results[variable_set_name][model_name] = (
                get_logistic_interpretation(
                    fitted_pipeline=pipeline,
                    x_columns=X_train.columns
                )
            )
    
    return (
        pd.DataFrame(general_results),
        pd.DataFrame(threshold_results),
        confusion_results,
        trained_models,
        interpretation_results
    )


def tune_logistic_hyperparameters_top_combinations(
    df,
    models_dict,
    estimators_dict,
    param_distributions_dict,
    top_combinations,
    target="AttritionFlag",
    n_iter=30,
    n_splits=10,
    scoring="f1",
    random_state=42,
    scale_numeric=True
):
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    tuning_results = []
    best_models = {}
    
    for combination in top_combinations:
        
        variable_set_name = combination["Variable_Set"]
        model_name = combination["Model"]
        
        if model_name not in param_distributions_dict:
            print(f"No parameter grid found for {model_name}. Skipping.")
            continue
        
        model_info = models_dict[variable_set_name]
        model_config = estimators_dict[model_name]
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        
        X, y = prepare_model_data(
            df=df,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            target=target
        )
        
        pipeline = build_logistic_pipeline(
            estimator=model_config["estimator"],
            numeric_vars=numeric_vars,
            x_columns=X.columns,
            scale_numeric=scale_numeric
        )
        
        param_grid = param_distributions_dict[model_name]
        n_possible_combinations = len(list(ParameterGrid(param_grid)))
        effective_n_iter = min(n_iter, n_possible_combinations)
        
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=effective_n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=True,
            refit=True
        )
        
        random_search.fit(X, y)
        
        tuning_results.append({
            "Variable_Set": variable_set_name,
            "Model": model_name,
            "Best_Score": round(random_search.best_score_, 3),
            "Scoring": scoring,
            "Best_Params": random_search.best_params_,
            "N_Parameter_Combinations_Tested": effective_n_iter,
            "N_Numeric_Variables": len(numeric_vars),
            "N_Categorical_Variables": len(categorical_vars),
            "N_Features_After_Dummies": X.shape[1]
        })
        
        best_models[(variable_set_name, model_name)] = random_search.best_estimator_
    
    return pd.DataFrame(tuning_results), best_models


def evaluate_thresholds_optimized_logistic_models_cv(
    df,
    models_dict,
    best_models,
    target="AttritionFlag",
    thresholds=None,
    test_size=0.30,
    random_state=42,
    n_splits=10,
    df_test=None,
    threshold_metric="F1-score"
):
    
    if thresholds is None:
        thresholds = np.arange(0.20, 0.751, 0.01)
    
    if df_test is None:
        df_train, df_test = split_train_test_df(
            df=df,
            target=target,
            test_size=test_size,
            random_state=random_state
        )
    else:
        df_train = df.copy()
        df_test = df_test.copy()
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    threshold_cv_results = []
    best_threshold_rows = []
    final_test_results = []
    confusion_results = {}
    fitted_models = {}
    interpretation_results = {}
    
    for (variable_set_name, model_name), best_model in best_models.items():
        
        model_info = models_dict[variable_set_name]
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        
        X_train, X_test, y_train, y_test = prepare_train_test_model_data(
            df_train=df_train,
            df_test=df_test,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            target=target
        )
        
        y_train_prob_cv = np.zeros(len(y_train))
        
        for train_idx, valid_idx in cv.split(X_train, y_train):
            
            X_fold_train = take_rows(X_train, train_idx)
            X_fold_valid = take_rows(X_train, valid_idx)
            y_fold_train = take_rows(y_train, train_idx)
            
            fold_model = clone(best_model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            y_train_prob_cv[valid_idx] = (
                fold_model.predict_proba(X_fold_valid)[:, 1]
            )
        
        auc_cv = roc_auc_score(y_train, y_train_prob_cv)
        
        model_threshold_results = []
        
        for threshold in thresholds:
            
            threshold = round(threshold, 3)
            y_train_pred_cv = (y_train_prob_cv >= threshold).astype(int)
            
            result = {
                "Variable_Set": variable_set_name,
                "Model": model_name,
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_train, y_train_pred_cv),
                "Precision": precision_score(
                    y_train,
                    y_train_pred_cv,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_train,
                    y_train_pred_cv,
                    zero_division=0
                ),
                "F1-score": f1_score(
                    y_train,
                    y_train_pred_cv,
                    zero_division=0
                ),
                "AUC": auc_cv
            }
            
            threshold_cv_results.append(result)
            model_threshold_results.append(result)
        
        model_threshold_df = pd.DataFrame(model_threshold_results)
        
        if threshold_metric not in model_threshold_df.columns:
            raise ValueError(
                f"threshold_metric must be one of: {list(model_threshold_df.columns)}"
            )
        
        best_threshold_row = (
            model_threshold_df
            .sort_values(threshold_metric, ascending=False)
            .iloc[0]
        )
        
        best_threshold = best_threshold_row["Threshold"]
        best_threshold_rows.append(best_threshold_row.to_dict())
        
        final_model = clone(best_model)
        final_model.fit(X_train, y_train)
        
        y_test_prob = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= best_threshold).astype(int)
        
        final_metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_prob=y_test_prob
        )
        
        final_test_results.append({
            "Variable_Set": variable_set_name,
            "Model": model_name,
            "Threshold": best_threshold,
            
            "Accuracy": final_metrics["Accuracy"],
            "Precision": final_metrics["Precision"],
            "Recall": final_metrics["Recall"],
            "F1-score": final_metrics["F1-score"],
            "AUC": final_metrics["AUC"]
        })
        
        confusion_results[(variable_set_name, model_name)] = confusion_matrix(
            y_test,
            y_test_pred
        )
        
        fitted_models[(variable_set_name, model_name)] = {
            "model": final_model,
            "best_threshold": best_threshold,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_test_prob": y_test_prob,
            "y_test_pred": y_test_pred,
            "numeric_vars": numeric_vars,
            "categorical_vars": categorical_vars
        }
        
        interpretation_results[(variable_set_name, model_name)] = (
            get_logistic_interpretation(
                fitted_pipeline=final_model,
                x_columns=X_train.columns
            )
        )
    
    threshold_cv_comparison = pd.DataFrame(threshold_cv_results)
    best_thresholds_cv = pd.DataFrame(best_threshold_rows)
    final_test_results_df = pd.DataFrame(final_test_results)
    
    best_thresholds_cv = best_thresholds_cv.sort_values(
        threshold_metric,
        ascending=False
    )
    
    final_test_results_df = final_test_results_df.sort_values(
        threshold_metric,
        ascending=False
    )
    
    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "AUC"
    ]
    
    for result_df in [
        threshold_cv_comparison,
        best_thresholds_cv,
        final_test_results_df
    ]:
        result_df["Threshold"] = result_df["Threshold"].round(3)
        result_df[metric_cols] = result_df[metric_cols].round(3)
    
    return (
        threshold_cv_comparison,
        best_thresholds_cv,
        final_test_results_df,
        confusion_results,
        fitted_models,
        interpretation_results
    )