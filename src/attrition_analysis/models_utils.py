import pandas as pd
import numpy as np
from .data_selection import CATEGORICAL_MODEL_VARS, QUANTITATIVE_MODEL_VARS, MIXED_MODEL_VARS

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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


def prepare_model_data(df, numeric_vars, categorical_vars, target="AttritionFlag"):
    
    selected_vars = numeric_vars + categorical_vars
    df_model = df[selected_vars + [target]].dropna()
    
    X = pd.get_dummies(
        df_model[selected_vars],
        columns=categorical_vars,
        drop_first=True
    )
    
    X = X.astype(float)
    y = df_model[target]
    
    return X, y


def run_cross_validation_mixed(
    df,
    models_dict,
    estimators_dict,
    target="AttritionFlag",
    n_splits=10,
    random_state=42,
    scale_numeric_for=None
):
    
    if scale_numeric_for is None:
        scale_numeric_for = ["Logistic Regression", "Logistic Regression Balanced"]
    
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
        
        numeric_cols_existing = [
            col for col in numeric_vars
            if col in X.columns
        ]
        
        for model_name, model_config in estimators_dict.items():
            scores = {
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1": [],
                "AUC": []
            }
            
            for train_index, test_index in cv.split(X, y):
                X_train = X.iloc[train_index].copy()
                X_test = X.iloc[test_index].copy()
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]
                
                estimator = clone(model_config["estimator"])
                balance_method = model_config.get("balance_method")
                
                if model_name in scale_numeric_for and len(numeric_vars) > 0:
                    scaler = StandardScaler()
                    
                    X_train[numeric_cols_existing] = scaler.fit_transform(
                        X_train[numeric_cols_existing]
                    )
                    
                    X_test[numeric_cols_existing] = scaler.transform(
                        X_test[numeric_cols_existing]
                    )
                
                if balance_method == "sample_weight":
                    sample_weight = compute_sample_weight(
                        class_weight="balanced",
                        y=y_train
                    )
                    
                    estimator.fit(
                        X_train,
                        y_train,
                        sample_weight=sample_weight
                    )
                else:
                    estimator.fit(X_train, y_train)
                
                y_prob = estimator.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.50).astype(int)
                
                scores["Accuracy"].append(
                    accuracy_score(y_test, y_pred)
                )
                
                scores["Precision"].append(
                    precision_score(y_test, y_pred, zero_division=0)
                )
                
                scores["Recall"].append(
                    recall_score(y_test, y_pred, zero_division=0)
                )
                
                scores["F1"].append(
                    f1_score(y_test, y_pred, zero_division=0)
                )
                
                scores["AUC"].append(
                    roc_auc_score(y_test, y_prob)
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


def run_cv_gap_analysis_mixed(
    df,
    models_dict,
    estimators_dict,
    target="AttritionFlag",
    n_splits=10,
    random_state=42,
    scale_numeric_for=None
):
    
    if scale_numeric_for is None:
        scale_numeric_for = ["Logistic Regression", "Logistic Regression Balanced"]
    
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
        
        numeric_cols_existing = [
            col for col in numeric_vars
            if col in X.columns
        ]
        
        for model_name, model_config in estimators_dict.items():
            
            train_f1_scores = []
            cv_f1_scores = []
            
            train_recall_scores = []
            cv_recall_scores = []
            
            train_auc_scores = []
            cv_auc_scores = []
            
            for train_index, test_index in cv.split(X, y):
                X_train = X.iloc[train_index].copy()
                X_test = X.iloc[test_index].copy()
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]
                
                estimator = clone(model_config["estimator"])
                balance_method = model_config.get("balance_method")
                
                if model_name in scale_numeric_for and len(numeric_cols_existing) > 0:
                    scaler = StandardScaler()
                    
                    X_train[numeric_cols_existing] = scaler.fit_transform(
                        X_train[numeric_cols_existing]
                    )
                    
                    X_test[numeric_cols_existing] = scaler.transform(
                        X_test[numeric_cols_existing]
                    )
                
                if balance_method == "sample_weight":
                    sample_weight = compute_sample_weight(
                        class_weight="balanced",
                        y=y_train
                    )
                    
                    estimator.fit(
                        X_train,
                        y_train,
                        sample_weight=sample_weight
                    )
                else:
                    estimator.fit(X_train, y_train)
                
                # Train predictions
                y_train_prob = estimator.predict_proba(X_train)[:, 1]
                y_train_pred = (y_train_prob >= 0.50).astype(int)
                
                train_f1_scores.append(
                    f1_score(y_train, y_train_pred, zero_division=0)
                )
                
                train_recall_scores.append(
                    recall_score(y_train, y_train_pred, zero_division=0)
                )
                
                train_auc_scores.append(
                    roc_auc_score(y_train, y_train_prob)
                )
                
                # Validation fold predictions
                y_test_prob = estimator.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_prob >= 0.50).astype(int)
                
                cv_f1_scores.append(
                    f1_score(y_test, y_test_pred, zero_division=0)
                )
                
                cv_recall_scores.append(
                    recall_score(y_test, y_test_pred, zero_division=0)
                )
                
                cv_auc_scores.append(
                    roc_auc_score(y_test, y_test_prob)
                )
            
            train_f1_mean = np.mean(train_f1_scores)
            cv_f1_mean = np.mean(cv_f1_scores)
            f1_gap = train_f1_mean - cv_f1_mean
            
            train_recall_mean = np.mean(train_recall_scores)
            cv_recall_mean = np.mean(cv_recall_scores)
            recall_gap = train_recall_mean - cv_recall_mean
            
            train_auc_mean = np.mean(train_auc_scores)
            cv_auc_mean = np.mean(cv_auc_scores)
            auc_gap = train_auc_mean - cv_auc_mean
            
            if f1_gap > 0.15:
                gap_diagnosis = "Possible overfitting"
            elif train_f1_mean < 0.40 and cv_f1_mean < 0.40:
                gap_diagnosis = "Possible underfitting"
            elif f1_gap <= 0.05:
                gap_diagnosis = "Stable generalization"
            else:
                gap_diagnosis = "Moderate gap"
            
            gap_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                
                "Train_F1_Mean": round(train_f1_mean, 3),
                "CV_F1_Mean": round(cv_f1_mean, 3),
                "F1_Gap": round(f1_gap, 3),
                
                "Train_Recall_Mean": round(train_recall_mean, 3),
                "CV_Recall_Mean": round(cv_recall_mean, 3),
                "Recall_Gap": round(recall_gap, 3),
                
                "Train_AUC_Mean": round(train_auc_mean, 3),
                "CV_AUC_Mean": round(cv_auc_mean, 3),
                "AUC_Gap": round(auc_gap, 3),
                
                "Gap_Diagnosis": gap_diagnosis,
                
                "N_Numeric_Variables": len(numeric_vars),
                "N_Categorical_Variables": len(categorical_vars),
                "N_Features_After_Dummies": X.shape[1]
            })
    
    return pd.DataFrame(gap_results)


def run_model_comparison_mixed(
    df,
    models_dict,
    estimators_dict,
    target="AttritionFlag",
    thresholds=None,
    test_size=0.30,
    random_state=42,
    scale_numeric_for=None
):
    
    if thresholds is None:
        thresholds = np.arange(0.20, 0.651, 0.025)
    
    if scale_numeric_for is None:
        scale_numeric_for = ["Logistic Regression", "Logistic Regression Balanced"]
    
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
        
        numeric_cols_existing = [
            col for col in numeric_vars
            if col in X_train.columns
        ]
        
        confusion_results[variable_set_name] = {}
        trained_models[variable_set_name] = {}
        interpretation_results[variable_set_name] = {}
        
        for model_name, model_config in estimators_dict.items():
            estimator = model_config["estimator"]
            balance_method = model_config.get("balance_method")
            
            X_train_model = X_train.copy()
            X_test_model = X_test.copy()
            scaler = None
            
            if model_name in scale_numeric_for and len(numeric_vars) > 0:
                scaler = StandardScaler()
                
                X_train_model[numeric_cols_existing] = scaler.fit_transform(
                    X_train_model[numeric_cols_existing]
                )
                
                X_test_model[numeric_cols_existing] = scaler.transform(
                    X_test_model[numeric_cols_existing]
                )
            
            if balance_method == "sample_weight":
                sample_weight = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train
                )
                
                estimator.fit(
                    X_train_model,
                    y_train,
                    sample_weight=sample_weight
                )
            else:
                estimator.fit(X_train_model, y_train)
            
            y_prob = estimator.predict_proba(X_test_model)[:, 1]
            y_pred_50 = (y_prob >= 0.50).astype(int)
            auc = roc_auc_score(y_test, y_prob)
            
            general_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                "Threshold": 0.50,
                "Accuracy": round(accuracy_score(y_test, y_pred_50), 3),
                "Precision": round(precision_score(y_test, y_pred_50, zero_division=0), 3),
                "Recall": round(recall_score(y_test, y_pred_50), 3),
                "F1-score": round(f1_score(y_test, y_pred_50), 3),
                "AUC": round(auc, 3),
                "N_Numeric_Variables": len(numeric_vars),
                "N_Categorical_Variables": len(categorical_vars),
                "N_Features_After_Dummies": X.shape[1]
            })
            
            for threshold in thresholds:
                y_pred_threshold = (y_prob >= threshold).astype(int)
                
                threshold_results.append({
                    "Variable_Set": variable_set_name,
                    "Model": model_name,
                    "Threshold": threshold,
                    "Accuracy": round(accuracy_score(y_test, y_pred_threshold), 3),
                    "Precision": round(precision_score(y_test, y_pred_threshold, zero_division=0), 3),
                    "Recall": round(recall_score(y_test, y_pred_threshold), 3),
                    "F1-score": round(f1_score(y_test, y_pred_threshold), 3),
                    "AUC": round(auc, 3)
                })
            
            confusion_results[variable_set_name][model_name] = confusion_matrix(
                y_test,
                y_pred_50
            )
            
            trained_models[variable_set_name][model_name] = {
                "X_train": X_train_model,
                "X_test": X_test_model,
                "y_train": y_train,
                "y_test": y_test,
                "model": estimator,
                "y_prob": y_prob,
                "scaler": scaler,
                "numeric_vars": numeric_vars,
                "categorical_vars": categorical_vars
            }
            
            if hasattr(estimator, "coef_"):
                interpretation_results[variable_set_name][model_name] = (
                    pd.DataFrame({
                        "Feature": X_train_model.columns,
                        "Coefficient": estimator.coef_[0],
                        "Odds_Ratio": np.exp(estimator.coef_[0])
                    })
                    .sort_values(by="Odds_Ratio", ascending=False)
                )
            
            elif hasattr(estimator, "feature_importances_"):
                interpretation_results[variable_set_name][model_name] = (
                    pd.DataFrame({
                        "Feature": X_train_model.columns,
                        "Importance": estimator.feature_importances_
                    })
                    .sort_values(by="Importance", ascending=False)
                )
            
            else:
                interpretation_results[variable_set_name][model_name] = None
    
    return (
        pd.DataFrame(general_results),
        pd.DataFrame(threshold_results),
        confusion_results,
        trained_models,
        interpretation_results
    )


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
    },
    
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=20,
            random_state=42
        )
    },
    
    "Decision Tree Balanced": {
        "estimator": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42
        )
    },
    
    "Random Forest": {
        "estimator": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
    },
    
    "Random Forest Balanced": {
        "estimator": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    },
    
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
    },
    
    "Gradient Boosting Balanced": {
        "estimator": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        ),
        "balance_method": "sample_weight"
    },
    
    "XGBoost": {
        "estimator": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    },
    
    "XGBoost Balanced": {
        "estimator": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        "balance_method": "sample_weight"
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
    },

    "Decision Tree": {
        "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
        "classifier__min_samples_leaf": [2, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10, 20, 30],
        "classifier__criterion": ["gini", "entropy"]
    },

    "Decision Tree Balanced": {
        "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
        "classifier__min_samples_leaf": [2, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10, 20, 30],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__class_weight": ["balanced"]
    },

    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [4, 6, 8, 10, 12, None],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__max_features": ["sqrt", "log2"]
    },

    "Random Forest Balanced": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [4, 6, 8, 10, 12, None],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": ["balanced"]
    },

    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0]
    },

    "Gradient Boosting Balanced": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4, 5],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0]
    },

    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__reg_alpha": [0, 0.01, 0.1, 1],
        "classifier__reg_lambda": [0.1, 1, 5, 10]
    },

    "XGBoost Balanced": {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__reg_alpha": [0, 0.01, 0.1, 1],
        "classifier__reg_lambda": [0.1, 1, 5, 10],
        "classifier__scale_pos_weight": [1, 2, 3, 4, 5, 6]
    }
}


def tune_hyperparameters_top_combinations(
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
    scale_numeric_for=None
):

    if scale_numeric_for is None:
        scale_numeric_for = [
            "Logistic Regression",
            "Logistic Regression Balanced"
        ]

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

        estimator = clone(model_config["estimator"])
        balance_method = model_config.get("balance_method")

        steps = []

        if model_name in scale_numeric_for:
            steps.append(("scaler", StandardScaler()))

        steps.append(("classifier", estimator))

        random_search = RandomizedSearchCV(
            estimator=Pipeline(steps=steps),
            param_distributions=param_distributions_dict[model_name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=True
        )

        if balance_method == "sample_weight":
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y
            )

            random_search.fit(
                X,
                y,
                classifier__sample_weight=sample_weight
            )

        else:
            random_search.fit(X, y)

        tuning_results.append({
            "Variable_Set": variable_set_name,
            "Model": model_name,
            "Best_Score": round(random_search.best_score_, 3),
            "Scoring": scoring,
            "Best_Params": random_search.best_params_,
            "N_Numeric_Variables": len(numeric_vars),
            "N_Categorical_Variables": len(categorical_vars),
            "N_Features_After_Dummies": X.shape[1]
        })

        best_models[(variable_set_name, model_name)] = random_search.best_estimator_

    return pd.DataFrame(tuning_results), best_models


def evaluate_thresholds_optimized_models(
    df,
    models_dict,
    best_models,
    estimators_dict,
    target="AttritionFlag",
    thresholds=None,
    test_size=0.30,
    random_state=42
):
    
    if thresholds is None:
        thresholds = np.arange(0.20, 0.751, 0.01)
    
    threshold_results = []
    confusion_results = {}
    fitted_models = {}
    
    for (variable_set_name, model_name), best_model in best_models.items():
        model_info = models_dict[variable_set_name]
        model_config = estimators_dict[model_name]
        
        numeric_vars = model_info["numeric_vars"]
        categorical_vars = model_info["categorical_vars"]
        balance_method = model_config.get("balance_method")
        
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
        
        model = clone(best_model)
        
        if balance_method == "sample_weight":
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
            
            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=sample_weight
            )
        else:
            model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        confusion_results[(variable_set_name, model_name)] = {}
        
        for threshold in thresholds:
            threshold = round(threshold, 3)
            y_pred = (y_prob >= threshold).astype(int)
            
            threshold_results.append({
                "Variable_Set": variable_set_name,
                "Model": model_name,
                "Threshold": threshold,
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
                "F1-score": round(f1_score(y_test, y_pred, zero_division=0), 3),
                "AUC": round(auc, 3)
            })
            
            confusion_results[(variable_set_name, model_name)][threshold] = confusion_matrix(
                y_test,
                y_pred
            )
        
        fitted_models[(variable_set_name, model_name)] = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_prob": y_prob
        }
    
    threshold_comparison = pd.DataFrame(threshold_results)
    
    best_thresholds = (
        threshold_comparison
        .sort_values("F1-score", ascending=False)
        .groupby(["Variable_Set", "Model"], as_index=False)
        .first()
        .sort_values("F1-score", ascending=False)
    )
    
    return (
        threshold_comparison,
        best_thresholds,
        confusion_results,
        fitted_models
    )