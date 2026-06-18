import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


def logistic_reg_model(df, selected_vars, target="AttritionFlag", categorical=True, show_summary=True):

    model_df = df[[target] + selected_vars].dropna().copy()

    if categorical:
        formula_terms = [f"C({col})" for col in selected_vars]
    else:
        scaler = StandardScaler()
        model_df[selected_vars] = scaler.fit_transform(model_df[selected_vars])
        formula_terms = selected_vars

    formula = target + " ~ " + " + ".join(formula_terms)

    model = smf.logit(formula=formula, data=model_df).fit(
        method="lbfgs",
        maxiter=1000,
        disp=False
    )

    if show_summary:
        print(model.summary())

    params = model.params
    conf = model.conf_int()

    results = pd.DataFrame({
        "Coefficient": params,
        "Odds_Ratio": np.exp(params),
        "CI_Lower": np.exp(conf[0]),
        "CI_Upper": np.exp(conf[1]),
        "p_value": model.pvalues
    })

    return (
        model,
        results
        .round(4)
        .sort_values(by="Odds_Ratio", ascending=False)
    )


def run_logistic_models(df, models_vars_dict, target="AttritionFlag", categorical=True, show_summary=False):

    fitted_models = {}
    model_results = {}

    for model_name, selected_vars in models_vars_dict.items():
        fitted_models[model_name], model_results[model_name] = logistic_reg_model(
            df=df,
            selected_vars=selected_vars,
            target=target,
            categorical=categorical,
            show_summary=show_summary
        )

    return fitted_models, model_results


def compare_logistic_models(models_dict):
    comparison = []

    for model_name, model in models_dict.items():
        comparison.append({
            "Model": model_name,
            "N_Observations": int(model.nobs),
            "Df_Model": int(model.df_model),
            "Log_Likelihood": model.llf,
            "LL_Null": model.llnull,
            "LLR_p_value": model.llr_pvalue,
            "AIC": model.aic,
            "BIC": model.bic,
            "Pseudo_R2": model.prsquared
        })

    return (
        pd.DataFrame(comparison)
        .sort_values(by="AIC", ascending=True)
        .reset_index(drop=True)
    )


def logistic_reg_model_mixed(df, numeric_vars, categorical_vars, target="AttritionFlag", show_summary=True):
    numeric_vars = [
        col for col in numeric_vars
        if col != target
    ]

    categorical_vars = [
        col for col in categorical_vars
        if col != target
    ]

    selected_vars = numeric_vars + categorical_vars
    model_df = df[[target] + selected_vars].dropna().copy()

    if len(numeric_vars) > 0:
        scaler = StandardScaler()
        model_df[numeric_vars] = scaler.fit_transform(model_df[numeric_vars])

    formula_parts = numeric_vars + [
        f"C({col})" for col in categorical_vars
    ]

    formula = target + " ~ " + " + ".join(formula_parts)

    model = smf.logit(formula=formula, data=model_df).fit(
        method="lbfgs",
        maxiter=1000,
        disp=False
    )

    if show_summary:
        print(model.summary())

    params = model.params
    conf = model.conf_int()

    logistic_results = pd.DataFrame({
        "Coefficient": params,
        "Odds_Ratio": np.exp(params),
        "CI_Lower": np.exp(conf[0]),
        "CI_Upper": np.exp(conf[1]),
        "p_value": model.pvalues
    })

    return (
        model,
        logistic_results
        .round(4)
        .sort_values(by="Odds_Ratio", ascending=False)
    )



def run_logistic_models_mixed(df, mixed_models_vars_dict, target="AttritionFlag", show_summary=False):
    fitted_models = {}
    model_results = {}

    for model_name, model_info in mixed_models_vars_dict.items():
        fitted_models[model_name], model_results[model_name] = logistic_reg_model_mixed(
            df=df,
            numeric_vars=model_info["numeric_vars"],
            categorical_vars=model_info["categorical_vars"],
            target=target,
            show_summary=show_summary
        )

    return fitted_models, model_results