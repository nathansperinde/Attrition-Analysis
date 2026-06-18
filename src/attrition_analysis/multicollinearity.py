import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data_selection import (
    get_categorical_vars,
    get_quantitative_vars
)


def classify_vif(vif):
    if vif < 5:
        return "Baixa"
    if vif < 10:
        return "Moderada"

    return "Forte"


def vif_categorical_table(df, categorical_vars=None, target="AttritionFlag"):

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    vars_to_analyze = [
        col for col in categorical_vars
        if col != target
    ]

    X = df[vars_to_analyze].dropna()
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    X = sm.add_constant(X)

    results = []

    for i, col in enumerate(X.columns):
        if col == "const":
            continue

        vif = variance_inflation_factor(X.values, i)

        results.append({
            "Feature": col,
            "VIF": round(vif, 3),
            "Multicollinearity": classify_vif(vif)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="VIF", ascending=False)
        .reset_index(drop=True)
    )


def vif_quantitative_table(df, quantitative_vars=None, target="AttritionFlag"):

    if quantitative_vars is None:
        quantitative_vars = get_quantitative_vars(df)

    vars_to_analyze = [
        col for col in quantitative_vars
        if col != target
    ]

    X = df[vars_to_analyze].dropna()
    X = sm.add_constant(X)

    results = []

    for i, col in enumerate(X.columns):
        if col == "const":
            continue

        vif = variance_inflation_factor(X.values, i)

        results.append({
            "Feature": col,
            "VIF": round(vif, 3),
            "Multicollinearity": classify_vif(vif)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="VIF", ascending=False)
        .reset_index(drop=True)
    )


def vif_mixed_table(df, numeric_vars=None, categorical_vars=None, target="AttritionFlag"):

    if numeric_vars is None:
        numeric_vars = get_quantitative_vars(df)

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    numeric_vars = [
        col for col in numeric_vars
        if col != target
    ]

    categorical_vars = [
        col for col in categorical_vars
        if col != target
    ]

    selected_vars = numeric_vars + categorical_vars

    X = df[selected_vars].dropna()
    X = pd.get_dummies(X, columns=categorical_vars, drop_first=True)
    X = X.astype(float)
    X = sm.add_constant(X)

    results = []

    for i, col in enumerate(X.columns):
        if col == "const":
            continue

        vif = variance_inflation_factor(X.values, i)

        results.append({
            "Feature": col,
            "VIF": round(vif, 3),
            "Multicollinearity": classify_vif(vif)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="VIF", ascending=False)
        .reset_index(drop=True)
    )