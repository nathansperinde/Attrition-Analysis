import pandas as pd
import numpy as np
from itertools import combinations, product

from scipy.stats import spearmanr, chi2_contingency
from scipy.stats import kruskal

from .data_selection import (
    get_categorical_vars,
    get_quantitative_vars
)

def classify_association(value):
    value = abs(value)

    if value < 0.10:
        return "Muito fraca"
    if value < 0.20:
        return "Fraca"
    if value < 0.30:
        return "Moderada"

    return "Forte"


def classify_correlation(value):
    value = abs(value)

    if value < 0.30:
        return "Fraca"
    if value < 0.50:
        return "Moderada"
    if value < 0.70:
        return "Forte"

    return "Muito forte"


def classify_eta_squared(eta):
    if pd.isna(eta):
        return np.nan

    if eta < 0.01:
        return "Muito fraca"
    if eta < 0.06:
        return "Fraca"
    if eta < 0.14:
        return "Moderada"

    return "Forte"
    

def cramers_v(x, y):
    table = pd.crosstab(x, y)

    chi2, p_value, _, _ = chi2_contingency(table)

    n = table.sum().sum()
    r, k = table.shape

    denominator = min(k - 1, r - 1)

    if denominator == 0:
        return np.nan, p_value

    v = np.sqrt((chi2 / n) / denominator)

    return v, p_value


def check_chi_square_assumptions(df, primary_col, categorical_vars=None):

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    results = []

    for secondary_col in categorical_vars:
        if secondary_col == primary_col:
            continue

        table = pd.crosstab(df[primary_col], df[secondary_col])
        rows, columns = table.shape

        _, _, _, expected = chi2_contingency(table)

        total_cells = expected.size
        cells_below_5 = (expected < 5).sum()
        percent_below_5 = cells_below_5 / total_cells * 100
        min_expected = expected.min()

        results.append({
            "Primary_Variable": primary_col,
            "Secondary_Variable": secondary_col,
            "Rows": rows,
            "Columns": columns,
            "Total_Cells": total_cells,
            "Cells_Expected_Below_5": cells_below_5,
            "Percent_Expected_Below_5": round(percent_below_5, 2),
            "Min_Expected": round(min_expected, 2),
            "Assumptions_OK": "Sim" if percent_below_5 <= 20 and min_expected >= 1 else "Não"
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            by=["Assumptions_OK", "Percent_Expected_Below_5"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )


def chi_square_table(df, primary_col, categorical_vars=None):

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    results = []

    for secondary_col in categorical_vars:
        if secondary_col == primary_col:
            continue

        table = pd.crosstab(df[primary_col], df[secondary_col])
        chi2, p_value, _, _ = chi2_contingency(table)

        results.append({
            "Primary_Variable": primary_col,
            "Secondary_Variable": secondary_col,
            "Chi_square": round(chi2, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não"
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="Chi_square", ascending=False)
        .reset_index(drop=True)
    )


def cramer_table(df, primary_col, categorical_vars=None):

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    results = []

    for secondary_col in categorical_vars:
        if secondary_col == primary_col:
            continue

        v, p_value = cramers_v(df[primary_col], df[secondary_col])

        results.append({
            "Primary_Variable": primary_col,
            "Secondary_Variable": secondary_col,
            "Cramers_V": round(v, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não",
            "Association": classify_association(v)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="Cramers_V", ascending=False)
        .reset_index(drop=True)
    )


def categorical_predictors_cramer_table(df, categorical_vars=None, target="AttritionFlag"):

    if categorical_vars is None:
        categorical_vars = get_categorical_vars(df)

    vars_to_analyze = [
        col for col in categorical_vars
        if col != target
    ]

    results = []

    for var_1, var_2 in combinations(vars_to_analyze, 2):
        v, p_value = cramers_v(df[var_1], df[var_2])

        results.append({
            "Variable_1": var_1,
            "Variable_2": var_2,
            "Cramers_V": round(v, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não",
            "Association": classify_association(v)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by="Cramers_V", ascending=False)
        .reset_index(drop=True)
    )

def correlation_table(df, quantitative_vars=None, target="AttritionFlag"):

    if quantitative_vars is None:
        quantitative_vars = get_quantitative_vars(df)

    results = []

    for col in quantitative_vars:
        if col == target:
            continue

        temp_df = df[[col, target]].dropna()

        corr, p_value = spearmanr(temp_df[col], temp_df[target])
        abs_corr = abs(corr)

        results.append({
            "Feature": col,
            "Correlation": round(corr, 3),
            "Abs_Correlation": round(abs_corr, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não",
            "Association": classify_correlation(corr)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by=["p_value", "Abs_Correlation"], ascending=[True, False])
        .reset_index(drop=True)
    )


def quantitative_predictors_corr_table(df, quantitative_vars=None, target="AttritionFlag"):

    if quantitative_vars is None:
        quantitative_vars = get_quantitative_vars(df)

    vars_to_analyze = [
        col for col in quantitative_vars
        if col != target
    ]

    results = []

    for var_1, var_2 in combinations(vars_to_analyze, 2):
        temp_df = df[[var_1, var_2]].dropna()

        corr, p_value = spearmanr(temp_df[var_1], temp_df[var_2])
        abs_corr = abs(corr)

        results.append({
            "Variable_1": var_1,
            "Variable_2": var_2,
            "Correlation": round(corr, 3),
            "Abs_Correlation": round(abs_corr, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não",
            "Association": classify_correlation(corr)
        })

    return (
        pd.DataFrame(results)
        .sort_values(by=["Abs_Correlation", "p_value"], ascending=[False, True])
        .reset_index(drop=True)
    )


def corr_cat_and_quant(df, categorical_col, numeric_col):
    temp_df = df[[categorical_col, numeric_col]].dropna()
    values = temp_df[numeric_col]

    groups = [
        group[numeric_col].values
        for _, group in temp_df.groupby(categorical_col, observed=True)
    ]

    if len(groups) < 2:
        return np.nan, np.nan, np.nan

    grand_mean = values.mean()

    ss_between = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for group in groups
    )

    ss_total = ((values - grand_mean) ** 2).sum()
    eta_squared = np.nan if ss_total == 0 else ss_between / ss_total

    stat, p_value = kruskal(*groups)

    return eta_squared, stat, p_value


def mixed_association_table(df):
    categorical_vars = get_categorical_vars(df)
    quantitative_vars = get_quantitative_vars(df)

    results = []

    for cat_col, num_col in product(categorical_vars, quantitative_vars):
        eta, stat, p_value = corr_cat_and_quant(
            df=df,
            categorical_col=cat_col,
            numeric_col=num_col
        )

        results.append({
            "Categorical_Variable": cat_col,
            "Quantitative_Variable": num_col,
            "Eta_Squared": round(eta, 3),
            "Kruskal_Statistic": round(stat, 3),
            "p_value": round(p_value, 4),
            "Significativo": "Sim" if p_value < 0.05 else "Não",
            "Força": classify_eta_squared(eta)
        })

    return pd.DataFrame(results)