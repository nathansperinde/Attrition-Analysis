import pandas as pd

from .data_selection import get_quantitative_vars

def attrition_summary(df, group_col, target="AttritionFlag"):
    return (
        df.groupby(group_col)
        .agg(
            total_employees=(target, "count"),
            attrition_count=(target, "sum"),
            attrition_rate=(target, "mean")
        )
        .reset_index()
        .assign(
            attrition_rate=lambda x: (x["attrition_rate"] * 100).round(2)
        )
        .sort_values(by="attrition_rate", ascending=False)
        .reset_index(drop=True)
    )


def quantitative_stats_by_attrition(df, target="AttritionFlag"):
    
    quantitative_vars = get_quantitative_vars(df)

    results = []

    for col in quantitative_vars:
        if col == target:
            continue

        summary = (
            df.groupby(target)[col]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                Std="std",
                Min="min",
                Q1=lambda x: x.quantile(0.25),
                Q3=lambda x: x.quantile(0.75),
                Max="max"
            )
            .reset_index()
        )

        summary.insert(0, "Variable", col)
        results.append(summary)

    return pd.concat(results, ignore_index=True).round(3)