def get_categorical_vars(df, exclude=None):

    if exclude is None:
        exclude = ["Attrition", "Over18"]

    return [
        col for col in df.select_dtypes(include="object").columns
        if col not in exclude
    ]


def get_quantitative_vars(df, exclude=None):

    if exclude is None:
        exclude = [
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours"
        ]

    return [
        col for col in df.select_dtypes(include="number").columns
        if col not in exclude
    ]


CATEGORICAL_MODEL_VARS = {
    "Modelo 1 — Função Profissional": [
        "OverTime",
        "JobRole",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 2 — Nível Hierárquico": [
        "OverTime",
        "JobLevelGroup",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 3 — Faixa Salarial": [
        "OverTime",
        "IncomeGroup",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 4 — Trajetória Organizacional": [
        "OverTime",
        "CareerStage",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 5 — Estabilidade e Benefícios": [
        "OverTime",
        "RoleStabilityGroup",
        "StockOption",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 6 — Perfil Pessoal": [
        "OverTime",
        "AgeGroup",
        "MaritalStatus",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ],

    "Modelo 7 — Reduzido Conservador": [
        "OverTime",
        "BusinessTravel",
        "JobInvolvementLevel",
        "EnvironmentSatisfactionLevel",
        "SatisfactionLevel",
        "WorkLifeBalanceLevel",
        "DistanceGroup"
    ]
}


QUANTITATIVE_MODEL_VARS = {
    "Modelo 1 — Experiência e Remuneração": [
        "TotalWorkingYears",
        "MonthlyIncome",
        "DistanceFromHome",
        "DailyRate",
        "TrainingTimesLastYear"
    ],

    "Modelo 2 — Idade e Experiência": [
        "Age",
        "TotalWorkingYears",
        "DistanceFromHome",
        "DailyRate",
        "TrainingTimesLastYear"
    ],

    "Modelo 3 — Antiguidade Organizacional": [
        "YearsAtCompany",
        "DistanceFromHome",
        "DailyRate",
        "TrainingTimesLastYear",
        "YearsSinceLastPromotion"
    ],

    "Modelo 4 — Reduzido Conservador": [
        "TotalWorkingYears",
        "DistanceFromHome",
        "DailyRate",
        "TrainingTimesLastYear",
        "YearsSinceLastPromotion"
    ]
}

MIXED_MODEL_VARS = {
    "Modelo 1 — Função Profissional Misto": {
        "numeric_vars": [
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "JobRole",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 2 — Nível Hierárquico e Benefícios": {
        "numeric_vars": [
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "JobLevelGroup",
            "StockOption",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 3 — Rendimento Quantitativo": {
        "numeric_vars": [
            "MonthlyIncome",
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 4 — Experiência Profissional": {
        "numeric_vars": [
            "TotalWorkingYears",
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 5 — Antiguidade Organizacional": {
        "numeric_vars": [
            "YearsAtCompany",
            "YearsSinceLastPromotion",
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 6 — Perfil Pessoal e Condições de Trabalho": {
        "numeric_vars": [
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "AgeGroup",
            "MaritalStatus",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 7 — Reduzido Conservador Misto": {
        "numeric_vars": [
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    },

    "Modelo 8 — Integrado Multidimensional": {
        "numeric_vars": [
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsSinceLastPromotion",
            "DistanceFromHome",
            "DailyRate",
            "TrainingTimesLastYear"
        ],
        "categorical_vars": [
            "OverTime",
            "JobRole",
            "JobLevelGroup",
            "StockOption",
            "AgeGroup",
            "MaritalStatus",
            "BusinessTravel",
            "JobInvolvementLevel",
            "EnvironmentSatisfactionLevel",
            "SatisfactionLevel",
            "WorkLifeBalanceLevel"
        ]
    }
}