"""
Model differences in Paper 2 affective sorting.

The script:

1. Loads the user-level affective sorting dataset.
2. Classifies users by majority news or influencer exposure.
3. Classifies users by majority left or right exposure.
4. Constructs four political media environments.
5. Estimates an OLS model of affective sorting.
6. Tests differences by media type, ideology, and their interaction.
7. Controls for network degree and total commenting activity.
8. Calculates adjusted predictions for the four media environments.
9. Saves the regression results and predicted values.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/paper2_2025_affective_sorting.csv"
)

TABLE_DIR = Path(
    "results/tables"
)

MODEL_OUTPUT = (
    TABLE_DIR
    / "paper2_affective_sorting_model.csv"
)

PREDICTION_OUTPUT = (
    TABLE_DIR
    / "paper2_affective_sorting_predictions.csv"
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load the user-level affective sorting dataset."""

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    required = [
        "author",
        "affective_sorting_net",
        "degree",
        "total_comments",
        "pct_influencer",
        "pct_right",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input data is missing required "
            f"columns: {missing}"
        )

    return df


# ---------------------------------------------------------------------
# Construct analytical variables
# ---------------------------------------------------------------------

def prepare_model_data(df):
    """
    Construct binary media and ideological classifications.

    Users are classified according to whether at least half of their
    comments occur in influencer or right-leaning environments.
    """

    df = df.copy()

    df["influencer_binary"] = (
        df["pct_influencer"]
        >= 0.5
    ).astype(int)

    df["right_binary"] = (
        df["pct_right"]
        >= 0.5
    ).astype(int)

    df["interaction"] = (
        df["influencer_binary"]
        * df["right_binary"]
    )

    conditions = [
        (
            (df["right_binary"] == 0)
            & (df["influencer_binary"] == 0)
        ),
        (
            (df["right_binary"] == 0)
            & (df["influencer_binary"] == 1)
        ),
        (
            (df["right_binary"] == 1)
            & (df["influencer_binary"] == 0)
        ),
        (
            (df["right_binary"] == 1)
            & (df["influencer_binary"] == 1)
        ),
    ]

    labels = [
        "Left News",
        "Left Influencer",
        "Right News",
        "Right Influencer",
    ]

    df["environment"] = np.select(
        conditions,
        labels,
        default="Unknown",
    )

    model_columns = [
        "author",
        "affective_sorting_net",
        "degree",
        "total_comments",
        "pct_influencer",
        "pct_right",
        "influencer_binary",
        "right_binary",
        "interaction",
        "environment",
    ]

    df = (
        df[model_columns]
        .dropna()
        .copy()
    )

    return df


# ---------------------------------------------------------------------
# Main regression
# ---------------------------------------------------------------------

def fit_model(df):
    """
    Estimate the main affective sorting model.

    The dependent variable is multiplied by 100 so coefficients can
    be interpreted as percentage-point differences in affective
    sorting.
    """

    predictors = [
        "influencer_binary",
        "right_binary",
        "interaction",
        "degree",
        "total_comments",
    ]

    X = sm.add_constant(
        df[predictors]
    )

    y = (
        df["affective_sorting_net"]
        * 100
    )

    model = sm.OLS(
        y,
        X,
    ).fit()

    return model


# ---------------------------------------------------------------------
# Model table
# ---------------------------------------------------------------------

def build_model_table(model):
    """Convert regression results into a clean output table."""

    conf_int = model.conf_int()

    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "ci_lower": conf_int[0].values,
            "ci_upper": conf_int[1].values,
            "p_value": model.pvalues.values,
        }
    )

    return table


# ---------------------------------------------------------------------
# Adjusted predictions
# ---------------------------------------------------------------------

def calculate_predictions(
    model,
    df,
):
    """
    Calculate adjusted predictions for the four media environments.

    Network degree and total commenting activity are held at their
    observed sample means.
    """

    mean_degree = (
        df["degree"].mean()
    )

    mean_comments = (
        df["total_comments"].mean()
    )

    prediction_data = pd.DataFrame(
        {
            "environment": [
                "Left News",
                "Left Influencer",
                "Right News",
                "Right Influencer",
            ],

            "influencer_binary": [
                0,
                1,
                0,
                1,
            ],

            "right_binary": [
                0,
                0,
                1,
                1,
            ],
        }
    )

    prediction_data["interaction"] = (
        prediction_data[
            "influencer_binary"
        ]
        * prediction_data[
            "right_binary"
        ]
    )

    prediction_data["degree"] = (
        mean_degree
    )

    prediction_data["total_comments"] = (
        mean_comments
    )

    predictors = [
        "influencer_binary",
        "right_binary",
        "interaction",
        "degree",
        "total_comments",
    ]

    X_pred = sm.add_constant(
        prediction_data[predictors],
        has_constant="add",
    )

    prediction = model.get_prediction(
        X_pred
    )

    prediction_summary = (
        prediction
        .summary_frame(
            alpha=0.05
        )
    )

    prediction_data[
        "predicted_sorting"
    ] = (
        prediction_summary[
            "mean"
        ].values
    )

    prediction_data[
        "ci_lower"
    ] = (
        prediction_summary[
            "mean_ci_lower"
        ].values
    )

    prediction_data[
        "ci_upper"
    ] = (
        prediction_summary[
            "mean_ci_upper"
        ].values
    )

    return prediction_data[
        [
            "environment",
            "predicted_sorting",
            "ci_lower",
            "ci_upper",
        ]
    ]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_results(
    df,
    model_table,
    predictions,
):
    """Validate the analytical and model outputs."""

    assert (
        df["author"]
        .is_unique
    )

    assert set(
        df["influencer_binary"]
        .unique()
    ).issubset(
        {0, 1}
    )

    assert set(
        df["right_binary"]
        .unique()
    ).issubset(
        {0, 1}
    )

    assert set(
        df["environment"]
        .unique()
    ).issubset(
        {
            "Left News",
            "Left Influencer",
            "Right News",
            "Right Influencer",
        }
    )

    assert not (
        model_table[
            [
                "coefficient",
                "std_error",
                "p_value",
            ]
        ]
        .isna()
        .any()
        .any()
    )

    assert (
        len(predictions)
        == 4
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(
    df,
    model,
    predictions,
):
    """Print the main regression and adjusted predictions."""

    print("\n" + "=" * 70)
    print("AFFECTIVE SORTING MODEL")
    print("=" * 70)

    print(
        f"Users analysed: "
        f"{len(df):,}"
    )

    print("\nUsers by environment:")

    print(
        df["environment"]
        .value_counts()
        .reindex(
            [
                "Left News",
                "Left Influencer",
                "Right News",
                "Right Influencer",
            ]
        )
        .fillna(0)
        .astype(int)
    )

    print("\nRegression results:")

    print(
        pd.DataFrame(
            {
                "Coefficient":
                    model.params,
                "SE":
                    model.bse,
                "p":
                    model.pvalues,
            }
        ).round(4)
    )

    print(
        f"\nR²: "
        f"{model.rsquared:.4f}"
    )

    print(
        f"Adjusted R²: "
        f"{model.rsquared_adj:.4f}"
    )

    print("\nAdjusted predictions:")

    print(
        predictions
        .set_index(
            "environment"
        )
        .round(4)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Estimate and save the main affective sorting model."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PAPER 2 — MODELLING AFFECTIVE SORTING")
    print("=" * 70)

    df = load_data()

    print(
        f"Users loaded: "
        f"{len(df):,}"
    )

    # Construct political media environment variables.

    model_df = prepare_model_data(
        df
    )

    # Estimate the main OLS model.

    model = fit_model(
        model_df
    )

    # Construct publication-ready model output.

    model_table = build_model_table(
        model
    )

    # Derive adjusted predictions directly from the fitted model.

    predictions = calculate_predictions(
        model,
        model_df,
    )

    validate_results(
        model_df,
        model_table,
        predictions,
    )

    print_summary(
        model_df,
        model,
        predictions,
    )

    model_table.to_csv(
        MODEL_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    print(
        f"\nModel saved: "
        f"{MODEL_OUTPUT}"
    )

    print(
        f"Predictions saved: "
        f"{PREDICTION_OUTPUT}"
    )


if __name__ == "__main__":
    main()
