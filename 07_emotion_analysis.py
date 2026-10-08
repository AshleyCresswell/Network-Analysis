"""
Analyse emotion-specific sorting and emotional contagion in Paper 2.

The script:

1. Loads the emotion-scored comment corpus and co-commenting network.
2. Constructs user-level affect profiles for eight emotions.
3. Measures emotion-specific sorting against a random-user baseline.
4. Estimates separate sorting models for each emotion.
5. Calculates adjusted predictions across the four media environments.
6. Orders comments within videos and constructs rolling emotion lags.
7. Estimates emotional contagion models using the previous 20 comments.
8. Tests whether emotional contagion differs between news and
   influencer environments.
9. Saves the emotion-specific sorting and contagion results.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMOTION_FILE = Path(
    "data/processed/paper2_2025_emotions.csv"
)

NETWORK_FILE = Path(
    "data/processed/paper2_2025_cocomment_network.graphml"
)

TABLE_DIR = Path(
    "results/tables"
)

SORTING_OUTPUT = (
    TABLE_DIR
    / "paper2_emotion_sorting_models.csv"
)

PREDICTION_OUTPUT = (
    TABLE_DIR
    / "paper2_emotion_sorting_predictions.csv"
)

CONTAGION_OUTPUT = (
    TABLE_DIR
    / "paper2_emotional_contagion_models.csv"
)

EMOTION_COLS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

RANDOM_K = 30
ROLLING_WINDOW = 20
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load the emotion-scored corpus and co-commenting network."""

    df = pd.read_csv(
        EMOTION_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = [
        "comment_id",
        "author",
        "video_id",
        "updated_at",
        "channel_type",
        "chan_ideo",
        *EMOTION_COLS,
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

    graph = nx.read_graphml(
        NETWORK_FILE
    )

    return df, graph


# ---------------------------------------------------------------------
# User-level data
# ---------------------------------------------------------------------

def build_user_data(df):
    """
    Construct user affect profiles and exposure measures.
    """

    user_affect = (
        df.groupby("author")[
            EMOTION_COLS
        ]
        .mean()
    )

    user_exposure = (
        df.groupby("author")
        .agg(
            total_comments=(
                "comment_id",
                "count",
            ),

            pct_influencer=(
                "channel_type",
                lambda values:
                    (
                        values
                        == "influencer"
                    ).mean(),
            ),

            pct_right=(
                "chan_ideo",
                lambda values:
                    (
                        values
                        == "right"
                    ).mean(),
            ),
        )
    )

    user_exposure[
        "influencer_binary"
    ] = (
        user_exposure[
            "pct_influencer"
        ]
        >= 0.5
    ).astype(int)

    user_exposure[
        "right_binary"
    ] = (
        user_exposure[
            "pct_right"
        ]
        >= 0.5
    ).astype(int)

    user_exposure[
        "interaction"
    ] = (
        user_exposure[
            "influencer_binary"
        ]
        * user_exposure[
            "right_binary"
        ]
    )

    return (
        user_affect,
        user_exposure,
    )


# ---------------------------------------------------------------------
# Emotion-specific sorting
# ---------------------------------------------------------------------

def compute_emotion_sorting(
    emotion,
    graph,
    user_affect,
    rng,
):
    """
    Measure user-level sorting for one emotion.

    Similarity is defined as one minus the absolute difference
    between two users' mean emotion scores.
    """

    common = sorted(
        set(graph.nodes())
        & set(user_affect.index)
    )

    common_set = set(common)

    affect = user_affect.loc[
        common
    ]

    values = (
        affect[emotion]
        .to_numpy(
            dtype=np.float32
        )
    )

    name_to_index = {
        name: index
        for index, name
        in enumerate(common)
    }

    n_users = len(common)

    rows = []

    for user in common:

        neighbours = [
            neighbour
            for neighbour
            in graph.neighbors(user)
            if neighbour in common_set
        ]

        if not neighbours:
            continue

        user_index = (
            name_to_index[user]
        )

        neighbour_indices = np.array(
            [
                name_to_index[
                    neighbour
                ]
                for neighbour
                in neighbours
            ],
            dtype=np.int32,
        )

        # Similarity ranges from 0 to 1 when the underlying
        # emotion scores are bounded between 0 and 1.

        observed = float(
            np.mean(
                1.0
                - np.abs(
                    values[user_index]
                    - values[
                        neighbour_indices
                    ]
                )
            )
        )

        # Match the original analysis by drawing 30 random
        # comparison users for each focal user.

        random_indices = rng.integers(
            0,
            n_users,
            size=min(
                RANDOM_K,
                n_users,
            ),
        )

        random_indices = (
            random_indices[
                random_indices
                != user_index
            ]
        )

        if len(random_indices) == 0:
            continue

        expected = float(
            np.mean(
                1.0
                - np.abs(
                    values[user_index]
                    - values[
                        random_indices
                    ]
                )
            )
        )

        rows.append(
            {
                "author": user,
                "observed_similarity":
                    observed,
                "random_similarity":
                    expected,
                "sorting":
                    observed - expected,
                "degree":
                    len(neighbour_indices),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Emotion-specific models
# ---------------------------------------------------------------------

def fit_emotion_models(
    graph,
    user_affect,
    user_exposure,
):
    """
    Estimate the affective sorting model separately for each emotion.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []
    models = {}

    for emotion in EMOTION_COLS:

        print(
            f"Emotion-specific sorting: "
            f"{emotion}"
        )

        emotion_df = (
            compute_emotion_sorting(
                emotion,
                graph,
                user_affect,
                rng,
            )
        )

        regression_df = (
            emotion_df
            .merge(
                user_exposure
                .reset_index(),
                on="author",
                how="left",
            )
            .dropna()
        )

        predictors = [
            "influencer_binary",
            "right_binary",
            "interaction",
            "degree",
            "total_comments",
        ]

        X = sm.add_constant(
            regression_df[
                predictors
            ]
        )

        # Scale by 100 to match the final emotion-specific
        # regression specification used in the analysis.

        y = (
            regression_df[
                "sorting"
            ]
            * 100
        )

        model = sm.OLS(
            y,
            X,
        ).fit()

        models[emotion] = model

        conf_int = (
            model.conf_int()
        )

        for term in model.params.index:

            results.append(
                {
                    "emotion":
                        emotion,
                    "term":
                        term,
                    "coefficient":
                        model.params[
                            term
                        ],
                    "std_error":
                        model.bse[
                            term
                        ],
                    "ci_lower":
                        conf_int.loc[
                            term, 0
                        ],
                    "ci_upper":
                        conf_int.loc[
                            term, 1
                        ],
                    "p_value":
                        model.pvalues[
                            term
                        ],
                    "r_squared":
                        model.rsquared,
                    "n":
                        int(
                            model.nobs
                        ),
                }
            )

    return (
        pd.DataFrame(results),
        models,
    )


# ---------------------------------------------------------------------
# Environment predictions
# ---------------------------------------------------------------------

def calculate_emotion_predictions(
    models,
    user_exposure,
):
    """
    Calculate adjusted emotion-specific sorting predictions for
    the four political media environments.
    """

    mean_comments = (
        user_exposure[
            "total_comments"
        ].mean()
    )

    predictions = []

    environments = [
        (
            "Left News",
            0,
            0,
        ),
        (
            "Left Influencer",
            1,
            0,
        ),
        (
            "Right News",
            0,
            1,
        ),
        (
            "Right Influencer",
            1,
            1,
        ),
    ]

    for emotion, model in models.items():

        # Degree was included in the original model.
        # Hold it at the model sample's approximate mean by
        # reconstructing it from the fitted data.

        mean_degree = float(
            model.model.exog[
                :,
                model.model.exog_names.index(
                    "degree"
                )
            ].mean()
        )

        for (
            environment,
            influencer,
            right,
        ) in environments:

            prediction_df = pd.DataFrame(
                {
                    "influencer_binary":
                        [influencer],
                    "right_binary":
                        [right],
                    "interaction":
                        [
                            influencer
                            * right
                        ],
                    "degree":
                        [mean_degree],
                    "total_comments":
                        [mean_comments],
                }
            )

            X_pred = sm.add_constant(
                prediction_df,
                has_constant="add",
            )

            prediction = (
                model
                .get_prediction(
                    X_pred
                )
                .summary_frame(
                    alpha=0.05
                )
                .iloc[0]
            )

            predictions.append(
                {
                    "emotion":
                        emotion,
                    "environment":
                        environment,
                    "predicted_sorting":
                        prediction[
                            "mean"
                        ],
                    "ci_lower":
                        prediction[
                            "mean_ci_lower"
                        ],
                    "ci_upper":
                        prediction[
                            "mean_ci_upper"
                        ],
                }
            )

    return pd.DataFrame(
        predictions
    )


# ---------------------------------------------------------------------
# Prepare contagion data
# ---------------------------------------------------------------------

def prepare_contagion_data(df):
    """
    Prepare comment-level variables for the emotional contagion models.
    """

    df = df.copy()

    df["updated_at"] = (
        pd.to_datetime(
            df["updated_at"],
            errors="coerce",
            utc=True,
        )
    )

    df = df.dropna(
        subset=[
            "updated_at",
            "video_id",
        ]
    )

    df = df.sort_values(
        [
            "video_id",
            "updated_at",
        ]
    )

    df[
        "influencer_binary"
    ] = (
        df["channel_type"]
        == "influencer"
    ).astype(int)

    df[
        "right_binary"
    ] = (
        df["chan_ideo"]
        == "right"
    ).astype(int)

    if "is_reply" in df.columns:

        df[
            "is_reply_bin"
        ] = (
            df["is_reply"]
            .astype(int)
        )

    else:

        df[
            "is_reply_bin"
        ] = 0

    return df


# ---------------------------------------------------------------------
# Emotional contagion
# ---------------------------------------------------------------------

def fit_contagion_models(df):
    """
    Estimate emotional contagion models for all eight emotions.

    The lag represents the mean emotion score among the preceding
    20 comments within the same video. Its interaction with the
    influencer indicator tests whether this relationship differs
    between influencer and news environments.
    """

    results = []

    for emotion in EMOTION_COLS:

        print(
            f"Emotional contagion: "
            f"{emotion}"
        )

        lag_column = (
            f"lag_{emotion}"
        )

        # Rolling context from the previous 20 comments
        # within each video.

        df[lag_column] = (
            df.groupby(
                "video_id"
            )[emotion]
            .transform(
                lambda values:
                    values
                    .rolling(
                        ROLLING_WINDOW,
                        min_periods=1,
                    )
                    .mean()
                    .shift(1)
            )
        )

        contagion_df = (
            df.dropna(
                subset=[
                    lag_column,
                    emotion,
                ]
            )
            .copy()
        )

        contagion_df[
            "lag_x_influencer"
        ] = (
            contagion_df[
                lag_column
            ]
            * contagion_df[
                "influencer_binary"
            ]
        )

        predictors = [
            lag_column,
            "lag_x_influencer",
            "influencer_binary",
            "right_binary",
            "is_reply_bin",
        ]

        X = sm.add_constant(
            contagion_df[
                predictors
            ]
        )

        y = (
            contagion_df[
                emotion
            ]
        )

        model = sm.OLS(
            y,
            X,
        ).fit()

        conf_int = (
            model.conf_int()
        )

        for term in model.params.index:

            results.append(
                {
                    "emotion":
                        emotion,
                    "term":
                        term,
                    "coefficient":
                        model.params[
                            term
                        ],
                    "std_error":
                        model.bse[
                            term
                        ],
                    "ci_lower":
                        conf_int.loc[
                            term, 0
                        ],
                    "ci_upper":
                        conf_int.loc[
                            term, 1
                        ],
                    "p_value":
                        model.pvalues[
                            term
                        ],
                    "r_squared":
                        model.rsquared,
                    "n":
                        int(
                            model.nobs
                        ),
                }
            )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_results(
    sorting_results,
    predictions,
    contagion_results,
):
    """Validate the final emotion-analysis outputs."""

    assert set(
        sorting_results[
            "emotion"
        ].unique()
    ) == set(
        EMOTION_COLS
    )

    assert set(
        contagion_results[
            "emotion"
        ].unique()
    ) == set(
        EMOTION_COLS
    )

    assert (
        predictions
        .groupby(
            "emotion"
        )
        .size()
        .eq(4)
        .all()
    )

    numeric_outputs = [
        sorting_results[
            "coefficient"
        ],
        predictions[
            "predicted_sorting"
        ],
        contagion_results[
            "coefficient"
        ],
    ]

    for values in numeric_outputs:

        assert np.isfinite(
            values.to_numpy()
        ).all()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(
    sorting_results,
    predictions,
    contagion_results,
):
    """Print concise summaries of the final analyses."""

    print("\n" + "=" * 70)
    print("EMOTION-SPECIFIC AFFECTIVE SORTING")
    print("=" * 70)

    sorting_summary = (
        sorting_results[
            sorting_results[
                "term"
            ]
            == "influencer_binary"
        ][
            [
                "emotion",
                "coefficient",
                "p_value",
            ]
        ]
        .sort_values(
            "coefficient",
            ascending=False,
        )
    )

    print(
        sorting_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ADJUSTED ENVIRONMENT PREDICTIONS")
    print("=" * 70)

    print(
        predictions
        .pivot(
            index="emotion",
            columns="environment",
            values="predicted_sorting",
        )
        .round(4)
    )

    print("\n" + "=" * 70)
    print("EMOTIONAL CONTAGION")
    print("=" * 70)

    contagion_summary = (
        contagion_results[
            contagion_results[
                "term"
            ].str.startswith(
                "lag_"
            )
            & (
                contagion_results[
                    "term"
                ]
                != "lag_x_influencer"
            )
        ][
            [
                "emotion",
                "coefficient",
                "p_value",
            ]
        ]
    )

    print(
        contagion_summary
        .round(4)
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Run and save the final Paper 2 emotion analyses."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PAPER 2 — EMOTION ANALYSIS")
    print("=" * 70)

    df, graph = load_data()

    print(
        f"Comments loaded: "
        f"{len(df):,}"
    )

    print(
        f"Network users: "
        f"{graph.number_of_nodes():,}"
    )

    # -------------------------------------------------------------
    # Emotion-specific affective sorting
    # -------------------------------------------------------------

    (
        user_affect,
        user_exposure,
    ) = build_user_data(
        df
    )

    (
        sorting_results,
        models,
    ) = fit_emotion_models(
        graph,
        user_affect,
        user_exposure,
    )

    predictions = (
        calculate_emotion_predictions(
            models,
            user_exposure,
        )
    )

    # -------------------------------------------------------------
    # Emotional contagion
    # -------------------------------------------------------------

    contagion_data = (
        prepare_contagion_data(
            df
        )
    )

    contagion_results = (
        fit_contagion_models(
            contagion_data
        )
    )

    # -------------------------------------------------------------
    # Validate and save
    # -------------------------------------------------------------

    validate_results(
        sorting_results,
        predictions,
        contagion_results,
    )

    print_summary(
        sorting_results,
        predictions,
        contagion_results,
    )

    sorting_results.to_csv(
        SORTING_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    contagion_results.to_csv(
        CONTAGION_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    print(
        f"\nSorting models saved: "
        f"{SORTING_OUTPUT}"
    )

    print(
        f"Predictions saved: "
        f"{PREDICTION_OUTPUT}"
    )

    print(
        f"Contagion models saved: "
        f"{CONTAGION_OUTPUT}"
    )


if __name__ == "__main__":
    main()
