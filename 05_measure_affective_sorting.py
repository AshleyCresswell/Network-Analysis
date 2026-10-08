"""
Measure affective sorting in the Paper 2 co-commenting network.

The script:

1. Loads the emotion-scored Paper 2 comment corpus.
2. Loads the final co-commenting network.
3. Aggregates the eight emotion scores into user affect profiles.
4. Retains users represented in both the affect data and network.
5. Measures affective similarity between users using cosine similarity.
6. Calculates each user's mean similarity to their network neighbours.
7. Calculates a random-user similarity baseline using 50 comparison users.
8. Measures affective sorting as observed network similarity minus
   random-user similarity.
9. Validates and saves the user-level affective sorting dataset.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMOTION_FILE = Path(
    "data/processed/paper2_2025_emotions.csv"
)

NETWORK_FILE = Path(
    "data/processed/paper2_2025_cocomment_network.graphml"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "paper2_2025_affective_sorting.csv"
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

RANDOM_K = 50
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load emotion scores and the co-commenting network."""

    df = pd.read_csv(
        EMOTION_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = [
        "author",
        *EMOTION_COLS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Emotion data is missing required "
            f"columns: {missing}"
        )

    graph = nx.read_graphml(
        NETWORK_FILE
    )

    return df, graph


# ---------------------------------------------------------------------
# User affect profiles
# ---------------------------------------------------------------------

def build_user_affect(df):
    """
    Construct an eight-dimensional affect profile for each user.

    Each dimension represents the user's mean comment-level score
    for one of the eight emotions.
    """

    user_affect = (
        df.groupby(
            "author"
        )[EMOTION_COLS]
        .mean()
    )

    return user_affect


# ---------------------------------------------------------------------
# Align affect data and network
# ---------------------------------------------------------------------

def align_users(user_affect, graph):
    """
    Retain users represented in both the emotion data and network.
    """

    network_users = set(
        graph.nodes()
    )

    affect_users = set(
        user_affect.index
    )

    common_users = sorted(
        network_users
        & affect_users
    )

    user_affect = user_affect.loc[
        common_users
    ].copy()

    graph = graph.subgraph(
        common_users
    ).copy()

    return user_affect, graph


# ---------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------

def calculate_similarity_matrix(user_affect):
    """
    Calculate pairwise cosine similarity between user affect vectors.
    """

    affect_matrix = (
        user_affect[
            EMOTION_COLS
        ]
        .to_numpy(
            dtype=float
        )
    )

    similarity_matrix = cosine_similarity(
        affect_matrix
    )

    users = user_affect.index.to_list()

    user_index = {
        user: index
        for index, user in enumerate(users)
    }

    return similarity_matrix, users, user_index


# ---------------------------------------------------------------------
# Affective sorting
# ---------------------------------------------------------------------

def measure_affective_sorting(
    graph,
    similarity_matrix,
    users,
    user_index,
):
    """
    Measure affective sorting for each network user.

    Observed similarity is the mean cosine similarity between a user
    and their actual co-commenting neighbours.

    Random similarity is the mean cosine similarity between that user
    and a random sample of other users.

    Affective sorting is the difference between these quantities.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []

    all_users = np.asarray(
        users,
        dtype=object,
    )

    for user in users:

        index = user_index[
            user
        ]

        # -------------------------------------------------------------
        # Actual network neighbours
        # -------------------------------------------------------------

        neighbours = [
            neighbour
            for neighbour in graph.neighbors(user)
            if neighbour in user_index
        ]

        if not neighbours:
            continue

        neighbour_indices = [
            user_index[neighbour]
            for neighbour in neighbours
        ]

        observed_similarity = float(
            similarity_matrix[
                index,
                neighbour_indices,
            ].mean()
        )

        # -------------------------------------------------------------
        # Random comparison users
        # -------------------------------------------------------------

        # Exclude the focal user from the random comparison pool.

        random_pool = all_users[
            all_users != user
        ]

        sample_size = min(
            RANDOM_K,
            len(random_pool),
        )

        random_users = rng.choice(
            random_pool,
            size=sample_size,
            replace=False,
        )

        random_indices = [
            user_index[random_user]
            for random_user in random_users
        ]

        random_similarity = float(
            similarity_matrix[
                index,
                random_indices,
            ].mean()
        )

        # -------------------------------------------------------------
        # Affective sorting
        # -------------------------------------------------------------

        affective_sorting = (
            observed_similarity
            - random_similarity
        )

        results.append(
            {
                "author": user,
                "observed_similarity": observed_similarity,
                "random_similarity": random_similarity,
                "affective_sorting_net": affective_sorting,
                "degree": len(neighbours),
            }
        )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------------------
# Add participation measures
# ---------------------------------------------------------------------

def add_user_measures(
    sorting_df,
    df,
):
    """
    Add user-level comment and exposure measures required for
    subsequent regression analysis.
    """

    user_measures = (
        df.groupby(
            "author"
        )
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
        .reset_index()
    )

    sorting_df = sorting_df.merge(
        user_measures,
        on="author",
        how="left",
    )

    return sorting_df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_results(df):
    """Validate the final affective sorting measures."""

    required = [
        "author",
        "observed_similarity",
        "random_similarity",
        "affective_sorting_net",
        "degree",
        "total_comments",
        "pct_influencer",
        "pct_right",
    ]

    assert all(
        column in df.columns
        for column in required
    )

    assert (
        df["author"]
        .is_unique
    )

    assert (
        df["degree"]
        > 0
    ).all()

    assert (
        df["observed_similarity"]
        .between(-1, 1)
        .all()
    )

    assert (
        df["random_similarity"]
        .between(-1, 1)
        .all()
    )

    assert (
        df["pct_influencer"]
        .between(0, 1)
        .all()
    )

    assert (
        df["pct_right"]
        .between(0, 1)
        .all()
    )

    assert np.isfinite(
        df[
            [
                "observed_similarity",
                "random_similarity",
                "affective_sorting_net",
            ]
        ].to_numpy()
    ).all()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(df):
    """Print a concise summary of affective sorting."""

    print("\n" + "=" * 70)
    print("AFFECTIVE SORTING")
    print("=" * 70)

    print(
        f"Users analysed: "
        f"{len(df):,}"
    )

    print(
        f"Mean observed similarity: "
        f"{df['observed_similarity'].mean():.4f}"
    )

    print(
        f"Mean random similarity: "
        f"{df['random_similarity'].mean():.4f}"
    )

    print(
        f"Mean affective sorting: "
        f"{df['affective_sorting_net'].mean():.4f}"
    )

    print(
        f"Median affective sorting: "
        f"{df['affective_sorting_net'].median():.4f}"
    )

    print(
        f"Users with positive sorting: "
        f"{(df['affective_sorting_net'] > 0).mean():.1%}"
    )

    print(
        f"Mean degree: "
        f"{df['degree'].mean():.2f}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Measure and save user-level affective sorting."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PAPER 2 — AFFECTIVE SORTING")
    print("=" * 70)

    # Load comment-level emotions and network.

    df, graph = load_data()

    print(
        f"Comments loaded: "
        f"{len(df):,}"
    )

    print(
        f"Network users: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Network edges: "
        f"{graph.number_of_edges():,}"
    )

    # Construct user affect profiles.

    user_affect = build_user_affect(
        df
    )

    print(
        f"Users with affect profiles: "
        f"{len(user_affect):,}"
    )

    # Retain users present in both sources.

    user_affect, graph = align_users(
        user_affect,
        graph,
    )

    print(
        f"Matched users: "
        f"{len(user_affect):,}"
    )

    # Calculate pairwise affective similarity.

    (
        similarity_matrix,
        users,
        user_index,
    ) = calculate_similarity_matrix(
        user_affect
    )

    # Compare actual neighbours with random users.

    sorting_df = measure_affective_sorting(
        graph,
        similarity_matrix,
        users,
        user_index,
    )

    # Add variables required by the regression stage.

    sorting_df = add_user_measures(
        sorting_df,
        df,
    )

    validate_results(
        sorting_df
    )

    print_summary(
        sorting_df
    )

    sorting_df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False,
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
