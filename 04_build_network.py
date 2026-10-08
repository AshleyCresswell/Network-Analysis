"""
Build the Paper 2 co-commenting network.

The script:

1. Loads the emotion-scored Paper 2 comment corpus.
2. Identifies unique user-video participation pairs.
3. Samples a maximum of 500 users per video to control
   the computational cost of network projection.
4. Constructs a sparse user-video participation matrix.
5. Projects the bipartite matrix into a user-user network.
6. Weights user connections by the number of shared videos.
7. Retains each user's 25 strongest co-commenting ties.
8. Validates and saves the final network for affective
   sorting analysis.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/paper2_2025_emotions.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

EDGE_OUTPUT = (
    OUTPUT_DIR
    / "paper2_2025_network_edges.csv"
)

NODE_OUTPUT = (
    OUTPUT_DIR
    / "paper2_2025_network_nodes.csv"
)

GRAPH_OUTPUT = (
    OUTPUT_DIR
    / "paper2_2025_cocomment_network.graphml"
)

SAMPLE_PER_VIDEO = 500
TOPK_PER_USER = 25
MIN_SHARED_VIDEOS = 1
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load the emotion-scored Paper 2 corpus."""

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = [
        "author",
        "video_id",
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
# User-video participation
# ---------------------------------------------------------------------

def prepare_user_video_pairs(df):
    """
    Construct unique user-video participation pairs.

    Multiple comments by the same user on the same video count as
    one participation tie when constructing the network.
    """

    pairs = (
        df[
            [
                "author",
                "video_id",
            ]
        ]
        .dropna()
        .drop_duplicates(
            [
                "author",
                "video_id",
            ]
        )
    )

    return pairs


# ---------------------------------------------------------------------
# Sample large videos
# ---------------------------------------------------------------------

def sample_video_users(df_uv):
    """
    Limit each video to a maximum of 500 participating users.

    This bounds the computational cost of projecting highly active
    videos into the user-user co-commenting network.
    """

    sampled = (
        df_uv
        .groupby(
            "video_id",
            group_keys=False,
        )[
            [
                "author",
                "video_id",
            ]
        ]
        .apply(
            lambda group:
                group.sample(
                    n=min(
                        len(group),
                        SAMPLE_PER_VIDEO,
                    ),
                    random_state=RANDOM_STATE,
                )
        )
        .reset_index(drop=True)
    )

    return sampled


# ---------------------------------------------------------------------
# User-video matrix
# ---------------------------------------------------------------------

def build_user_video_matrix(df_uv):
    """
    Construct a sparse binary user-video participation matrix.
    """

    user_codes, users = pd.factorize(
        df_uv["author"],
        sort=True,
    )

    video_codes, videos = pd.factorize(
        df_uv["video_id"],
        sort=True,
    )

    pairs = (
        pd.DataFrame(
            {
                "user": user_codes,
                "video": video_codes,
            }
        )
        .drop_duplicates()
    )

    matrix = coo_matrix(
        (
            np.ones(
                len(pairs),
                dtype=np.int8,
            ),
            (
                pairs["user"].values,
                pairs["video"].values,
            ),
        ),
        shape=(
            len(users),
            len(videos),
        ),
    ).tocsr()

    return matrix, users, videos


# ---------------------------------------------------------------------
# Project network
# ---------------------------------------------------------------------

def project_network(matrix, users):
    """
    Project the user-video matrix into a weighted user-user network.

    Edge weights represent the number of videos on which two users
    both participated.
    """

    shared_videos = (
        matrix
        @ matrix.T
    ).tocsr()

    # Remove self-connections.

    shared_videos.setdiag(0)
    shared_videos.eliminate_zeros()

    graph = nx.Graph()

    graph.add_nodes_from(
        users.tolist()
    )

    for user_index in range(
        shared_videos.shape[0]
    ):

        start = (
            shared_videos
            .indptr[user_index]
        )

        end = (
            shared_videos
            .indptr[user_index + 1]
        )

        neighbours = (
            shared_videos
            .indices[start:end]
        )

        weights = (
            shared_videos
            .data[start:end]
        )

        if len(neighbours) == 0:
            continue

        # Retain ties meeting the minimum shared-video threshold.

        mask = (
            weights
            >= MIN_SHARED_VIDEOS
        )

        neighbours = neighbours[mask]
        weights = weights[mask]

        if len(neighbours) == 0:
            continue

        # Retain the strongest ties for each user to control
        # network density.

        if len(neighbours) > TOPK_PER_USER:

            top = np.argpartition(
                weights,
                -TOPK_PER_USER,
            )[
                -TOPK_PER_USER:
            ]

            neighbours = neighbours[top]
            weights = weights[top]

        user_name = users[
            user_index
        ]

        for neighbour, weight in zip(
            neighbours,
            weights,
        ):

            neighbour_name = users[
                neighbour
            ]

            if user_name == neighbour_name:
                continue

            graph.add_edge(
                user_name,
                neighbour_name,
                weight=int(weight),
            )

    return graph


# ---------------------------------------------------------------------
# Export network
# ---------------------------------------------------------------------

def create_edge_list(graph):
    """Convert the network into an edge-list dataframe."""

    edges = nx.to_pandas_edgelist(
        graph,
        source="source",
        target="target",
    )

    if not edges.empty:

        edges = edges[
            [
                "source",
                "target",
                "weight",
            ]
        ]

    return edges


def create_node_list(graph):
    """Create a node-level network summary."""

    degree = dict(
        graph.degree()
    )

    weighted_degree = dict(
        graph.degree(
            weight="weight"
        )
    )

    nodes = pd.DataFrame(
        {
            "author": list(
                graph.nodes()
            ),
        }
    )

    nodes["degree"] = (
        nodes["author"]
        .map(degree)
    )

    nodes["weighted_degree"] = (
        nodes["author"]
        .map(weighted_degree)
    )

    return nodes


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_network(
    graph,
    edges,
    nodes,
):
    """Run basic checks on the projected network."""

    assert (
        graph.number_of_nodes()
        == len(nodes)
    )

    assert (
        graph.number_of_edges()
        == len(edges)
    )

    assert not nx.number_of_selfloops(
        graph
    )

    if not edges.empty:

        assert (
            edges["weight"]
            >= MIN_SHARED_VIDEOS
        ).all()

        assert (
            edges["source"]
            != edges["target"]
        ).all()

    assert (
        nodes["author"]
        .is_unique
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(
    df_uv,
    graph,
    edges,
    nodes,
):
    """Print a concise network summary."""

    print("\n" + "=" * 70)
    print("CO-COMMENTING NETWORK")
    print("=" * 70)

    print(
        f"User-video pairs: "
        f"{len(df_uv):,}"
    )

    print(
        f"Users: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges():,}"
    )

    print(
        f"Videos: "
        f"{df_uv['video_id'].nunique():,}"
    )

    if graph.number_of_nodes() > 0:

        isolates = list(
            nx.isolates(graph)
        )

        print(
            f"Isolated users: "
            f"{len(isolates):,}"
        )

        print(
            f"Mean degree: "
            f"{nodes['degree'].mean():.2f}"
        )

        print(
            f"Median degree: "
            f"{nodes['degree'].median():.2f}"
        )

    if not edges.empty:

        print(
            f"Mean shared videos per edge: "
            f"{edges['weight'].mean():.2f}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Build and save the Paper 2 co-commenting network."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PAPER 2 — NETWORK CONSTRUCTION")
    print("=" * 70)

    df = load_data()

    print(
        f"Comments loaded: "
        f"{len(df):,}"
    )

    # Construct unique participation pairs.

    df_uv = prepare_user_video_pairs(
        df
    )

    print(
        f"Unique user-video pairs: "
        f"{len(df_uv):,}"
    )

    # Bound projection cost for highly active videos.

    df_uv = sample_video_users(
        df_uv
    )

    print(
        f"Pairs after video sampling: "
        f"{len(df_uv):,}"
    )

    # Construct sparse user-video matrix.

    matrix, users, videos = (
        build_user_video_matrix(
            df_uv
        )
    )

    print(
        f"Matrix: "
        f"{matrix.shape[0]:,} users × "
        f"{matrix.shape[1]:,} videos"
    )

    # Project into user-user network.

    graph = project_network(
        matrix,
        users,
    )

    edges = create_edge_list(
        graph
    )

    nodes = create_node_list(
        graph
    )

    validate_network(
        graph,
        edges,
        nodes,
    )

    print_summary(
        df_uv,
        graph,
        edges,
        nodes,
    )

    # Save edge and node tables.

    edges.to_csv(
        EDGE_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    nodes.to_csv(
        NODE_OUTPUT,
        encoding="utf-8-sig",
        index=False,
    )

    # GraphML preserves the network structure for later reuse
    # without requiring the projection to be recomputed.

    nx.write_graphml(
        graph,
        GRAPH_OUTPUT,
    )

    print(
        f"\nEdges saved: "
        f"{EDGE_OUTPUT}"
    )

    print(
        f"Nodes saved: "
        f"{NODE_OUTPUT}"
    )

    print(
        f"Network saved: "
        f"{GRAPH_OUTPUT}"
    )


if __name__ == "__main__":
    main()
