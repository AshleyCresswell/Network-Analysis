"""
Prepare the final Paper 2 comment corpus.

The script:

1. Loads raw comment files collected from the study channels.
2. Adds channel type and ideological metadata.
3. Calculates comment word counts.
4. Identifies English-language comments.
5. Removes short textual comments with fewer than five words.
6. Retains users contributing at least three eligible comments
   within a channel.
7. Combines all channels into one analysis corpus.
8. Validates and saves the processed dataset.

The final channel design uses Wall Street Journal in place of
New York Post.
"""

from pathlib import Path
import re

import pandas as pd
from langdetect import detect, DetectorFactory


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = (
    PROCESSED_DIR
    / "paper2_2025_corpus.csv"
)

MIN_WORDS = 5
MIN_COMMENTS_PER_AUTHOR = 3

DetectorFactory.seed = 42


# ---------------------------------------------------------------------
# Final channel design
# ---------------------------------------------------------------------

CHANNELS = {
    # -------------------------------------------------------------
    # News — Right
    # -------------------------------------------------------------
    "Fox News": {
        "file": "fox_news_comments.csv",
        "channel_type": "news",
        "chan_ideo": "right",
    },

    "Blaze TV": {
        "file": "blaze_tv_comments.csv",
        "channel_type": "news",
        "chan_ideo": "right",
    },

    "Wall Street Journal": {
        "file": "wall_street_journal_comments.csv",
        "channel_type": "news",
        "chan_ideo": "right",
    },

    # -------------------------------------------------------------
    # News — Left
    # -------------------------------------------------------------
    "MSNOW": {
        "file": "msnow_comments.csv",
        "channel_type": "news",
        "chan_ideo": "left",
    },

    "New York Times": {
        "file": "new_york_times_comments.csv",
        "channel_type": "news",
        "chan_ideo": "left",
    },

    "Young Turks": {
        "file": "young_turks_comments.csv",
        "channel_type": "news",
        "chan_ideo": "left",
    },

    # -------------------------------------------------------------
    # Influencer — Right
    # -------------------------------------------------------------
    "Ben Shapiro": {
        "file": "ben_shapiro_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "right",
    },

    "Steven Crowder": {
        "file": "steven_crowder_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "right",
    },

    "Officer Tatum": {
        "file": "officer_tatum_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "right",
    },

    # -------------------------------------------------------------
    # Influencer — Left
    # -------------------------------------------------------------
    "Brian Cohen": {
        "file": "brian_cohen_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "left",
    },

    "Sam Seder": {
        "file": "sam_seder_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "left",
    },

    "MediasTouch": {
        "file": "mediastouch_comments.csv",
        "channel_type": "influencer",
        "chan_ideo": "left",
    },
}


# ---------------------------------------------------------------------
# Emoji detection
# ---------------------------------------------------------------------

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2700-\u27BF"
    "\u24C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------
# Load channel data
# ---------------------------------------------------------------------

def load_channel(
    channel_name,
    metadata
):
    """Load one channel and attach study metadata."""

    path = (
        RAW_DIR
        / metadata["file"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing channel file: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    required = [
        "comment_id",
        "author",
        "video_id",
        "text",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{channel_name} is missing "
            f"columns: {missing}"
        )

    # Channel classifications are defined centrally rather
    # than relying on potentially inconsistent raw files.

    df["channel"] = channel_name
    df["channel_type"] = (
        metadata["channel_type"]
    )
    df["chan_ideo"] = (
        metadata["chan_ideo"]
    )

    return df


# ---------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------

def add_word_count(df):
    """Calculate the number of whitespace-separated words."""

    df = df.copy()

    df["word_count"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return df


# ---------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------

def detect_english(text):
    """
    Identify English-language textual comments.

    Symbols and emoji are removed before language detection.
    Comments containing no detectable textual content are retained,
    matching the original cleaning procedure.
    """

    if pd.isna(text):
        return False

    text = str(text)

    text_for_detection = re.sub(
        r"[^\w\s,.!?]",
        "",
        text
    )

    # Emoji/symbol-only comments cannot be language-detected
    # and were retained in the original cleaning procedure.

    if not text_for_detection.strip():
        return True

    try:
        return (
            detect(
                text_for_detection
            )
            == "en"
        )

    except Exception:
        return False


def add_language_indicator(df):
    """Add the English-language eligibility indicator."""

    df = df.copy()

    df["is_english"] = (
        df["text"]
        .apply(
            detect_english
        )
    )

    return df


# ---------------------------------------------------------------------
# Text eligibility
# ---------------------------------------------------------------------

def text_is_eligible(
    text,
    word_count
):
    """
    Apply the original short-comment rule.

    Textual comments containing fewer than five words are excluded.
    Emoji/symbol-only comments are retained.
    """

    if pd.isna(text):
        return False

    text = str(text)

    has_emoji = bool(
        EMOJI_PATTERN.search(
            text
        )
    )

    has_text = bool(
        re.search(
            r"\w",
            text
        )
    )

    if word_count < MIN_WORDS:

        # Short text-only comments and short comments combining
        # text with emoji were excluded.

        if has_text:
            return False

    # Emoji/symbol-only comments remain eligible.

    return True


def add_text_eligibility(df):
    """Add the text eligibility indicator."""

    df = df.copy()

    df["text_confirmation"] = (
        df.apply(
            lambda row:
                text_is_eligible(
                    row["text"],
                    row["word_count"]
                ),
            axis=1
        )
    )

    return df


# ---------------------------------------------------------------------
# Clean one channel
# ---------------------------------------------------------------------

def clean_channel(df):
    """Apply comment-level cleaning rules."""

    df = df.copy()

    # Core network fields must be available.

    df = df.dropna(
        subset=[
            "comment_id",
            "author",
            "video_id",
            "text",
        ]
    )

    df = add_word_count(
        df
    )

    df = add_language_indicator(
        df
    )

    df = add_text_eligibility(
        df
    )

    # Retain eligible English-language comments.

    df = df[
        df["is_english"]
        & df["text_confirmation"]
    ].copy()

    # Duplicate comment IDs represent the same API comment.

    df = df.drop_duplicates(
        subset="comment_id",
        keep="first"
    )

    return df


# ---------------------------------------------------------------------
# User activity filter
# ---------------------------------------------------------------------

def filter_active_users(df):
    """
    Retain authors with at least three eligible comments
    within the channel.

    The threshold is applied separately to each channel, matching
    the original Paper 2 data-preparation workflow.
    """

    counts = (
        df["author"]
        .value_counts()
    )

    keep_authors = (
        counts[
            counts
            >= MIN_COMMENTS_PER_AUTHOR
        ]
        .index
    )

    return (
        df[
            df["author"]
            .isin(
                keep_authors
            )
        ]
        .copy()
    )


# ---------------------------------------------------------------------
# Prepare one channel
# ---------------------------------------------------------------------

def prepare_channel(
    channel_name,
    metadata
):
    """Load and prepare one study channel."""

    print("\n" + "-" * 70)
    print(channel_name)
    print("-" * 70)

    df = load_channel(
        channel_name,
        metadata
    )

    original_rows = len(df)
    original_users = (
        df["author"]
        .nunique()
    )

    df = clean_channel(
        df
    )

    eligible_rows = len(df)

    df = filter_active_users(
        df
    )

    final_rows = len(df)
    final_users = (
        df["author"]
        .nunique()
    )

    print(
        f"Raw comments:      "
        f"{original_rows:,}"
    )

    print(
        f"Eligible comments: "
        f"{eligible_rows:,}"
    )

    print(
        f"Final comments:    "
        f"{final_rows:,}"
    )

    print(
        f"Raw users:         "
        f"{original_users:,}"
    )

    print(
        f"Users retained:    "
        f"{final_users:,}"
    )

    return df


# ---------------------------------------------------------------------
# Build corpus
# ---------------------------------------------------------------------

def build_corpus():
    """Prepare and combine all study channels."""

    frames = []

    for channel_name, metadata in (
        CHANNELS.items()
    ):

        channel_df = (
            prepare_channel(
                channel_name,
                metadata
            )
        )

        frames.append(
            channel_df
        )

    corpus = pd.concat(
        frames,
        ignore_index=True
    )

    return corpus


# ---------------------------------------------------------------------
# Standardise variables
# ---------------------------------------------------------------------

def standardise_corpus(df):
    """Standardise variables used in downstream analysis."""

    df = df.copy()

    # Standardise boolean fields from the API.

    if "is_reply" in df.columns:

        if not pd.api.types.is_bool_dtype(
            df["is_reply"]
        ):

            df["is_reply"] = (
                df["is_reply"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "t",
                        "yes",
                        "y",
                    ]
                )
            )

    # Standardise comment timestamps.

    for column in [
        "published_at",
        "updated_at",
    ]:

        if column in df.columns:

            df[column] = (
                pd.to_datetime(
                    df[column],
                    errors="coerce",
                    utc=True
                )
            )

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_corpus(df):
    """Run basic checks on the final processed corpus."""

    required = [
        "comment_id",
        "author",
        "video_id",
        "text",
        "channel",
        "channel_type",
        "chan_ideo",
        "word_count",
        "is_english",
        "text_confirmation",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Final corpus is missing: "
            f"{missing}"
        )

    assert (
        df["comment_id"]
        .notna()
        .all()
    )

    assert (
        ~df["comment_id"]
        .duplicated()
        .any()
    )

    assert (
        df["author"]
        .notna()
        .all()
    )

    assert (
        df["video_id"]
        .notna()
        .all()
    )

    assert (
        df["text"]
        .notna()
        .all()
    )

    assert (
        df["is_english"]
        .all()
    )

    assert (
        df["text_confirmation"]
        .all()
    )

    assert set(
        df["channel_type"]
        .dropna()
        .unique()
    ) <= {
        "news",
        "influencer",
    }

    assert set(
        df["chan_ideo"]
        .dropna()
        .unique()
    ) <= {
        "left",
        "right",
    }

    # The three-comment threshold was applied separately
    # within each channel.

    channel_author_counts = (
        df.groupby(
            [
                "channel",
                "author",
            ]
        )
        .size()
    )

    assert (
        channel_author_counts
        >= MIN_COMMENTS_PER_AUTHOR
    ).all()

    expected_channels = set(
        CHANNELS.keys()
    )

    observed_channels = set(
        df["channel"]
        .unique()
    )

    assert (
        observed_channels
        == expected_channels
    ), (
        "Final channel set does not match "
        "the configured study design."
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(df):
    """Print a concise corpus summary."""

    print("\n" + "=" * 70)
    print("FINAL CORPUS")
    print("=" * 70)

    print(
        f"Comments: "
        f"{len(df):,}"
    )

    print(
        f"Unique users: "
        f"{df['author'].nunique():,}"
    )

    print(
        f"Unique videos: "
        f"{df['video_id'].nunique():,}"
    )

    print(
        f"Channels: "
        f"{df['channel'].nunique():,}"
    )

    print(
        "\nComments by channel:"
    )

    print(
        df["channel"]
        .value_counts()
        .to_string()
    )

    print(
        "\nComments by environment:"
    )

    print(
        df["channel_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nComments by ideology:"
    )

    print(
        df["chan_ideo"]
        .value_counts()
        .to_string()
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Construct the final cleaned Paper 2 corpus."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("PAPER 2 — CORPUS PREPARATION")
    print("=" * 70)

    corpus = build_corpus()

    corpus = standardise_corpus(
        corpus
    )

    validate_corpus(
        corpus
    )

    print_summary(
        corpus
    )

    corpus.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
