"""
Score emotions in the final Paper 2 comment corpus.

The script:

1. Loads the cleaned Paper 2 comment corpus.
2. Loads the NRC Emotion Intensity Lexicon.
3. Loads the required GloVe word embeddings.
4. Tokenises and prepares comment text for emotion scoring.
5. Measures semantic similarity between comment tokens and
   emotion-associated terms using GloVe embeddings.
6. Weights semantic similarity by NRC emotion intensity scores.
7. Calculates scores for eight emotions: anger, anticipation,
   disgust, fear, joy, sadness, surprise, and trust.
8. Validates and saves the emotion-scored comment corpus.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "processed" / "paper2_2025_corpus.csv"

NRC_FILE = DATA_DIR / "external" / "NRC-Emotion-Intensity-Lexicon.txt"
GLOVE_FILE = DATA_DIR / "external" / "glove.6B.100d.txt"

OUTPUT_FILE = DATA_DIR / "processed" / "paper2_2025_emotions.csv"


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

TEXT_COLUMN = "text"

STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------
# Load NRC Emotion Intensity Lexicon
# ---------------------------------------------------------------------

def load_nrc_lexicon(path):
    """
    Load NRC emotion-intensity scores.

    Returns
    -------
    dict
        Mapping from emotion -> {word: intensity}.
    set
        All words required from the GloVe vocabulary.
    """

    emotion_lexicon = {emotion: {} for emotion in EMOTIONS}
    required_words = set()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")

            if len(parts) != 3:
                continue

            word, emotion, intensity = parts

            if emotion not in EMOTIONS:
                continue

            try:
                intensity = float(intensity)
            except ValueError:
                continue

            word = word.lower()

            emotion_lexicon[emotion][word] = intensity
            required_words.add(word)

    return emotion_lexicon, required_words


# ---------------------------------------------------------------------
# Load relevant GloVe vectors
# ---------------------------------------------------------------------

def load_glove_vectors(path, required_words):
    """
    Load only GloVe vectors required by the NRC lexicon.

    Restricting the vocabulary avoids loading the complete embedding
    file into memory.
    """

    embeddings = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            values = line.rstrip().split(" ")

            word = values[0]

            if word not in required_words:
                continue

            try:
                vector = np.asarray(values[1:], dtype=np.float32)
            except ValueError:
                continue

            embeddings[word] = vector

    return embeddings


# ---------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------

def tokenize_comment(text):
    """
    Prepare comment text for emotion scoring.

    URLs, user mentions and hashtag markers are removed before
    tokenisation. English stopwords are excluded.
    """

    if pd.isna(text):
        return []

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)

    tokens = word_tokenize(text)

    return [
        token
        for token in tokens
        if token.isalpha()
        and token not in STOP_WORDS
    ]


# ---------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two word vectors."""

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


# ---------------------------------------------------------------------
# Prepare NRC emotion vectors
# ---------------------------------------------------------------------

def prepare_emotion_vectors(emotion_lexicon, embeddings):
    """
    Store NRC words, vectors and intensity weights for each emotion.
    """

    emotion_vectors = {}

    for emotion in EMOTIONS:

        words = []
        vectors = []
        intensities = []

        for word, intensity in emotion_lexicon[emotion].items():

            if word not in embeddings:
                continue

            words.append(word)
            vectors.append(embeddings[word])
            intensities.append(intensity)

        emotion_vectors[emotion] = {
            "words": words,
            "vectors": np.asarray(vectors),
            "intensities": np.asarray(intensities),
        }

    return emotion_vectors


# ---------------------------------------------------------------------
# Emotion scoring
# ---------------------------------------------------------------------

def score_comment(text, embeddings, emotion_vectors):
    """
    Calculate NRC/GloVe emotion scores for one comment.

    For each comment token and emotion, semantic similarity with NRC
    emotion terms is weighted by NRC intensity. The strongest weighted
    association is retained for that emotion.
    """

    tokens = tokenize_comment(text)

    scores = {emotion: 0.0 for emotion in EMOTIONS}

    token_vectors = [
        embeddings[token]
        for token in tokens
        if token in embeddings
    ]

    if not token_vectors:
        return scores

    for emotion in EMOTIONS:

        nrc_vectors = emotion_vectors[emotion]["vectors"]
        intensities = emotion_vectors[emotion]["intensities"]

        if len(nrc_vectors) == 0:
            continue

        best_score = 0.0

        for token_vector in token_vectors:

            token_norm = np.linalg.norm(token_vector)
            nrc_norms = np.linalg.norm(nrc_vectors, axis=1)

            denominator = nrc_norms * token_norm

            similarities = np.divide(
                nrc_vectors @ token_vector,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator != 0,
            )

            weighted_scores = similarities * intensities

            best_score = max(
                best_score,
                float(np.max(weighted_scores)),
            )

        scores[emotion] = best_score

    return scores


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PAPER 2 — EMOTION SCORING")
    print("=" * 70)

    # Load cleaned corpus
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    assert TEXT_COLUMN in df.columns, (
        f"Missing required column: {TEXT_COLUMN}"
    )

    print(f"\nComments loaded: {len(df):,}")

    # Load NRC lexicon
    emotion_lexicon, required_words = load_nrc_lexicon(NRC_FILE)

    print("\nNRC vocabulary by emotion:")

    for emotion in EMOTIONS:
        print(
            f"{emotion:<14}: "
            f"{len(emotion_lexicon[emotion]):,}"
        )

    # Load GloVe vectors required for scoring
    print("\nLoading GloVe vectors...")

    embeddings = load_glove_vectors(
        GLOVE_FILE,
        required_words,
    )

    print(f"GloVe vectors loaded: {len(embeddings):,}")

    emotion_vectors = prepare_emotion_vectors(
        emotion_lexicon,
        embeddings,
    )

    # Score comments
    print("\nScoring comments...")

    scores = df[TEXT_COLUMN].apply(
        lambda text: score_comment(
            text,
            embeddings,
            emotion_vectors,
        )
    )

    emotion_df = pd.DataFrame(scores.tolist())

    df = pd.concat(
        [
            df.reset_index(drop=True),
            emotion_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Basic validation
    assert all(
        emotion in df.columns
        for emotion in EMOTIONS
    )

    assert not df[EMOTIONS].isna().any().any()

    print("\nMean emotion scores:")

    print(
        df[EMOTIONS]
        .mean()
        .sort_values(ascending=False)
        .round(4)
    )

    # Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
