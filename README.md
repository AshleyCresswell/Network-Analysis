# Affective Sorting in Online Political Networks

This repository contains the computational pipeline for a study of **affective sorting in online political networks**.

Using YouTube comments from political news and influencer channels, the study examines whether users with similar emotional patterns are more likely to interact within the same political communities, and whether these patterns differ across ideological and media environments.

## Research Overview

Research on social sorting typically focuses on ideological similarity. However, political communities may also be structured around the emotions that individuals express.

This project examines **affective sorting** by combining emotion analysis with network analysis. Users are represented by their patterns of emotional expression across eight emotions: **anger, anticipation, disgust, fear, joy, sadness, surprise, and trust**.

Affective sorting is then measured by comparing the emotional similarity of users who occupy neighbouring positions within a co-commenting network against the similarity expected between randomly selected users.

The analysis additionally examines whether affective sorting differs between **left- and right-leaning political environments** and between **political news and influencer channels**, alongside patterns of emotional contagion.

## Data

The analysis uses YouTube comments collected during **2025** from a sample of political news and political influencer channels representing both left- and right-leaning political environments.

Comments are cleaned and filtered before emotion scoring and network construction. The final analytical corpus retains users with sufficient commenting activity to support reliable estimates of emotional expression and network position.

Raw YouTube comments are not redistributed through this repository.

## Methodology

The project combines emotion analysis, social network analysis, and statistical modelling.

### 1. Emotion Measurement

Comments are scored across eight emotions:

- **Anger**
- **Anticipation**
- **Disgust**
- **Fear**
- **Joy**
- **Sadness**
- **Surprise**
- **Trust**

User-level affective profiles are constructed by aggregating emotional expression across comments.

### 2. Network Construction

A co-commenting (Bipartite) network is constructed from shared engagement with political videos. Users are connected when they comment on the same videos, with stronger connections representing greater shared engagement.

Affective sorting is measured by comparing the emotional similarity between a user and their network neighbours with a random-user baseline.

### 3. Analysis

Statistical models (OLS Regressions) examine whether affective sorting differs across ideological and media environments. Additional analyses examine **emotion-specific sorting** and whether emotional expression in comments is associated with the emotional content of preceding comments within the same discussion (Time-lagged Analysis).

## Repository Structure

```text
src/
├── 01_collect_data.py
├── 02_prepare_corpus.py
├── 03_score_emotions.py
├── 04_build_network.py
├── 05_measure_affective_sorting.py
├── 06_model_affective_sorting.py
└── 07_emotion_analysis.py
```

The pipeline covers:

**Data collection → Corpus construction → Emotion scoring → Network construction → Affective sorting measurement → Statistical modelling → Emotion-specific analysis**

## Key Findings

Affective sorting varies across political and media environments, while emotion-specific analyses demonstrate that different emotions contribute differently to these patterns. The results largely confirm the presence of affective sorting as users display higher emotional similarity to their interaction neighbours than expected under random mixing. Right-lean influencer spaces also appear to reduce negative emotional sorting, contrary to expectations.


## Reproduction

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Scripts are designed to be run sequentially from `01_collect_data.py` through `07_emotion_analysis.py`.

YouTube collection requires a YouTube Data API key supplied through an environment variable. Large raw datasets and intermediate network files are not included in the repository.

## Citation

This repository accompanies ongoing doctoral research on affective polarisation, emotional expression, and online political networks. A full citation will be added following publication.
