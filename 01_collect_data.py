"""
Collect YouTube video metadata and comments for Paper 2.

The script:

1. Retrieves all uploaded videos from each selected YouTube channel.
2. Retains videos published during the study year (2025).
3. Collects video metadata and engagement statistics.
4. Collects comments and available reply-chain information.
5. Saves channel-level video and comment files for later processing.

The YouTube API key is read from the YOUTUBE_API_KEY environment
variable and should never be stored directly in the repository.
"""

from pathlib import Path
from datetime import datetime
import os
import re

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DIR = Path("data/raw")

STUDY_YEAR = 2025

# Add the final channel IDs used in the study here.
#
# Example:
#
# CHANNELS = {
#     "Channel_Name": "UCxxxxxxxxxxxxxxxxxxxxxx",
# }
#
# Keeping channel names here also makes their provenance explicit.

CHANNELS = {
    # News channels
    "CHANNEL_NAME_1": "CHANNEL_ID_HERE",
    "CHANNEL_NAME_2": "CHANNEL_ID_HERE",

    # Political influencer channels
    "CHANNEL_NAME_3": "CHANNEL_ID_HERE",
    "CHANNEL_NAME_4": "CHANNEL_ID_HERE",
}


# ---------------------------------------------------------------------
# YouTube client
# ---------------------------------------------------------------------

def create_youtube_client():
    """Create an authenticated YouTube Data API client."""

    api_key = os.getenv(
        "YOUTUBE_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY environment variable "
            "has not been set."
        )

    return build(
        "youtube",
        "v3",
        developerKey=api_key
    )


# ---------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------

def get_channel_metadata(
    youtube,
    channel_id
):
    """Retrieve channel metadata and uploads playlist ID."""

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_id
    )

    response = request.execute()

    if not response.get("items"):
        raise ValueError(
            f"No channel found for ID: {channel_id}"
        )

    channel = response["items"][0]

    return {
        "channel_id":
            channel_id,

        "channel_title":
            channel["snippet"]["title"],

        "uploads_playlist":
            channel[
                "contentDetails"
            ][
                "relatedPlaylists"
            ][
                "uploads"
            ],
    }


# ---------------------------------------------------------------------
# Video IDs
# ---------------------------------------------------------------------

def get_video_ids(
    youtube,
    uploads_playlist
):
    """Retrieve every video ID from a channel's uploads playlist."""

    video_ids = []

    request = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist,
        maxResults=50
    )

    while request is not None:

        response = request.execute()

        for item in response.get(
            "items",
            []
        ):

            video_id = (
                item[
                    "contentDetails"
                ][
                    "videoId"
                ]
            )

            if video_id not in video_ids:
                video_ids.append(
                    video_id
                )

        request = (
            youtube.playlistItems()
            .list_next(
                request,
                response
            )
        )

    return video_ids


# ---------------------------------------------------------------------
# Video duration
# ---------------------------------------------------------------------

def iso_duration_to_hms(duration):
    """
    Convert a YouTube ISO-8601 duration to HH:MM:SS.

    Example:
        PT1H2M9S -> 01:02:09
    """

    hours = 0
    minutes = 0
    seconds = 0

    hour_match = re.search(
        r"(\d+)H",
        duration
    )

    minute_match = re.search(
        r"(\d+)M",
        duration
    )

    second_match = re.search(
        r"(\d+)S",
        duration
    )

    if hour_match:
        hours = int(
            hour_match.group(1)
        )

    if minute_match:
        minutes = int(
            minute_match.group(1)
        )

    if second_match:
        seconds = int(
            second_match.group(1)
        )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ---------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------

def get_video_metadata(
    youtube,
    video_ids,
    channel_name
):
    """
    Retrieve metadata for videos published during the study year.

    The API accepts a maximum of 50 video IDs per request.
    """

    records = []

    for start in range(
        0,
        len(video_ids),
        50
    ):

        batch = video_ids[
            start:start + 50
        ]

        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        )

        response = request.execute()

        for video in response.get(
            "items",
            []
        ):

            published = (
                video[
                    "snippet"
                ][
                    "publishedAt"
                ]
            )

            published_dt = (
                datetime.fromisoformat(
                    published.replace(
                        "Z",
                        "+00:00"
                    )
                )
            )

            # Paper 2 focuses on videos published
            # during the 2020 study period.

            if published_dt.year != STUDY_YEAR:
                continue

            statistics = video.get(
                "statistics",
                {}
            )

            duration = (
                video[
                    "contentDetails"
                ].get(
                    "duration",
                    "PT0S"
                )
            )

            records.append(
                {
                    "video_id":
                        video["id"],

                    "channel_name":
                        channel_name,

                    "title":
                        video[
                            "snippet"
                        ][
                            "title"
                        ],

                    "description":
                        video[
                            "snippet"
                        ].get(
                            "description",
                            ""
                        ),

                    "published_at":
                        published,

                    "video_length":
                        iso_duration_to_hms(
                            duration
                        ),

                    "view_count":
                        int(
                            statistics.get(
                                "viewCount",
                                0
                            )
                        ),

                    "like_count":
                        int(
                            statistics.get(
                                "likeCount",
                                0
                            )
                        ),

                    "comment_count":
                        int(
                            statistics.get(
                                "commentCount",
                                0
                            )
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------

def comment_record(
    comment,
    channel_name,
    is_reply=False,
    reply_to_comment_id=None
):
    """Convert one YouTube comment into a consistent record."""

    snippet = comment[
        "snippet"
    ]

    return {
        "comment_id":
            comment["id"],

        "reply_to_comment_id":
            reply_to_comment_id,

        "video_id":
            snippet["videoId"],

        "channel_name":
            channel_name,

        "author":
            snippet.get(
                "authorDisplayName"
            ),

        "published_at":
            snippet.get(
                "publishedAt"
            ),

        "updated_at":
            snippet.get(
                "updatedAt"
            ),

        "like_count":
            snippet.get(
                "likeCount",
                0
            ),

        "text":
            snippet.get(
                "textOriginal",
                ""
            ),

        "is_reply":
            is_reply,
    }


def get_video_comments(
    youtube,
    video_id,
    channel_name
):
    """
    Retrieve top-level comments and available replies for one video.

    Replies returned with commentThreads are retained so that reply
    relationships remain available for later analysis.
    """

    comments = []

    try:

        request = (
            youtube.commentThreads()
            .list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText"
            )
        )

        while request is not None:

            response = request.execute()

            for thread in response.get(
                "items",
                []
            ):

                top_comment = (
                    thread[
                        "snippet"
                    ][
                        "topLevelComment"
                    ]
                )

                top_comment_id = (
                    top_comment["id"]
                )

                comments.append(
                    comment_record(
                        comment=top_comment,
                        channel_name=channel_name,
                        is_reply=False,
                        reply_to_comment_id=None
                    )
                )

                # Retain replies supplied with the
                # comment thread response.

                replies = (
                    thread.get(
                        "replies",
                        {}
                    ).get(
                        "comments",
                        []
                    )
                )

                for reply in replies:

                    comments.append(
                        comment_record(
                            comment=reply,
                            channel_name=channel_name,
                            is_reply=True,
                            reply_to_comment_id=(
                                top_comment_id
                            )
                        )
                    )

            request = (
                youtube.commentThreads()
                .list_next(
                    request,
                    response
                )
            )

    except HttpError as error:

        # Some videos may have comments disabled,
        # been removed, or otherwise be inaccessible.

        print(
            f"Could not retrieve comments "
            f"for {video_id}: {error}"
        )

    return comments


# ---------------------------------------------------------------------
# Channel comments
# ---------------------------------------------------------------------

def get_channel_comments(
    youtube,
    video_ids,
    channel_name
):
    """Collect comments across all retained videos for one channel."""

    records = []

    total = len(
        video_ids
    )

    for number, video_id in enumerate(
        video_ids,
        start=1
    ):

        print(
            f"[{channel_name}] "
            f"Comments {number}/{total}: "
            f"{video_id}"
        )

        records.extend(
            get_video_comments(
                youtube=youtube,
                video_id=video_id,
                channel_name=channel_name
            )
        )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Collect one channel
# ---------------------------------------------------------------------

def collect_channel(
    youtube,
    channel_name,
    channel_id
):
    """Collect video metadata and comments for one study channel."""

    print("\n" + "=" * 70)
    print(
        f"COLLECTING: {channel_name}"
    )
    print("=" * 70)

    channel = get_channel_metadata(
        youtube,
        channel_id
    )

    print(
        f"YouTube channel: "
        f"{channel['channel_title']}"
    )

    video_ids = get_video_ids(
        youtube,
        channel[
            "uploads_playlist"
        ]
    )

    print(
        f"Uploaded videos found: "
        f"{len(video_ids):,}"
    )

    # -------------------------------------------------------------
    # Video metadata
    # -------------------------------------------------------------

    videos = get_video_metadata(
        youtube=youtube,
        video_ids=video_ids,
        channel_name=channel_name
    )

    print(
        f"{STUDY_YEAR} videos retained: "
        f"{len(videos):,}"
    )

    if videos.empty:

        print(
            "No study-period videos found."
        )

        return

    # Only collect comments from videos that
    # survived the study-year restriction.

    retained_video_ids = (
        videos[
            "video_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    # -------------------------------------------------------------
    # Comments
    # -------------------------------------------------------------

    comments = get_channel_comments(
        youtube=youtube,
        video_ids=retained_video_ids,
        channel_name=channel_name
    )

    print(
        f"Comments collected: "
        f"{len(comments):,}"
    )

    # -------------------------------------------------------------
    # Save raw channel data
    # -------------------------------------------------------------

    safe_name = (
        channel_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    videos.to_csv(
        RAW_DIR
        / f"{safe_name}_videos.csv",
        encoding="utf-8-sig",
        index=False
    )

    comments.to_csv(
        RAW_DIR
        / f"{safe_name}_comments.csv",
        encoding="utf-8-sig",
        index=False
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Collect the raw Paper 2 YouTube dataset."""

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    youtube = (
        create_youtube_client()
    )

    print("=" * 70)
    print("PAPER 2 — YOUTUBE DATA COLLECTION")
    print("=" * 70)

    print(
        f"Study year: {STUDY_YEAR}"
    )

    print(
        f"Channels: {len(CHANNELS)}"
    )

    for channel_name, channel_id in (
        CHANNELS.items()
    ):

        collect_channel(
            youtube=youtube,
            channel_name=channel_name,
            channel_id=channel_id
        )

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
