from __future__ import annotations

from typing import Any

import streamlit as st

FRUITS = ("Apple", "Banana", "Grape", "Mango", "Strawberry")


def _default_stats() -> dict[str, int]:
    return {
        "known_scans": 0,
        "unknown_scans": 0,
        "quiz_attempts": 0,
        "quiz_correct": 0,
        "activities_opened": 0,
        "adaptive_correct": 0,
    }


def _default_fruit_stats() -> dict[str, dict[str, Any]]:
    return {
        fruit: {
            "scans": 0,
            "quiz_attempts": 0,
            "quiz_correct": 0,
            "activities": [],
            "adaptive_correct": 0,
        }
        for fruit in FRUITS
    }


def ensure_progress_state() -> None:
    if "progress_stats" not in st.session_state:
        st.session_state.progress_stats = _default_stats()
    if "fruit_progress" not in st.session_state:
        st.session_state.fruit_progress = _default_fruit_stats()
    if "progress_seen_scans" not in st.session_state:
        st.session_state.progress_seen_scans = set()
    if "progress_seen_events" not in st.session_state:
        st.session_state.progress_seen_events = set()


def _fruit_bucket(fruit_name: str) -> dict[str, Any]:
    ensure_progress_state()
    if fruit_name not in st.session_state.fruit_progress:
        st.session_state.fruit_progress[fruit_name] = {
            "scans": 0,
            "quiz_attempts": 0,
            "quiz_correct": 0,
            "activities": [],
            "adaptive_correct": 0,
        }
    return st.session_state.fruit_progress[fruit_name]


def record_scan(scan_token: str, fruit_name: str | None, verified: bool) -> None:
    ensure_progress_state()
    event = f"scan:{scan_token}"
    if event in st.session_state.progress_seen_scans:
        return
    st.session_state.progress_seen_scans.add(event)

    if verified and fruit_name in FRUITS:
        st.session_state.progress_stats["known_scans"] += 1
        _fruit_bucket(fruit_name)["scans"] += 1
    else:
        st.session_state.progress_stats["unknown_scans"] += 1


def record_activity_open(fruit_name: str, activity: str) -> None:
    ensure_progress_state()
    event = f"activity:{fruit_name}:{activity}"
    if event not in st.session_state.progress_seen_events:
        st.session_state.progress_seen_events.add(event)
        st.session_state.progress_stats["activities_opened"] += 1

    bucket = _fruit_bucket(fruit_name)
    if activity not in bucket["activities"]:
        bucket["activities"].append(activity)


def record_quiz_attempt(
    fruit_name: str,
    correct: bool,
    event_key: str | None = None,
) -> None:
    ensure_progress_state()
    if event_key:
        event = f"quiz:{event_key}"
        if event in st.session_state.progress_seen_events:
            return
        st.session_state.progress_seen_events.add(event)

    st.session_state.progress_stats["quiz_attempts"] += 1
    bucket = _fruit_bucket(fruit_name)
    bucket["quiz_attempts"] += 1
    if correct:
        st.session_state.progress_stats["quiz_correct"] += 1
        bucket["quiz_correct"] += 1


def record_adaptive_success(fruit_name: str, level: int) -> bool:
    ensure_progress_state()
    event = f"adaptive:{fruit_name}:{level}"
    if event in st.session_state.progress_seen_events:
        return False

    st.session_state.progress_seen_events.add(event)
    st.session_state.progress_stats["adaptive_correct"] += 1
    _fruit_bucket(fruit_name)["adaptive_correct"] += 1

    if "reward_stars" not in st.session_state:
        st.session_state.reward_stars = 0
    if "rewarded_events" not in st.session_state:
        st.session_state.rewarded_events = set()

    star_event = f"adaptive-star:{fruit_name}:{level}"
    if star_event not in st.session_state.rewarded_events:
        st.session_state.rewarded_events.add(star_event)
        st.session_state.reward_stars += 1
    return True


def learning_level(fruit_name: str) -> int:
    bucket = _fruit_bucket(fruit_name)
    score = int(bucket["quiz_correct"]) + int(bucket["adaptive_correct"])
    if score >= 8:
        return 6
    if score >= 6:
        return 5
    if score >= 4:
        return 4
    if score >= 3:
        return 3
    if score >= 1:
        return 2
    return 1


def progress_snapshot() -> dict[str, Any]:
    ensure_progress_state()
    unlocked = st.session_state.get("unlocked_fruits", set())
    return {
        "unlocked_fruits": sorted(str(item) for item in unlocked if item in FRUITS),
        "reward_stars": int(st.session_state.get("reward_stars", 0)),
        "progress_stats": dict(st.session_state.progress_stats),
        "fruit_progress": {
            fruit: {
                "scans": int(data.get("scans", 0)),
                "quiz_attempts": int(data.get("quiz_attempts", 0)),
                "quiz_correct": int(data.get("quiz_correct", 0)),
                "activities": list(data.get("activities", [])),
                "adaptive_correct": int(data.get("adaptive_correct", 0)),
            }
            for fruit, data in st.session_state.fruit_progress.items()
            if fruit in FRUITS
        },
    }
