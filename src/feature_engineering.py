"""
Feature engineering for March Madness matchup prediction.

Transforms per-team season statistics into per-matchup differential
features that the ML model uses for prediction.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DIFFERENTIAL_FEATURES, DIFF_COL_NAMES, ALL_MODEL_FEATURES, EXTRA_FEATURES,
    TEAM_COL, YEAR_COL, SEED_COL, ADJOE_COL, ADJDE_COL, WINS_COL, GAMES_COL,
)
from src.utils import seed_to_int, win_percentage, net_efficiency_margin


def compute_matchup_features(team_a: dict, team_b: dict) -> dict:
    """
    Compute differential features for a single matchup.

    Args:
        team_a: Dict-like row of team A's season stats.
        team_b: Dict-like row of team B's season stats.

    Returns:
        Dict mapping feature names to values.
    """
    features = {}

    # Stat differentials (team_a - team_b)
    for stat, diff_name in zip(DIFFERENTIAL_FEATURES, DIFF_COL_NAMES):
        val_a = float(team_a.get(stat, 0) or 0)
        val_b = float(team_b.get(stat, 0) or 0)
        features[diff_name] = val_a - val_b

    # Seed features
    seed_a = seed_to_int(team_a.get(SEED_COL, 0))
    seed_b = seed_to_int(team_b.get(SEED_COL, 0))
    features["seed_diff"] = seed_a - seed_b
    features["seed_product"] = seed_a * seed_b

    # Net efficiency margin differential
    em_a = net_efficiency_margin(
        float(team_a.get(ADJOE_COL, 0) or 0),
        float(team_a.get(ADJDE_COL, 0) or 0),
    )
    em_b = net_efficiency_margin(
        float(team_b.get(ADJOE_COL, 0) or 0),
        float(team_b.get(ADJDE_COL, 0) or 0),
    )
    features["adj_em_diff"] = em_a - em_b

    # Win percentage differential
    wp_a = team_a.get("WIN_PCT", 0) or 0
    wp_b = team_b.get("WIN_PCT", 0) or 0
    if wp_a == 0 and WINS_COL in team_a and GAMES_COL in team_a:
        wp_a = win_percentage(int(team_a[WINS_COL] or 0), int(team_a[GAMES_COL] or 1))
    if wp_b == 0 and WINS_COL in team_b and GAMES_COL in team_b:
        wp_b = win_percentage(int(team_b[WINS_COL] or 0), int(team_b[GAMES_COL] or 1))
    features["win_pct_diff"] = float(wp_a) - float(wp_b)

    return features


def build_training_matchups(team_stats_df: pd.DataFrame,
                            matchups: list[tuple]) -> pd.DataFrame:
    """
    Build the training dataset from historical matchups.

    For each (year, team_a, team_b, winner) tuple, computes
    differential features and a binary label.  The dataset is
    doubled by also adding the flipped matchup (swap A/B, negate
    diffs, flip label) for balance.

    Args:
        team_stats_df: Full team stats DataFrame (all years, cleaned).
        matchups: List of (year, team_a, team_b, winner) tuples.

    Returns:
        DataFrame with columns = ALL_MODEL_FEATURES + ["label", "year"].
    """
    rows = []

    for year, team_a_name, team_b_name, winner in matchups:
        # Look up team stats
        if YEAR_COL in team_stats_df.columns:
            mask_a = (team_stats_df[TEAM_COL] == team_a_name) & (team_stats_df[YEAR_COL] == year)
            mask_b = (team_stats_df[TEAM_COL] == team_b_name) & (team_stats_df[YEAR_COL] == year)
        else:
            mask_a = team_stats_df[TEAM_COL] == team_a_name
            mask_b = team_stats_df[TEAM_COL] == team_b_name

        rows_a = team_stats_df[mask_a]
        rows_b = team_stats_df[mask_b]

        if rows_a.empty or rows_b.empty:
            continue

        ta = rows_a.iloc[0].to_dict()
        tb = rows_b.iloc[0].to_dict()

        features = compute_matchup_features(ta, tb)
        label = 1 if winner == team_a_name else 0

        row = {**features, "label": label, "year": year}
        rows.append(row)

        # Flipped matchup (swap A/B, negate diffs, flip label)
        flipped = {}
        for col in DIFF_COL_NAMES + EXTRA_FEATURES:
            flipped[col] = -features.get(col, 0)
        # seed_product is symmetric, keep positive
        flipped["seed_product"] = features["seed_product"]
        flipped["label"] = 1 - label
        flipped["year"] = year
        rows.append(flipped)

    df = pd.DataFrame(rows)

    # Ensure column order
    cols = ALL_MODEL_FEATURES + ["label", "year"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


def prepare_prediction_features(team_a_stats: dict,
                                 team_b_stats: dict) -> pd.DataFrame:
    """
    Prepare a single-row feature DataFrame for prediction (no label).

    Args:
        team_a_stats: Dict of team A's season stats.
        team_b_stats: Dict of team B's season stats.

    Returns:
        Single-row DataFrame with columns = ALL_MODEL_FEATURES.
    """
    features = compute_matchup_features(team_a_stats, team_b_stats)
    df = pd.DataFrame([features])
    # Ensure correct column order
    for c in ALL_MODEL_FEATURES:
        if c not in df.columns:
            df[c] = 0
    return df[ALL_MODEL_FEATURES]
