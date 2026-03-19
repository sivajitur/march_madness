"""
Data cleaning and preprocessing for March Madness prediction.

Normalizes team names, handles missing values, and reconstructs
historical tournament matchups from bracket seedings.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEAM_COL, YEAR_COL, SEED_COL, POSTSEASON_COL, CONF_COL,
    WINS_COL, GAMES_COL, ADJOE_COL, ADJDE_COL,
    DIFFERENTIAL_FEATURES, POSTSEASON_ROUNDS, HISTORICAL_YEARS,
)
from src.utils import normalize_team_name, seed_to_int, win_percentage


# Standard bracket matchups: in R64 seed X plays seed Y
_R64_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def clean_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a team-stats DataFrame.

    - Normalise team names
    - Convert seeds to integers
    - Compute win percentage
    - Impute missing numeric values with conference averages
    """
    df = df.copy()

    # Normalise team names
    df[TEAM_COL] = df[TEAM_COL].apply(normalize_team_name)

    # Integer seeds (0 = no seed / not in tourney)
    if SEED_COL in df.columns:
        df[SEED_COL] = df[SEED_COL].apply(seed_to_int)

    # Win percentage
    if WINS_COL in df.columns and GAMES_COL in df.columns:
        df["WIN_PCT"] = df.apply(
            lambda r: win_percentage(r[WINS_COL], r[GAMES_COL]), axis=1
        )

    # Impute missing numeric cols with conference mean, then global mean
    numeric_cols = [c for c in DIFFERENTIAL_FEATURES if c in df.columns]
    if CONF_COL in df.columns:
        for col in numeric_cols:
            if df[col].isna().any():
                conf_mean = df.groupby(CONF_COL)[col].transform("mean")
                df[col] = df[col].fillna(conf_mean)
                df[col] = df[col].fillna(df[col].mean())

    return df


def get_tournament_teams(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return only seeded (tournament) teams for a given year."""
    mask = df[SEED_COL] > 0
    if YEAR_COL in df.columns:
        mask = mask & (df[YEAR_COL] == year)
    return df[mask].copy()


def build_historical_matchups(df: pd.DataFrame) -> list[tuple]:
    """
    Reconstruct tournament game matchups from bracket seedings
    and POSTSEASON results.

    For each year, pairs teams by seed matchup in each round.
    The winner is the team that advanced further (higher
    POSTSEASON_ROUNDS value).

    Returns:
        List of (year, team_a, team_b, winner) tuples.
    """
    matchups = []

    for year in HISTORICAL_YEARS:
        if YEAR_COL in df.columns:
            yr_df = df[(df[YEAR_COL] == year) & (df[SEED_COL] > 0)].copy()
        else:
            continue

        if yr_df.empty:
            continue

        # Map postseason label → numeric depth
        yr_df["_round_depth"] = yr_df[POSTSEASON_COL].map(POSTSEASON_ROUNDS).fillna(-2)

        # Group by seed for pairing
        seed_groups = yr_df.groupby(SEED_COL)

        for seed_a, seed_b in _R64_MATCHUPS:
            teams_a = yr_df[yr_df[SEED_COL] == seed_a]
            teams_b = yr_df[yr_df[SEED_COL] == seed_b]

            # Pair them up (same conference region isn't in data, so pair by row)
            for (_, ta), (_, tb) in zip(teams_a.iterrows(), teams_b.iterrows()):
                depth_a = ta["_round_depth"]
                depth_b = tb["_round_depth"]

                if depth_a == depth_b:
                    # Tie-break: lower seed wins (this is a heuristic)
                    winner = ta[TEAM_COL] if seed_a < seed_b else tb[TEAM_COL]
                elif depth_a > depth_b:
                    winner = ta[TEAM_COL]
                else:
                    winner = tb[TEAM_COL]

                matchups.append((year, ta[TEAM_COL], tb[TEAM_COL], winner))

    return matchups


def merge_and_clean(historical_df: pd.DataFrame,
                    current_df: pd.DataFrame | None = None) -> tuple:
    """
    Full cleaning pipeline.

    Returns:
        (cleaned_historical, cleaned_current_or_None)
    """
    cleaned_hist = clean_team_stats(historical_df)
    cleaned_curr = clean_team_stats(current_df) if current_df is not None else None
    return cleaned_hist, cleaned_curr
