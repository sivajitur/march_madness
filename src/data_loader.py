"""
Data loading utilities for March Madness prediction.

Downloads datasets from Kaggle and loads team statistics CSVs.
"""

import os
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, CBB_HISTORICAL_FILE, CBB_2026_FILE, CBB_2020_FILE,
    KAGGLE_CBB_DATASET, TEAM_COL, YEAR_COL, SEED_COL, POSTSEASON_COL,
    HISTORICAL_YEARS, CURRENT_YEAR,
)


def download_kaggle_dataset(dataset_id: str = KAGGLE_CBB_DATASET,
                            output_dir: str = RAW_DATA_DIR) -> None:
    """
    Download a Kaggle dataset to *output_dir*.

    Tries kagglehub first, then falls back to the kaggle CLI.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_id)
        # Copy files to our raw data dir
        import shutil
        for f in os.listdir(path):
            src = os.path.join(path, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"Downloaded {dataset_id} → {output_dir}")
        return
    except ImportError:
        pass

    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id,
             "--unzip", "-p", output_dir],
            check=True,
        )
        print(f"Downloaded {dataset_id} → {output_dir}")
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    print(
        f"Could not download dataset '{dataset_id}'.\n"
        f"Please download it manually from:\n"
        f"  https://www.kaggle.com/datasets/{dataset_id}\n"
        f"and place the CSV files in:\n"
        f"  {output_dir}"
    )


def load_team_stats(year: int | None = None) -> pd.DataFrame:
    """
    Load team statistics for a given year (or all years).

    Args:
        year: Specific year to load.  If *None*, loads the full
              historical file (cbb.csv).  Use ``CURRENT_YEAR``
              (2026) for the current-season file.

    Returns:
        DataFrame of team stats.
    """
    if year == CURRENT_YEAR:
        path = CBB_2026_FILE
    elif year == 2020:
        path = CBB_2020_FILE
    else:
        path = CBB_HISTORICAL_FILE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run `python -c \"from src.data_loader import download_kaggle_dataset; "
            f"download_kaggle_dataset()\"` first, or download manually from Kaggle."
        )

    df = pd.read_csv(path)

    # If requesting a specific historical year, filter
    if year is not None and year != CURRENT_YEAR and YEAR_COL in df.columns:
        df = df[df[YEAR_COL] == year].copy()

    return df


def load_all_historical_stats() -> pd.DataFrame:
    """Load the full historical dataset (cbb.csv, 2013-2025)."""
    return load_team_stats(year=None)


def load_tournament_results() -> pd.DataFrame:
    """
    Load historical stats filtered to tournament teams only.

    Returns a DataFrame where every row is a team that participated
    in the NCAA tournament (has a non-null POSTSEASON value).
    """
    df = load_all_historical_stats()

    # Filter to tournament years and teams with postseason data
    df = df[df[YEAR_COL].isin(HISTORICAL_YEARS)].copy()
    df = df[df[POSTSEASON_COL].notna()].copy()

    return df
