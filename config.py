"""
Global configuration for the March Madness bracket prediction system.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
BRACKET_DIR = os.path.join(DATA_DIR, "bracket")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ── Data Files ────────────────────────────────────────────────────────────────
# Primary dataset: Andrew Sundberg's College Basketball Dataset (Kaggle)
CBB_HISTORICAL_FILE = os.path.join(RAW_DATA_DIR, "cbb.csv")       # 2013-2025
CBB_2026_FILE = os.path.join(RAW_DATA_DIR, "cbb26.csv")           # 2026 season
CBB_2020_FILE = os.path.join(RAW_DATA_DIR, "cbb20.csv")           # 2020 (no tourney)

# Processed outputs
TRAINING_MATCHUPS_FILE = os.path.join(PROCESSED_DATA_DIR, "training_matchups.csv")
TEAM_STATS_2026_FILE = os.path.join(PROCESSED_DATA_DIR, "team_stats_2026.csv")

# Bracket definition
BRACKET_2026_FILE = os.path.join(BRACKET_DIR, "bracket_2026.json")

# Model artifacts
TRAINED_MODEL_FILE = os.path.join(MODELS_DIR, "trained_model.pkl")
MODEL_METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

# ── Kaggle Dataset Identifiers ────────────────────────────────────────────────
KAGGLE_CBB_DATASET = "andrewsundberg/college-basketball-dataset"

# ── Tournament Years ──────────────────────────────────────────────────────────
# Years with tournament data (excluding 2020 - COVID cancellation)
HISTORICAL_YEARS = [y for y in range(2013, 2026) if y != 2020]
CURRENT_YEAR = 2026

# ── Column Names (from Andrew Sundberg's dataset) ─────────────────────────────
# These are the column names in cbb.csv / cbb26.csv
TEAM_COL = "TEAM"
YEAR_COL = "YEAR"
SEED_COL = "SEED"
POSTSEASON_COL = "POSTSEASON"
CONF_COL = "CONF"
WINS_COL = "W"
GAMES_COL = "G"

# Efficiency metrics
ADJOE_COL = "ADJOE"      # Adjusted Offensive Efficiency
ADJDE_COL = "ADJDE"      # Adjusted Defensive Efficiency
BARTHAG_COL = "BARTHAG"  # Power Rating (Barttorvik)
ADJ_T_COL = "ADJ_T"      # Adjusted Tempo
WAB_COL = "WAB"           # Wins Above Bubble

# Four Factors (offense)
EFG_O_COL = "EFG_O"      # Effective FG% (shot)
TOR_COL = "TOR"           # Turnover Rate (committed)
ORB_COL = "ORB"           # Offensive Rebound Rate
FTR_COL = "FTR"           # Free Throw Rate (shot)

# Four Factors (defense)
EFG_D_COL = "EFG_D"      # Effective FG% (allowed)
TORD_COL = "TORD"         # Turnover Rate (forced)
DRB_COL = "DRB"           # Defensive Rebound Rate
FTRD_COL = "FTRD"         # Free Throw Rate (allowed)

# Shooting splits
TWO_P_O_COL = "2P_O"     # 2-point FG% (shot)
TWO_P_D_COL = "2P_D"     # 2-point FG% (allowed)
THREE_P_O_COL = "3P_O"   # 3-point FG% (shot)
THREE_P_D_COL = "3P_D"   # 3-point FG% (allowed)

# ── Feature Lists ─────────────────────────────────────────────────────────────
# Stats used to compute matchup differentials
DIFFERENTIAL_FEATURES = [
    ADJOE_COL, ADJDE_COL, BARTHAG_COL, ADJ_T_COL, WAB_COL,
    EFG_O_COL, EFG_D_COL, TOR_COL, TORD_COL, ORB_COL, DRB_COL,
    FTR_COL, FTRD_COL, TWO_P_O_COL, TWO_P_D_COL, THREE_P_O_COL, THREE_P_D_COL,
]

# Differential column names (auto-generated from DIFFERENTIAL_FEATURES)
DIFF_COL_NAMES = [f"{col}_diff" for col in DIFFERENTIAL_FEATURES]

# Additional engineered features (not simple differentials)
EXTRA_FEATURES = [
    "seed_diff",       # seed_a - seed_b (negative = team A is favored)
    "seed_product",    # seed_a * seed_b (captures matchup magnitude)
    "adj_em_diff",     # (ADJOE - ADJDE) diff between teams (net efficiency margin)
    "win_pct_diff",    # win percentage differential
]

# All features fed into the model
ALL_MODEL_FEATURES = DIFF_COL_NAMES + EXTRA_FEATURES

# ── Postseason Labels (from dataset) ──────────────────────────────────────────
# Maps postseason column values to tournament round depth
POSTSEASON_ROUNDS = {
    "Champions": 6,
    "2ND": 5,
    "F4": 4,
    "E8": 3,
    "S16": 2,
    "R32": 1,
    "R64": 0,
    "R68": -1,  # First Four losers
}

# ── ESPN Scoring ──────────────────────────────────────────────────────────────
# Points awarded per correct pick by round
ESPN_POINTS = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "Championship": 320,
}

# ── Model Hyperparameters ─────────────────────────────────────────────────────
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

LOGISTIC_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

# ── Monte Carlo Settings ─────────────────────────────────────────────────────
DEFAULT_SIMULATIONS = 10_000
