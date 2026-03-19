#!/usr/bin/env python3
"""
March Madness Bracket Prediction System

Usage:
    python main.py --backtest                                  # LOTO-CV
    python main.py --predict                                   # Deterministic
    python main.py --predict --strategy upset_aware            # With randomness
    python main.py --predict --strategy monte_carlo            # ESPN-optimized
    python main.py --predict --strategy monte_carlo --simulations 50000
"""

import argparse
import os
import sys

from config import (
    ALL_MODEL_FEATURES, OUTPUT_DIR, TRAINED_MODEL_FILE,
    BRACKET_2026_FILE, CURRENT_YEAR, DEFAULT_SIMULATIONS,
)
from src.data_loader import load_all_historical_stats, load_team_stats
from src.data_cleaner import clean_team_stats, build_historical_matchups
from src.feature_engineering import build_training_matchups, prepare_prediction_features
from src.model import BracketModel, evaluate_loto_cv, train_final_model
from src.bracket_builder import BracketSimulator
from src.bracket_visualizer import (
    print_bracket, print_quick_fill_guide,
    print_team_advancement_odds, print_upset_alerts,
)


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           🏀 March Madness Bracket Predictor 🏀             ║
║                      2026 Edition                           ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Prediction System"
    )
    parser.add_argument("--backtest", action="store_true",
                        help="Run leave-one-tournament-out cross-validation")
    parser.add_argument("--predict", action="store_true",
                        help="Generate 2026 tournament bracket")
    parser.add_argument("--strategy",
                        choices=["deterministic", "upset_aware", "monte_carlo"],
                        default="deterministic",
                        help="Bracket simulation strategy (default: deterministic)")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help=f"Monte Carlo simulation count (default: {DEFAULT_SIMULATIONS})")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining even if saved model exists")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for bracket files")
    return parser.parse_args()


def _load_and_prepare_training_data():
    """Load historical data → clean → build matchups → build features."""
    print("Loading historical stats...")
    hist_df = load_all_historical_stats()

    print("Cleaning data...")
    hist_df = clean_team_stats(hist_df)

    print("Building historical matchups...")
    matchups = build_historical_matchups(hist_df)
    print(f"  Found {len(matchups)} matchups")

    print("Engineering features...")
    training_data = build_training_matchups(hist_df, matchups)
    print(f"  Training set: {len(training_data)} rows, {len(ALL_MODEL_FEATURES)} features")

    return hist_df, training_data


def run_backtest():
    """Run leave-one-tournament-out cross-validation."""
    _, training_data = _load_and_prepare_training_data()
    metrics = evaluate_loto_cv(training_data, ALL_MODEL_FEATURES)

    overall = metrics["overall"]
    print(f"Overall accuracy: {overall['accuracy']*100:.1f}%")
    print(f"Overall log-loss: {overall['log_loss']:.3f}")


def run_prediction(strategy, simulations, retrain, output_dir):
    """Generate a 2026 bracket prediction."""
    hist_df, training_data = _load_and_prepare_training_data()

    # Train or load model
    model_exists = os.path.exists(TRAINED_MODEL_FILE)
    if retrain or not model_exists:
        print("\nTraining model on all historical data...")
        model = train_final_model(training_data, ALL_MODEL_FEATURES)
    else:
        print(f"\nLoading saved model from {TRAINED_MODEL_FILE}")
        model = BracketModel.load(TRAINED_MODEL_FILE)

    # Load 2026 team stats
    print("\nLoading 2026 season stats...")
    current_df = load_team_stats(CURRENT_YEAR)
    current_df = clean_team_stats(current_df)
    print(f"  {len(current_df)} teams loaded")

    # Simulate bracket
    print(f"\nSimulating bracket (strategy: {strategy})...")
    simulator = BracketSimulator(
        bracket_path=BRACKET_2026_FILE,
        team_stats_df=current_df,
        model=model,
        feature_builder=prepare_prediction_features,
    )

    results = simulator.simulate_tournament(strategy=strategy)

    # Output
    bracket_file = os.path.join(output_dir, "bracket_2026.txt")
    print_bracket(results, file_path=bracket_file)
    print_quick_fill_guide(results)

    if strategy == "monte_carlo":
        print_team_advancement_odds(results)

    print_upset_alerts(results)


def main():
    args = parse_args()

    if not args.backtest and not args.predict:
        print(BANNER)
        print("Please specify --backtest or --predict.")
        print("Run with --help for usage information.")
        sys.exit(1)

    print(BANNER)

    try:
        if args.backtest:
            run_backtest()
        if args.predict:
            run_prediction(
                strategy=args.strategy,
                simulations=args.simulations,
                retrain=args.retrain,
                output_dir=args.output_dir,
            )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nMake sure you've downloaded the Kaggle dataset first.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
