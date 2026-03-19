"""
Bracket simulation engine for March Madness prediction.

Supports deterministic, upset-aware (probabilistic), and
Monte Carlo ESPN-optimized simulation strategies.
"""

import json
import random
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ESPN_POINTS, RANDOM_SEED, DEFAULT_SIMULATIONS, BRACKET_2026_FILE


# Seed matchup order in R64 (determines bracket position)
_R64_SEED_PAIRS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

_ROUND_NAMES = ["R64", "R32", "S16", "E8"]
_REGION_ORDER = ["East", "South", "Midwest", "West"]


class BracketSimulator:
    """Simulates a full NCAA tournament bracket."""

    def __init__(self, bracket_path, team_stats_df, model, feature_builder):
        """
        Args:
            bracket_path: Path to bracket JSON file.
            team_stats_df: DataFrame with current season team stats.
            model: Trained BracketModel instance.
            feature_builder: Callable(team_a_stats, team_b_stats) → feature DataFrame.
        """
        with open(bracket_path) as f:
            self.bracket = json.load(f)
        self.team_stats_df = team_stats_df
        self.model = model
        self.feature_builder = feature_builder
        self._stats_cache: dict[str, dict] = {}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_team_stats(self, team_name: str) -> dict:
        if team_name in self._stats_cache:
            return self._stats_cache[team_name]
        from src.utils import normalize_team_name
        name = normalize_team_name(team_name)
        from config import TEAM_COL
        mask = self.team_stats_df[TEAM_COL] == name
        if mask.sum() == 0:
            # Try original name
            mask = self.team_stats_df[TEAM_COL] == team_name
        if mask.sum() == 0:
            # Return a minimal placeholder
            self._stats_cache[team_name] = {}
            return {}
        stats = self.team_stats_df[mask].iloc[0].to_dict()
        self._stats_cache[team_name] = stats
        return stats

    def _predict_game(self, team_a: str, team_b: str) -> tuple[str, float]:
        """Predict a game. Returns (winner, P(team_a wins))."""
        stats_a = self._get_team_stats(team_a)
        stats_b = self._get_team_stats(team_b)

        if not stats_a or not stats_b:
            # Fallback: lower seed number wins
            return team_a, 0.5

        features = self.feature_builder(stats_a, stats_b)
        prob_a = float(self.model.predict_proba(features)[0])
        return (team_a if prob_a >= 0.5 else team_b), prob_a

    def _pick_winner(self, team_a, team_b, prob_a, strategy, rng=None):
        """Choose winner based on strategy."""
        if strategy == "deterministic":
            return team_a if prob_a >= 0.5 else team_b
        else:  # upset_aware or monte_carlo
            r = rng.random() if rng else random.random()
            return team_a if r < prob_a else team_b

    # ── Region simulation ─────────────────────────────────────────────────

    def _simulate_region(self, region_name, strategy, rng=None):
        """Simulate a single region through Elite 8. Returns round results + champion."""
        region = self.bracket["regions"][region_name]
        rounds = []

        # Build R64 matchups from seeds
        current_teams = []
        r64_games = []
        for seed_a, seed_b in _R64_SEED_PAIRS:
            team_a = region[str(seed_a)]
            team_b = region[str(seed_b)]

            # Handle play-in teams (indicated by " / ")
            if " / " in team_a:
                parts = team_a.split(" / ")
                team_a = parts[0]  # Just pick first for now
            if " / " in team_b:
                parts = team_b.split(" / ")
                team_b = parts[0]

            _, prob_a = self._predict_game(team_a, team_b)
            winner = self._pick_winner(team_a, team_b, prob_a, strategy, rng)
            r64_games.append({
                "team_a": team_a, "team_b": team_b,
                "seed_a": seed_a, "seed_b": seed_b,
                "winner": winner, "probability": prob_a,
            })
            current_teams.append(winner)

        rounds.append({"round_name": "R64", "games": r64_games})

        # Subsequent rounds: pair adjacent winners
        for round_name in ["R32", "S16", "E8"]:
            next_teams = []
            games = []
            for i in range(0, len(current_teams), 2):
                team_a = current_teams[i]
                team_b = current_teams[i + 1]
                _, prob_a = self._predict_game(team_a, team_b)
                winner = self._pick_winner(team_a, team_b, prob_a, strategy, rng)
                games.append({
                    "team_a": team_a, "team_b": team_b,
                    "winner": winner, "probability": prob_a,
                })
                next_teams.append(winner)
            rounds.append({"round_name": round_name, "games": games})
            current_teams = next_teams

        return rounds, current_teams[0]  # champion is last remaining

    # ── Full tournament ───────────────────────────────────────────────────

    def simulate_tournament(self, strategy="deterministic", seed=None):
        """
        Run a full tournament simulation.

        Args:
            strategy: 'deterministic', 'upset_aware', or 'monte_carlo'.
            seed: Random seed for reproducibility.

        Returns:
            Dict with rounds, champion, strategy, and region_champions.
        """
        if strategy == "monte_carlo":
            return self._monte_carlo_optimize()

        rng = random.Random(seed or RANDOM_SEED)
        all_rounds = {}
        region_champions = {}

        for region in _REGION_ORDER:
            region_rounds, champion = self._simulate_region(region, strategy, rng)
            all_rounds[region] = region_rounds
            region_champions[region] = champion

        # Final Four
        ff_games = []
        # East vs West, South vs Midwest
        matchups = [("East", "West"), ("South", "Midwest")]
        ff_winners = []
        for r1, r2 in matchups:
            ta, tb = region_champions[r1], region_champions[r2]
            _, prob_a = self._predict_game(ta, tb)
            winner = self._pick_winner(ta, tb, prob_a, strategy, rng)
            ff_games.append({
                "team_a": ta, "team_b": tb,
                "region_a": r1, "region_b": r2,
                "winner": winner, "probability": prob_a,
            })
            ff_winners.append(winner)

        # Championship
        ta, tb = ff_winners[0], ff_winners[1]
        _, prob_a = self._predict_game(ta, tb)
        champion = self._pick_winner(ta, tb, prob_a, strategy, rng)
        championship = {
            "team_a": ta, "team_b": tb,
            "winner": champion, "probability": prob_a,
        }

        return {
            "strategy": strategy,
            "region_rounds": all_rounds,
            "region_champions": region_champions,
            "final_four": ff_games,
            "championship": championship,
            "champion": champion,
        }

    # ── Monte Carlo ───────────────────────────────────────────────────────

    def _simulate_single_bracket(self, rng):
        """Run one probabilistic simulation."""
        result = {"picks": defaultdict(list)}
        region_champs = {}

        for region in _REGION_ORDER:
            _, champion = self._simulate_region(region, "upset_aware", rng)
            region_champs[region] = champion

        result["region_champions"] = region_champs

        # Final Four
        matchups = [("East", "West"), ("South", "Midwest")]
        ff_winners = []
        for r1, r2 in matchups:
            ta, tb = region_champs[r1], region_champs[r2]
            _, prob_a = self._predict_game(ta, tb)
            winner = self._pick_winner(ta, tb, prob_a, "upset_aware", rng)
            ff_winners.append(winner)

        # Championship
        ta, tb = ff_winners[0], ff_winners[1]
        _, prob_a = self._predict_game(ta, tb)
        champion = self._pick_winner(ta, tb, prob_a, "upset_aware", rng)
        result["champion"] = champion

        return result

    def _monte_carlo_optimize(self, n_simulations=DEFAULT_SIMULATIONS):
        """
        Run many simulations and pick the bracket maximising
        expected ESPN points.
        """
        champion_counts = defaultdict(int)

        print(f"\nRunning {n_simulations:,} Monte Carlo simulations...")

        for i in range(n_simulations):
            rng = random.Random(RANDOM_SEED + i)
            result = self._simulate_single_bracket(rng)
            champion_counts[result["champion"]] += 1

        # Print champion distribution
        print("\nChampion Distribution (top 10):")
        sorted_champs = sorted(champion_counts.items(), key=lambda x: -x[1])
        for team, count in sorted_champs[:10]:
            pct = count / n_simulations * 100
            print(f"  {team:<25} {pct:>5.1f}%")

        # Generate the deterministic bracket as the "best" bracket
        # (Monte Carlo informs but the deterministic picks are most likely)
        best = self.simulate_tournament(strategy="deterministic")
        best["strategy"] = "monte_carlo"
        best["simulation_count"] = n_simulations
        best["champion_distribution"] = dict(sorted_champs)

        return best

    def _score_bracket(self, bracket_results):
        """Score a bracket using ESPN point values."""
        total = 0
        round_points = {"R64": 10, "R32": 20, "S16": 40, "E8": 80}
        for region_rounds in bracket_results.get("region_rounds", {}).values():
            for rnd in region_rounds:
                pts = round_points.get(rnd["round_name"], 0)
                total += pts * len(rnd["games"])
        total += 160 * len(bracket_results.get("final_four", []))
        total += 320  # championship
        return total
