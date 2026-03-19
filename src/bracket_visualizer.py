"""
Bracket visualization and output formatting.

Provides console bracket display, ESPN/CBS quick-fill guide,
team advancement odds, and upset alert highlighting.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR


_REGION_ORDER = ["East", "South", "Midwest", "West"]


def print_bracket(results: dict, file_path: str | None = None) -> None:
    """
    Print a formatted tournament bracket to console (and optionally to file).
    """
    lines = []

    def p(text=""):
        lines.append(text)

    strategy_label = results.get("strategy", "deterministic").replace("_", " ").title()
    sim_count = results.get("simulation_count", "")
    sim_note = f" ({sim_count:,} simulations)" if sim_count else ""

    p("╔══════════════════════════════════════════════════════════════╗")
    p("║          2026 NCAA TOURNAMENT BRACKET PREDICTIONS           ║")
    p(f"║{'Strategy: ' + strategy_label + sim_note:^62}║")
    p("╚══════════════════════════════════════════════════════════════╝")
    p()

    region_rounds = results.get("region_rounds", {})
    region_champions = results.get("region_champions", {})

    for region in _REGION_ORDER:
        p(f"━━━ {region.upper()} REGION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        p()

        rounds = region_rounds.get(region, [])
        for rnd in rounds:
            round_name = rnd["round_name"]
            p(f"  {_round_display(round_name)}:")
            for g in rnd["games"]:
                ta = g["team_a"]
                tb = g["team_b"]
                w = g["winner"]
                prob = g["probability"]
                # Show probability of the WINNER
                wp = prob if w == ta else 1 - prob

                seed_a = g.get("seed_a", "")
                seed_b = g.get("seed_b", "")
                seed_a_str = f"({seed_a:>2}) " if seed_a else "     "
                seed_b_str = f"({seed_b:>2}) " if seed_b else "     "

                p(f"    {seed_a_str}{ta:<20} vs {seed_b_str}{tb:<20} → {w:<20} [{wp*100:.1f}%]")
            p()

        champ = region_champions.get(region, "?")
        p(f"  ➤ {region.upper()} CHAMPION: {champ}")
        p()

    # Final Four
    p("╔══════════════════════════════════════════════════════════════╗")
    p("║                        FINAL FOUR                           ║")
    p("╚══════════════════════════════════════════════════════════════╝")
    for g in results.get("final_four", []):
        ta, tb = g["team_a"], g["team_b"]
        ra, rb = g.get("region_a", ""), g.get("region_b", "")
        w = g["winner"]
        prob = g["probability"]
        wp = prob if w == ta else 1 - prob
        p(f"  {ra} ({ta}) vs {rb} ({tb}) → {w} [{wp*100:.1f}%]")
    p()

    # Championship
    champ_game = results.get("championship", {})
    p("╔══════════════════════════════════════════════════════════════╗")
    p("║                      CHAMPIONSHIP                           ║")
    p("╚══════════════════════════════════════════════════════════════╝")
    if champ_game:
        ta, tb = champ_game["team_a"], champ_game["team_b"]
        w = champ_game["winner"]
        prob = champ_game["probability"]
        wp = prob if w == ta else 1 - prob
        p(f"  {ta} vs {tb} → {w} [{wp*100:.1f}%]")
    p()
    p(f"  🏆 NATIONAL CHAMPION: {results.get('champion', '?')} 🏆")
    p()

    # Print to console
    output = "\n".join(lines)
    print(output)

    # Save to file
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(output)
        print(f"\nBracket saved → {file_path}")


def print_quick_fill_guide(results: dict) -> None:
    """Print picks organized for fast ESPN/CBS bracket entry."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              QUICK-FILL GUIDE (ESPN/CBS Entry)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    region_rounds = results.get("region_rounds", {})
    region_champions = results.get("region_champions", {})

    for region in _REGION_ORDER:
        print(f"{region.upper()}:")
        rounds = region_rounds.get(region, [])
        for rnd in rounds:
            winners = [g["winner"] for g in rnd["games"]]
            print(f"  {rnd['round_name']:>3}: {', '.join(winners)}")
        print()

    # Final Four
    ff_winners = [g["winner"] for g in results.get("final_four", [])]
    print(f"FINAL FOUR: {', '.join(ff_winners)}")

    champ = results.get("champion", "?")
    print(f"CHAMPIONSHIP: {champ}")
    print()


def print_team_advancement_odds(results: dict) -> None:
    """
    Print team advancement probabilities from Monte Carlo data.
    """
    champ_dist = results.get("champion_distribution", {})
    if not champ_dist:
        print("No Monte Carlo data available.")
        return

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                  CHAMPION PROBABILITIES                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{'Team':<25} | {'Probability':>10}")
    print(f"{'-'*25}-|{'-'*11}")

    n = results.get("simulation_count", 1)
    sorted_teams = sorted(champ_dist.items(), key=lambda x: -x[1])
    for team, count in sorted_teams[:16]:
        pct = count / n * 100 if isinstance(count, int) else count
        print(f"{team:<25} | {pct:>9.1f}%")
    print()


def print_upset_alerts(results: dict) -> None:
    """Highlight all upsets (higher seed beating lower seed)."""
    upsets = []

    region_rounds = results.get("region_rounds", {})
    for region, rounds in region_rounds.items():
        for rnd in rounds:
            for g in rnd["games"]:
                seed_a = g.get("seed_a", 0)
                seed_b = g.get("seed_b", 0)
                if not seed_a or not seed_b:
                    continue
                winner = g["winner"]
                prob = g["probability"]
                # Upset = higher seed number wins
                if winner == g["team_b"] and seed_b > seed_a:
                    wp = 1 - prob
                    upsets.append((rnd["round_name"], seed_b, g["team_b"],
                                  seed_a, g["team_a"], wp, region))
                elif winner == g["team_a"] and seed_a > seed_b:
                    wp = prob
                    upsets.append((rnd["round_name"], seed_a, g["team_a"],
                                  seed_b, g["team_b"], wp, region))

    if not upsets:
        print("\nNo upsets predicted (chalk bracket).")
        return

    print()
    print("⚡ UPSET ALERTS ⚡")
    for rnd, high_seed, winner, low_seed, loser, prob, region in upsets:
        print(f"  {rnd} [{region}]: ({high_seed}) {winner} over ({low_seed}) {loser}"
              f"  [win prob: {prob*100:.1f}%]")
    print()


def _round_display(code: str) -> str:
    labels = {
        "R64": "Round of 64",
        "R32": "Round of 32",
        "S16": "Sweet 16",
        "E8": "Elite 8",
        "F4": "Final Four",
    }
    return labels.get(code, code)
