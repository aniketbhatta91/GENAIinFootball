"""
PSG vs Arsenal - UEFA Champions League Final 2026
Penalty Shootout Eligibility Analysis

Loads the match commentary from commentary_psg_arsenal.txt and runs the
offline penalty-shootout predictor on it. Prints a squad eligibility report
and detailed per-player rankings, and saves the results to JSON.

Usage:
    python psg_arsenal_analysis.py
    python psg_arsenal_analysis.py --commentary path/to/file.txt
"""

import os
import json
import argparse
from datetime import datetime
from dataclasses import asdict

from penalty_predictor_backend_OFFLINE import PenaltyShootoutAnalyzer

# Real result for reference / verification
ACTUAL_RESULT = {
    "match": "PSG vs Arsenal",
    "competition": "UEFA Champions League Final 2026",
    "venue": "Puskas Arena, Budapest",
    "score": "PSG 1-1 Arsenal (PSG win 4-3 on penalties)",
    "goals": ["Havertz 6' (ARS)", "Dembele 65' pen (PSG)"],
    "shootout": {
        "PSG": "Dembele scored, Nuno Mendes missed (saved), Vitinha scored, Beraldo scored",
        "Arsenal": "Eze missed (wide), Rice scored, Gabriel missed (over)",
        "winner": "PSG 4-3",
    },
}

DEFAULT_COMMENTARY = "commentary_psg_arsenal.txt"


def load_commentary(path: str) -> str:
    """Read the commentary text file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Commentary file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def print_actual_result():
    print("=" * 64)
    print("ACTUAL MATCH RESULT (for reference)")
    print("=" * 64)
    print(f"  {ACTUAL_RESULT['match']} - {ACTUAL_RESULT['competition']}")
    print(f"  Venue : {ACTUAL_RESULT['venue']}")
    print(f"  Score : {ACTUAL_RESULT['score']}")
    print(f"  Goals : {', '.join(ACTUAL_RESULT['goals'])}")
    print(f"  Shootout winner: {ACTUAL_RESULT['shootout']['winner']}")
    print("=" * 64 + "\n")


def run(commentary_path: str, save_json: bool = True):
    print_actual_result()

    print(f"Loading commentary from: {commentary_path}\n")
    commentary = load_commentary(commentary_path)

    analyzer = PenaltyShootoutAnalyzer()
    results = analyzer.analyze(commentary)

    print("\n" + "=" * 64)
    print(results["summary"])
    print("=" * 64)

    print("\nDETAILED PLAYER RANKINGS:\n")
    for i, player in enumerate(results["players"], 1):
        print(f"{i}. {player.name} ({player.position})")
        print(f"   Category: {player.category} | Probability: {player.probability:.1f}%")
        print(f"   Mental State: {player.mental_state} | Fatigue: {player.fatigue_level} "
              f"| Stress signals: {player.stress_count}")
        print(f"   {player.confidence_explanation}\n")

    if save_json:
        out = {
            "actual_result": ACTUAL_RESULT,
            "match_context": results["match_context"],
            "timestamp": datetime.now().isoformat(),
            "players": [asdict(p) for p in results["players"]],
            "summary": results["summary"],
        }
        out_path = "psg_arsenal_analysis_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSG vs Arsenal penalty analysis")
    parser.add_argument("--commentary", default=DEFAULT_COMMENTARY,
                        help="Path to commentary text file")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the JSON results file")
    args = parser.parse_args()

    # Resolve relative to this script's directory so it works from anywhere
    base = os.path.dirname(os.path.abspath(__file__))
    commentary_path = args.commentary
    if not os.path.isabs(commentary_path):
        commentary_path = os.path.join(base, commentary_path)

    try:
        run(commentary_path, save_json=not args.no_save)
        print("\nAnalysis complete.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
