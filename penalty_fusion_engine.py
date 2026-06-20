"""
Penalty Shootout Fusion Engine  (Phase 1)
=========================================
Combines TWO signal sources into one penalty-suitability score per player so a
coach can pick the shootout order:

  1. Historical / technical penalty stats  -> player_penalty_stats.csv
  2. In-match psychological readiness       -> sentiment on the commentary text
                                               (reuses OfflineSentimentAnalyzer)

The score is a transparent weighted blend (easy for a coach to trust). Video
features (technique_rating, composure_rating) currently come from the CSV as
placeholders -- when you run the pose/video pipeline later, overwrite those
columns and the engine picks them up automatically. No code change needed.

Usage:
    python penalty_fusion_engine.py
    python penalty_fusion_engine.py --stats player_penalty_stats.csv \
                                    --commentary commentary_psg_arsenal.txt \
                                    --team Arsenal
"""

import os
import re
import csv
import json
import argparse
import unicodedata
from datetime import datetime

from penalty_predictor_backend_OFFLINE import OfflineSentimentAnalyzer

# ────────────────────────────────────────────────────────────────
# Scoring weights  (must sum to 1.0). Tune these to taste.
# ────────────────────────────────────────────────────────────────
WEIGHTS = {
    "history":   0.35,   # career penalty conversion rate
    "technique": 0.20,   # technique_rating  (from video later)
    "composure": 0.20,   # composure_rating + live body-language sentiment
    "readiness": 0.15,   # psychological readiness from commentary
    "experience":0.10,   # big-game penalty experience
}

# Category thresholds
RECOMMENDED_MIN = 70
BACKUP_MIN = 50


def strip_accents(text: str) -> str:
    """Dembele <-> Dembélé matching."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def find_player_sentences(commentary: str, player: str):
    """Return sentences from the commentary that mention this player."""
    norm_commentary_sentences = re.split(r"[.!?\n]", commentary)
    key = strip_accents(player).lower()
    # match on any name token (handles "Nuno Mendes" -> match on "mendes")
    tokens = [t for t in key.split() if len(t) > 2]
    hits = []
    for sent in norm_commentary_sentences:
        norm_sent = strip_accents(sent).lower()
        if any(tok in norm_sent for tok in tokens):
            hits.append(sent.strip())
    return hits


class PenaltyFusionEngine:
    def __init__(self, weights=None, recommended_min=RECOMMENDED_MIN, backup_min=BACKUP_MIN):
        self.weights = weights or WEIGHTS
        self.recommended_min = recommended_min
        self.backup_min = backup_min
        self.sentiment = OfflineSentimentAnalyzer()

    # ---- component scores (all normalised to 0-100) -------------------
    def _history_score(self, taken, scored):
        if taken <= 0:
            return 50.0  # unknown -> neutral
        rate = scored / taken
        # small-sample shrinkage toward 50% so 1/1 doesn't beat 14/16
        confidence = min(1.0, taken / 10.0)
        shrunk = rate * confidence + 0.5 * (1 - confidence)
        return round(shrunk * 100, 1)

    def _readiness_score(self, sentences):
        """Psychological readiness from the commentary mentioning the player."""
        if not sentences:
            return 50.0, "NEUTRAL", "NONE", 0  # not mentioned -> neutral
        text = " ".join(sentences)
        score, label = self.sentiment.analyze_text(text)
        stress = self.sentiment.extract_stress_indicators(text)
        # map sentiment (-1..1) to 0..100
        readiness = (score + 1) * 50
        # fatigue read
        tl = text.lower()
        if any(k in tl for k in ["exhausted", "flagging", "running on fumes"]):
            fatigue = "HIGH"
        elif any(k in tl for k in ["tired", "energy", "worn out"]):
            fatigue = "MEDIUM"
        elif "fresh" in tl:
            fatigue = "LOW"
        else:
            fatigue = "MEDIUM" if stress else "LOW"
        # penalties for fatigue & stress
        if fatigue == "HIGH":
            readiness -= 18
        elif fatigue == "MEDIUM":
            readiness -= 6
        readiness -= 5 * len(stress)
        readiness = max(0, min(100, readiness))
        return round(readiness, 1), label, fatigue, len(stress)

    # ---- fusion -------------------------------------------------------
    def score_player(self, row, commentary):
        taken = int(row.get("pens_taken", 0) or 0)
        scored = int(row.get("pens_scored", 0) or 0)
        technique = float(row.get("technique_rating", 50) or 50)
        composure_base = float(row.get("composure_rating", 50) or 50)
        experience = float(row.get("big_game_experience", 0) or 0)

        history = self._history_score(taken, scored)
        sentences = find_player_sentences(commentary, row["player"])
        readiness, mental, fatigue, stress = self._readiness_score(sentences)

        # Blend static composure rating with live body-language sentiment
        composure = round(0.6 * composure_base + 0.4 * readiness, 1)
        experience_score = min(100.0, experience * 11)  # ~9 big games -> ~99

        w = self.weights
        suitability = (
            w["history"]    * history +
            w["technique"]  * technique +
            w["composure"]  * composure +
            w["readiness"]  * readiness +
            w["experience"] * experience_score
        )
        suitability = round(max(0, min(100, suitability)), 1)

        if suitability >= self.recommended_min:
            category = "RECOMMENDED"
        elif suitability >= self.backup_min:
            category = "BACKUP"
        else:
            category = "AVOID"

        return {
            "player": row["player"],
            "team": row.get("team", ""),
            "position": row.get("position", ""),
            "suitability": suitability,
            "category": category,
            "history_score": history,
            "technique": technique,
            "composure": composure,
            "readiness": readiness,
            "experience_score": round(experience_score, 1),
            "mental_state": mental,
            "fatigue": fatigue,
            "stress_signals": stress,
            "pen_record": f"{scored}/{taken}",
            "mentioned_in_match": bool(sentences),
        }

    def run(self, stats_path, commentary, team_filter=None):
        with open(stats_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if team_filter:
            rows = [r for r in rows if r.get("team", "").lower() == team_filter.lower()]
        results = [self.score_player(r, commentary) for r in rows]
        results.sort(key=lambda x: x["suitability"], reverse=True)
        return results


def print_report(results, team_filter=None):
    title = f"PENALTY SHOOTOUT ORDER - COACH'S LIST"
    if team_filter:
        title += f"  ({team_filter})"
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'#':<3}{'PLAYER':<16}{'TEAM':<9}{'SCORE':<7}{'REC':<13}{'PEN REC':<9}{'MENTAL':<9}")
    print("-" * 70)
    for i, p in enumerate(results, 1):
        print(f"{i:<3}{p['player']:<16}{p['team']:<9}{p['suitability']:<7}"
              f"{p['category']:<13}{p['pen_record']:<9}{p['mental_state']:<9}")
    print("-" * 70)
    recommended = [p["player"] for p in results if p["category"] == "RECOMMENDED"]
    print(f"\nSuggested takers (in order): {', '.join(recommended) if recommended else 'none above threshold'}")
    avoid = [p["player"] for p in results if p["category"] == "AVOID"]
    if avoid:
        print(f"Hold back tonight: {', '.join(avoid)}")
    print("=" * 70)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Penalty shootout fusion engine")
    ap.add_argument("--stats", default="player_penalty_stats.csv")
    ap.add_argument("--commentary", default="commentary_psg_arsenal.txt")
    ap.add_argument("--team", default=None, help="Filter to one team, e.g. Arsenal")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    stats_path = args.stats if os.path.isabs(args.stats) else os.path.join(base, args.stats)
    comm_path = args.commentary if os.path.isabs(args.commentary) else os.path.join(base, args.commentary)

    with open(comm_path, encoding="utf-8") as f:
        commentary = f.read()

    engine = PenaltyFusionEngine()
    results = engine.run(stats_path, commentary, team_filter=args.team)
    print_report(results, team_filter=args.team)

    if not args.no_save:
        ts = datetime.now().isoformat()
        with open(os.path.join(base, "penalty_fusion_results.json"), "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "weights": engine.weights, "results": results}, f, indent=2)
        with open(os.path.join(base, "penalty_fusion_results.csv"), "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        print("\nSaved: penalty_fusion_results.json + penalty_fusion_results.csv")


if __name__ == "__main__":
    main()
