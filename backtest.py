"""
Backtest / Evaluation harness for the Scouting Engine
=====================================================
Validates the engine against REAL outcomes.

Method:
  * Input  : scouting-style descriptions of real ISL Emerging Player winners
             (validation/validation_commentary.txt)
  * Truth  : their real position + whether they earned an India national call-up
             (validation/groundtruth.csv)
  * Test   : does the engine, reading only the descriptions, (a) assign the
             correct position and (b) rate the eventual internationals as top
             talent?

Metrics reported:
  - Role-classification accuracy : engine best_role == real position
  - Talent precision@K           : of the engine's top-K by rating, share who
                                   became internationals
  - Recall of internationals     : share of internationals the engine flags
                                   (verdict SIGN/MONITOR/DEVELOP, i.e. not PASS)
  - Mean rating: internationals vs non-internationals (separation)
  - Rank correlation (Spearman)  : engine rating vs national-call-up label

Run:
    python backtest.py
"""

import os
import csv

from scouting_engine import ScoutingEngine, ROLE_NAMES

BASE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(BASE, "validation")


def load_truth():
    with open(os.path.join(VAL, "groundtruth.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spearman(xs, ys):
    """Spearman rank correlation, no numpy needed."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) *
           sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den else 0.0


def run_backtest(base=None):
    """Run the validation and return (rows, metrics) without printing."""
    val = os.path.join(base, "validation") if base else VAL
    with open(os.path.join(val, "groundtruth.csv"), newline="", encoding="utf-8") as f:
        truth = list(csv.DictReader(f))
    commentary = open(os.path.join(val, "validation_commentary.txt"), encoding="utf-8").read()
    roster = [r["player"] for r in truth]
    engine = ScoutingEngine()
    players = engine.detect_players(commentary, roster=roster)
    rows = []
    for r in truth:
        name = r["player"]
        prof = engine.profile_player(name, players.get(name, []), target_role=r["position"])
        rows.append({
            "player": name, "real_pos": r["position"], "engine_pos": prof.best_role,
            "pos_correct": prof.best_role == r["position"], "rating": prof.role_rating,
            "verdict": prof.verdict,
            "national": r["national_call_up"].strip().upper() == "Y",
            "flagged": prof.verdict != "PASS",
        })
    rows.sort(key=lambda x: x["rating"], reverse=True)
    n = len(rows)
    intl = [r for r in rows if r["national"]]
    non = [r for r in rows if not r["national"]]
    def p_at_k(k):
        top = rows[:k]
        return sum(r["national"] for r in top) / len(top)
    metrics = {
        "n": n,
        "role_acc": sum(r["pos_correct"] for r in rows) / n,
        "internationals": len(intl),
        "recall": (sum(r["flagged"] for r in intl) / len(intl)) if intl else 0,
        "p3": p_at_k(3), "p5": p_at_k(5),
        "mean_intl": (sum(r["rating"] for r in intl) / len(intl)) if intl else 0,
        "mean_non": (sum(r["rating"] for r in non) / len(non)) if non else 0,
        "rho": spearman([r["rating"] for r in rows], [1 if r["national"] else 0 for r in rows]),
    }
    metrics["separation"] = metrics["mean_intl"] - metrics["mean_non"]
    return rows, metrics


def main():
    truth = load_truth()
    commentary = open(os.path.join(VAL, "validation_commentary.txt"), encoding="utf-8").read()
    roster = [r["player"] for r in truth]
    engine = ScoutingEngine()

    players = engine.detect_players(commentary, roster=roster)
    rows = []
    for r in truth:
        name = r["player"]
        sents = players.get(name, [])
        prof = engine.profile_player(name, sents, target_role=r["position"])
        rows.append({
            "player": name,
            "real_pos": r["position"],
            "engine_pos": prof.best_role,
            "pos_correct": prof.best_role == r["position"],
            "rating": prof.role_rating,
            "verdict": prof.verdict,
            "national": r["national_call_up"].strip().upper() == "Y",
            "flagged": prof.verdict != "PASS",
        })

    rows.sort(key=lambda x: x["rating"], reverse=True)

    # ---- metrics ----
    n = len(rows)
    role_acc = sum(r["pos_correct"] for r in rows) / n
    internationals = [r for r in rows if r["national"]]
    non = [r for r in rows if not r["national"]]
    recall = sum(r["flagged"] for r in internationals) / len(internationals) if internationals else 0
    mean_intl = sum(r["rating"] for r in internationals) / len(internationals) if internationals else 0
    mean_non = sum(r["rating"] for r in non) / len(non) if non else 0

    def precision_at_k(k):
        top = rows[:k]
        return sum(r["national"] for r in top) / len(top)

    rho = spearman([r["rating"] for r in rows], [1 if r["national"] else 0 for r in rows])

    # ---- report ----
    print("=" * 74)
    print("SCOUTING ENGINE - REAL-WORLD VALIDATION (ISL Emerging Player winners)")
    print("=" * 74)
    print(f"{'PLAYER':<22}{'REAL':<5}{'ENGINE':<7}{'RATING':<8}{'VERDICT':<10}{'INDIA?'}")
    print("-" * 74)
    for r in rows:
        chk = "OK" if r["pos_correct"] else "x"
        print(f"{r['player']:<22}{r['real_pos']:<5}{r['engine_pos']:<3}{chk:<4}"
              f"{r['rating']:<8}{r['verdict']:<10}{'YES' if r['national'] else 'no'}")
    print("-" * 74)
    print("\nMETRICS")
    print(f"  Players evaluated              : {n}")
    print(f"  Role-classification accuracy   : {role_acc*100:.0f}%  "
          f"({sum(r['pos_correct'] for r in rows)}/{n})")
    print(f"  Internationals in sample       : {len(internationals)}/{n}")
    print(f"  Recall of internationals flagged: {recall*100:.0f}%  (engine did not 'PASS' them)")
    print(f"  Precision@3 (top 3 by rating)  : {precision_at_k(3)*100:.0f}% are internationals")
    print(f"  Precision@5 (top 5 by rating)  : {precision_at_k(5)*100:.0f}% are internationals")
    print(f"  Mean rating - internationals   : {mean_intl:.1f}")
    print(f"  Mean rating - non-international : {mean_non:.1f}")
    print(f"  Separation (intl - non)        : {mean_intl-mean_non:+.1f} points")
    print(f"  Spearman(rating, call-up)      : {rho:+.2f}")
    print("=" * 74)

    # save results
    out = os.path.join(VAL, "validation_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved per-player results: {out}")
    return rows, {
        "role_acc": role_acc, "recall": recall, "p3": precision_at_k(3),
        "p5": precision_at_k(5), "mean_intl": mean_intl, "mean_non": mean_non, "rho": rho,
    }


if __name__ == "__main__":
    main()
