"""
Scout Shortlist Runner
======================
Ties the scouting pipeline together for a recruitment scout:

  transcript/commentary  ->  ScoutingEngine  ->  ranked shortlist for a role

Usage:
    # shortlist strikers from a commentary/transcript file
    python scout_shortlist.py --commentary sample_isl_commentary.txt --role ST

    # transcribe a match video first (needs whisper), then shortlist wingers
    python scout_shortlist.py --video match.mp4 --language hi --role W

Roles: ST (striker) W (winger) CM (midfield) CB (centre-back) FB (full-back) GK
"""

import os
import argparse

from scouting_engine import (ScoutingEngine, ROLE_NAMES, ROLE_WEIGHTS,
                              save_results)


def load_roster(path):
    """Read a roster file: one player per line, optional ',POSITION' note."""
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line.split(",")[0].strip())
    return names


def get_commentary(args):
    if args.commentary:
        with open(args.commentary, encoding="utf-8") as f:
            return f.read()
    if args.video:
        from transcription import transcribe
        print(f"Transcribing {args.video} ...")
        return transcribe(args.video, language=args.language, translate_to_english=True)
    raise SystemExit("Provide --commentary <file> or --video <file>")


def print_shortlist(profiles, role):
    rn = ROLE_NAMES.get(role, role)
    print("\n" + "=" * 72)
    print(f"SCOUTING SHORTLIST  -  Target role: {rn} ({role})")
    print("=" * 72)
    print(f"{'#':<3}{'PLAYER':<18}{'FIT':<7}{'VERDICT':<10}{'POT':<5}{'TOP STRENGTHS'}")
    print("-" * 72)
    for i, p in enumerate(profiles, 1):
        pot = "★" if p.potential_flag else ""
        strengths = ", ".join(s.split(" (")[0] for s in p.strengths[:3]) or "-"
        print(f"{i:<3}{p.player:<18}{p.role_rating:<7}{p.verdict:<10}{pot:<5}{strengths}")
    print("-" * 72)

    print("\nDETAILED PROFILES")
    for i, p in enumerate(profiles, 1):
        print(f"\n{i}. {p.player}  -  {ROLE_NAMES.get(p.best_role, p.best_role)} "
              f"| Fit {p.role_rating} | {p.verdict}"
              f"{'  ★ prospect' if p.potential_flag else ''}")
        if p.strengths:
            print(f"   Strengths : {', '.join(p.strengths)}")
        if p.weaknesses:
            print(f"   To improve: {', '.join(p.weaknesses)}")
        top_attrs = sorted(p.attributes.items(), key=lambda x: x[1], reverse=True)
        print("   Attributes: " + ", ".join(f"{a} {v:.0f}" for a, v in top_attrs))
        if p.note:
            print(f"   Note: {p.note}")
    print("\n" + "=" * 72)
    signs = [p.player for p in profiles if p.verdict == "SIGN"]
    devs = [p.player for p in profiles if p.verdict == "DEVELOP"]
    if signs:
        print(f"Recommended to sign: {', '.join(signs)}")
    if devs:
        print(f"Development prospects: {', '.join(devs)}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Generate a scouting shortlist for a role")
    ap.add_argument("--commentary", help="commentary / transcript text file")
    ap.add_argument("--video", help="match video to transcribe first")
    ap.add_argument("--language", default=None, help="language code for transcription, e.g. hi")
    ap.add_argument("--role", default="ST", choices=list(ROLE_WEIGHTS.keys()),
                    help="target position to scout for")
    ap.add_argument("--roster", help="optional roster file for exact player detection")
    ap.add_argument("--top", type=int, default=10, help="shortlist size")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    commentary = get_commentary(args)
    roster = load_roster(args.roster) if args.roster else None
    engine = ScoutingEngine()
    profiles = engine.shortlist(commentary, target_role=args.role,
                                roster=roster, top_n=args.top)
    print_shortlist(profiles, args.role)

    if not args.no_save:
        base = os.path.dirname(os.path.abspath(__file__))
        j, c = save_results(profiles, args.role, base_path=base)
        print(f"\nSaved: {os.path.basename(j)} + {os.path.basename(c)}")


if __name__ == "__main__":
    main()
