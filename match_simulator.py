"""
Match Simulator
===============
Turns commentary / a video transcript into a minute-by-minute playback of player
ACTIONS, each tagged good / bad / neutral and paired with a coaching SUGGESTION.
The frontend plays through it like a match unfolding and can show key moments on
the 2D pitch.

Everything is read from the entered commentary — nothing is hard-coded per match.
"""

import re

try:
    from scouting_engine import strip_accents, STOPWORDS
except Exception:  # self-contained fallback
    import unicodedata
    def strip_accents(t):
        return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    STOPWORDS = {"The", "A", "Min", "Goal", "Penalty", "Attempt", "Foul", "Corner",
                 "Offside", "Substitution", "Half", "Full", "Second", "First"}

# action rules in priority order: (id, keywords, verdict, scenario, suggestion)
ACTION_RULES = [
    ("goal", ["goal!", " scores", "scored", "nets", "converts", "slots home",
              "finds the net", "buries", "tucks it"], "good", "shot",
     "Clinical finish — reinforce this composure and placement in training."),
    # a shot that is saved/blocked/off-target is a MISSED chance for the shooter
    ("miss", ["missed", "misses", "attempt saved", "is saved", "shot saved", "saved in",
              "penalty saved", "wide", "over the bar", "blazed", "skied", "off target",
              "dragged", "spurned", "hits the post", "wasteful", "fluffed", "blocked",
              "header misses", "high and wide"], "bad", "shot",
     "Rushed or mis-hit. Take a settling touch and place it into the corner — or square to a teammate."),
    # keeper making a save (keeper-specific phrasing only)
    ("save", ["parries", "palms", "tips over", "denies", "keeps it out",
              "magnificent stop", "point-blank", "fingertip"], "good", None,
     "Sharp goalkeeping — good set position and reflexes; now restart play quickly."),
    ("lost_ball", ["miscontrol", "heavy touch", "loses possession", "gave the ball away",
                   "gives it away", "dispossessed", "careless", "sloppy", "loose touch"],
     "bad", "control",
     "Lost the ball under pressure. First touch away from pressure, then a simple pass to retain."),
    ("foul", ["foul", "brings down", "trips", "booking", "yellow card", "red card",
              "mistimed", "lunges", "clatters"], "bad", "duel",
     "Mistimed challenge. Stay on your feet, jockey and time the tackle."),
    ("misplaced", ["misplaced", "overhit", "underhit", "wayward", "poor pass",
                   "stray pass", "gives it straight"], "bad", "pass",
     "Turned it over. Scan earlier and pick the simple, safe pass."),
    ("dribble", ["dribble", "beats his man", "beats two", "mazy", "skips past",
                 "nutmeg", "takes on", "glides past", "jinks"], "good", None,
     "Excellent dribble — keep driving at defenders and pick the right moment to release."),
    ("key_pass", ["assist", "through ball", "key pass", "threads", "defence-splitting",
                  "incisive pass", "wonderful cross", "lovely delivery", "sets up"],
     "good", "pass",
     "Great vision/delivery — keep looking for these line-breaking passes."),
    ("tackle", ["tackle", "interception", "intercepts", "block", "clearance",
                "last-ditch", "wins the ball", "recovers", "shackles"], "good", None,
     "Strong defensive action — good timing and positioning."),
    ("shot", ["shot", "strike", "effort", "attempt", "volley", "drives", "fires",
              "unleashes", "tests the keeper", "header"], "neutral", "shot",
     "Good to get a shot away — work on placement and shot selection."),
    ("foul_won", ["wins a free kick", "wins a foul", "is fouled"], "neutral", None,
     "Drew the foul — use the set-piece well."),
    ("sub", ["substitution", "replaces", "comes on", "is replaced"], "neutral", None,
     "Squad rotation — fresh legs to influence the game."),
]

ICONS = {"goal": "⚽", "save": "🧤", "miss": "❌", "lost_ball": "⚠️", "foul": "🟨",
         "misplaced": "↪️", "dribble": "🌀", "key_pass": "🎯", "tackle": "🛡️",
         "shot": "👟", "foul_won": "🤝", "sub": "🔁", "event": "•"}
LABELS = {"goal": "Goal", "save": "Save", "miss": "Miss / Saved", "lost_ball": "Lost ball",
          "foul": "Foul", "misplaced": "Misplaced pass", "dribble": "Dribble",
          "key_pass": "Key pass", "tackle": "Defending", "shot": "Shot",
          "foul_won": "Wins free kick", "sub": "Substitution", "event": "Play"}


def _minute(line):
    m = re.match(r"\s*\[?\s*(?:min\s*)?(\d{1,3})(?:\s*\+\s*(\d+))?\s*'?\]?\s*[:\-]?", line, re.I)
    if m:
        return int(m.group(1)), (int(m.group(2)) if m.group(2) else 0)
    return None, None


def _players(line):
    out = []
    for nm in re.findall(r"\b([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)?)\b", line):
        w = nm.split()
        if w[0] in STOPWORDS or w[-1] in STOPWORDS:
            continue
        out.append(nm)
    # de-dup preserve order
    seen, res = set(), []
    for n in out:
        if n not in seen:
            seen.add(n); res.append(n)
    return res


def _classify(line):
    low = strip_accents(line).lower()
    for aid, kws, verdict, scenario, suggestion in ACTION_RULES:
        if any(k in low for k in kws):
            return aid, verdict, scenario, suggestion
    return None, None, None, None


SKIP_WORDS = {"Lineups", "Match", "Players", "Both", "Teams", "Half", "Full", "Time",
              "Extra", "Penalty", "Shootout", "Result", "Source", "Goal"}


def simulate(commentary):
    """Parse the commentary into ordered minute-by-minute events."""
    events = []
    for seq, raw in enumerate(commentary.split("\n")):
        line = raw.strip()
        if len(line) < 4:
            continue
        minute, plus = _minute(line)
        aid, verdict, scenario, suggestion = _classify(line)
        if aid is None and minute is None:
            continue  # skip headers/blank
        players = [p for p in _players(line) if p.split()[0] not in SKIP_WORDS]
        text = re.sub(r"^\s*\[?\s*(?:min\s*)?\d{1,3}(?:\s*\+\s*\d+)?\s*'?\]?\s*[:\-]?\s*", "", line, flags=re.I)
        events.append({
            "seq": seq,
            "minute": minute, "plus": plus,
            "minute_label": (f"{minute}+{plus}'" if plus else (f"{minute}'" if minute is not None else "")),
            "player": players[0] if players else "",
            "players": players,
            "action": aid or "event",
            "label": LABELS.get(aid or "event", "Play"),
            "icon": ICONS.get(aid or "event", "•"),
            "verdict": verdict or "neutral",
            "scenario": scenario,
            "suggestion": suggestion if aid else "",
            "text": text,
        })
    events.sort(key=lambda e: (e["minute"] if e["minute"] is not None else 999, e["plus"], e["seq"]))
    summary = {
        "events": len(events),
        "goals": sum(1 for e in events if e["action"] == "goal"),
        "good": sum(1 for e in events if e["verdict"] == "good"),
        "bad": sum(1 for e in events if e["verdict"] == "bad"),
        "minutes": sorted({e["minute"] for e in events if e["minute"] is not None}),
    }
    return {"events": events, "summary": summary}
