"""
Development Simulation Engine  -  grassroots player development
===============================================================
Goes beyond "is this player good?" to "how does this player get better?".

For each player it:
  1. detects MISTAKE-MOMENTS in the commentary (wasteful finish, caught out of
     position, sloppy control, misplaced pass, beaten in a duel ...),
  2. explains WHAT WENT WRONG, WHY (root cause linked to a weak attribute), and
     the BETTER APPROACH + a training drill,
  3. attaches a 2D-pitch SCENARIO id so the frontend can animate actual-vs-ideal,
  4. estimates the player's CEILING / potential (not just current ability),
  5. rolls everything up into TEAM-STRATEGY notes.

The 2D "simulation" is a coaching schematic (illustrating the correct approach),
not a frame-accurate reconstruction of the real video.
"""

import re
import unicodedata


def _strip(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


# mistake-type -> detection + coaching content + animation scenario
MISTAKE_TYPES = {
    "finishing": {
        "keywords": ["wasteful", "dragged", "wide", "wasted", "squandered",
                     "blazed", "scuffed", "missed the target", "fluffed", "ballooned"],
        "attr": "finishing", "scenario": "shot",
        "wrong": "Rushed or mis-hit the shot and wasted a goalscoring chance.",
        "why": "Composure and finishing technique under pressure need work.",
        "better": "Take one settling touch, open the body and pass it into the corner — or square to a better-placed teammate.",
        "drill": "Finishing reps under fatigue; 1v1-vs-keeper choosing placement over power.",
    },
    "positioning": {
        "keywords": ["out of position", "caught", "exposed", "ball-watching",
                     "lost his man", "switched off", "wrong side", "beaten for pace"],
        "attr": "positioning", "scenario": "position",
        "wrong": "Got caught out of position, opening space in behind the defence.",
        "why": "Game-reading and defensive awareness are still developing.",
        "better": "Hold the line, stay goal-side of the runner, and scan the danger before the ball arrives.",
        "drill": "Shadow-defending and positional small-sided games; cue-recognition video sessions.",
    },
    "control": {
        "keywords": ["sloppy", "miscontrol", "careless", "gave away",
                     "lost possession", "heavy touch", "dispossessed", "loose touch"],
        "attr": "composure", "scenario": "control",
        "wrong": "Lost the ball with a poor first touch or careless control under pressure.",
        "why": "First touch and composure in tight areas need sharpening.",
        "better": "Take the first touch away from pressure, then play a simple pass to keep possession.",
        "drill": "Rondos (5v2) and tight-space ball-mastery to improve first touch under a press.",
    },
    "passing": {
        "keywords": ["misplaced", "poor pass", "overhit", "underhit",
                     "sloppy pass", "gave the ball away", "wayward"],
        "attr": "passing", "scenario": "pass",
        "wrong": "Misplaced a pass and turned possession over cheaply.",
        "why": "Passing weight and decision-making under pressure are inconsistent.",
        "better": "Scan early, choose the simple option and weight the pass to the teammate's feet.",
        "drill": "Passing-pattern drills and scan-and-pass repetitions against a live press.",
    },
    "duel": {
        "keywords": ["beaten", "skinned", "brushed off", "muscled off",
                     "outmuscled", "lost the duel", "shrugged off"],
        "attr": "physical", "scenario": "duel",
        "wrong": "Lost a one-on-one duel, letting the opponent through.",
        "why": "Strength, timing and body positioning in duels need work.",
        "better": "Get side-on, use the body to shield, and time the tackle rather than diving in.",
        "drill": "1v1 defending and strength/core work; jockeying and shielding drills.",
    },
}

MINUTE_RE = re.compile(r"min\s*(\d+)", re.I)

# map any attribute to the coaching scenario that best develops it
ATTR_TO_TYPE = {
    "finishing": "finishing", "positioning": "positioning", "defending": "positioning",
    "composure": "control", "passing": "passing", "creativity": "passing",
    "physical": "duel", "dribbling": "control", "pace": "positioning",
    "workrate": "positioning", "goalkeeping": "control",
}


def focus_area(attributes):
    """Pick the weakest attribute that maps to a scenario, so every player gets a
    'recommended approach' simulation even when no specific mistake was flagged."""
    if not attributes:
        return None
    for attr, val in sorted(attributes.items(), key=lambda x: x[1]):
        t = ATTR_TO_TYPE.get(attr)
        if t:
            cfg = MISTAKE_TYPES[t]
            return {"attribute": attr, "score": val, "scenario": cfg["scenario"],
                    "better_approach": cfg["better"], "drill": cfg["drill"]}
    return None


def detect_mistakes(sentences):
    """Return a list of mistake dicts found in the player's commentary."""
    found = []
    for s in sentences:
        low = _strip(s).lower()
        for mtype, cfg in MISTAKE_TYPES.items():
            if any(k in low for k in cfg["keywords"]):
                m = MINUTE_RE.search(s)
                minute = int(m.group(1)) if m else None
                # dedupe: same type+minute, or a minute-less repeat of an existing type
                if any(f["type"] == mtype and f["minute"] == minute for f in found):
                    continue
                if minute is None and any(f["type"] == mtype for f in found):
                    continue
                found.append({
                    "type": mtype,
                    "scenario": cfg["scenario"],
                    "minute": minute,
                    "snippet": s.strip()[:160],
                    "attribute": cfg["attr"],
                    "what_went_wrong": cfg["wrong"],
                    "why": cfg["why"],
                    "better_approach": cfg["better"],
                    "drill": cfg["drill"],
                })
    return found


def estimate_ceiling(current_rating, attributes, potential_flag):
    """Estimate a development ceiling (0-100) above current ability.
    Young/raw players with a few standout attributes have higher headroom."""
    if not attributes:
        top = current_rating
    else:
        vals = sorted(attributes.values(), reverse=True)
        top = sum(vals[:3]) / len(vals[:3])
    headroom = 14 if potential_flag else 7
    standout_bonus = 6 if any(v >= 80 for v in attributes.values()) else 0
    ceiling = max(current_rating + 4, top + headroom + standout_bonus)
    ceiling = round(min(99, ceiling), 1)
    gap = round(ceiling - current_rating, 1)
    if potential_flag and gap >= 12:
        band = "HIGH CEILING"
    elif gap >= 8:
        band = "ROOM TO GROW"
    else:
        band = "NEAR CEILING"
    return {"current": current_rating, "ceiling": ceiling, "gap": gap, "band": band}


def player_development(name, attributes, sentences, current_rating,
                      verdict, potential_flag):
    mistakes = detect_mistakes(sentences)
    ceiling = estimate_ceiling(current_rating, attributes, potential_flag)
    # unlock note
    weak = sorted(attributes.items(), key=lambda x: x[1])[:2]
    weak_names = ", ".join(a for a, _ in weak) if weak else "consistency"
    if ceiling["band"] == "HIGH CEILING":
        unlock = f"High-potential prospect: tightening {weak_names} could lift {name} from {ceiling['current']} toward {ceiling['ceiling']}."
    elif ceiling["band"] == "ROOM TO GROW":
        unlock = f"Clear room to grow: focused work on {weak_names} is the path to a higher level."
    else:
        unlock = f"{name} is close to their current ceiling — refine {weak_names} and add match minutes."
    return {
        "player": name, "current_rating": current_rating, "verdict": verdict,
        "potential_flag": potential_flag, "ceiling": ceiling,
        "mistakes": mistakes, "unlock": unlock,
        "focus": focus_area(attributes),
        "summary": (f"{len(mistakes)} correctable moment(s) identified."
                    if mistakes else "No specific mistakes flagged — showing the key development area.")
    }


def team_strategy(player_devs):
    """Aggregate individual mistakes into team-level recommendations."""
    counts = {}
    for pd in player_devs:
        for m in pd["mistakes"]:
            counts[m["type"]] = counts.get(m["type"], 0) + 1
    notes = []
    rec = {
        "control":     "Ball retention is a recurring issue — drill playing out under pressure with rondos and a calm first pass.",
        "passing":     "Passing turnovers are common — work on scanning, simple options and pass weight in training.",
        "positioning": "Defensive shape is being broken — coach the line, tracking runners and pre-scanning danger.",
        "finishing":   "Chances are being wasted — dedicate sessions to final-third composure and placement over power.",
        "duel":        "Players are losing duels — add strength/core work and 1v1 body-positioning drills.",
    }
    for mtype, c in sorted(counts.items(), key=lambda x: -x[1]):
        notes.append({"area": mtype, "count": c, "recommendation": rec[mtype]})
    if not notes:
        notes.append({"area": "general", "count": 0,
                      "recommendation": "Few errors detected — maintain standards and progress to tougher opposition."})
    # one overall line
    if counts:
        top = max(counts, key=counts.get)
        headline = f"Priority team focus: {top} ({counts[top]} flagged moments across the squad)."
    else:
        headline = "Squad showed few correctable errors in this sample."
    return {"headline": headline, "notes": notes}
