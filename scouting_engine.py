"""
Scouting Engine  -  Grassroots / ISL talent identification
===========================================================
Reads football commentary / transcripts (and optional video signals) and
produces, for a scout:

  * a per-player ATTRIBUTE PROFILE (finishing, dribbling, passing, pace, ...)
  * a ROLE-FIT rating for a target position (ST, W, CM, CB, GK, ...)
  * STRENGTHS and WEAKNESSES
  * a scout VERDICT: SIGN / MONITOR / DEVELOP / PASS
  * a ranked SHORTLIST for a target role

It reuses the offline sentiment analyzer from the penalty project so positive
vs negative mentions of an attribute push the score up or down.

The improvement-plan and richer video features are separate phases; this module
focuses on the scouting shortlist.
"""

import re
import csv
import json
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from penalty_predictor_backend_OFFLINE import OfflineSentimentAnalyzer

# ────────────────────────────────────────────────────────────────
# Attribute lexicons  (English; feed translated transcripts here)
# ────────────────────────────────────────────────────────────────
ATTRIBUTE_KEYWORDS = {
    "finishing":   ["finish", "goal", "score", "clinical", "shot", "strike",
                    "conversion", "poacher", "tap-in", "volley"],
    "dribbling":   ["dribble", "skill", "beats", "take-on", "close control",
                    "mazy", "run", "trickery", "flair", "nutmeg", "glides"],
    "passing":     ["pass", "distribution", "vision", "through ball", "range",
                    "switch", "ball-playing", "pinged", "threaded", "incisive"],
    "pace":        ["pace", "quick", "fast", "sprint", "accelerate", "rapid",
                    "burst", "lightning", "speed"],
    "physical":    ["strong", "power", "aerial", "header", "duel", "robust",
                    "tall", "muscular", "wins the ball", "commanding"],
    "defending":   ["tackle", "intercept", "block", "clearance", "marking",
                    "recover", "last-ditch", "stops", "shackled", "covering"],
    "positioning": ["positioning", "awareness", "anticipation", "reads",
                    "space", "intelligent", "timing", "well-placed"],
    "workrate":    ["work rate", "press", "energy", "tireless", "tracking",
                    "harry", "relentless", "engine", "covers ground"],
    "composure":   ["composed", "calm", "confident", "assured", "mature",
                    "unflustered", "ice-cool", "presence"],
    "creativity":  ["creative", "assist", "chance", "key pass", "inventive",
                    "playmaker", "unlocks", "delivery", "cross", "set up"],
    "goalkeeping": ["save", "shot-stopping", "reflexes", "claim", "command",
                    "parried", "denied", "one-on-one", "handling"],
}

NEGATIVE_HINTS = ["poor", "weak", "mistake", "error", "wasteful", "sloppy",
                  "lost", "slow", "lightweight", "caught", "miscontrol",
                  "misplaced", "struggled", "lazy", "exposed", "off the pace"]

POTENTIAL_HINTS = ["young", "youngster", "promising", "potential", "raw",
                   "prospect", "talented", "teenager", "academy", "exciting",
                   "future", "developing", "rough diamond"]

# ────────────────────────────────────────────────────────────────
# Role -> attribute weights (must roughly sum to 1.0 per role)
# ────────────────────────────────────────────────────────────────
ROLE_WEIGHTS = {
    "ST": {"finishing": .30, "positioning": .20, "pace": .15, "physical": .15,
           "composure": .10, "dribbling": .10},
    "W":  {"dribbling": .30, "pace": .25, "creativity": .20, "finishing": .12,
           "workrate": .13},
    "CM": {"passing": .26, "creativity": .18, "workrate": .16, "composure": .14,
           "positioning": .14, "defending": .12},
    "CB": {"defending": .30, "positioning": .24, "physical": .22, "composure": .14,
           "passing": .10},
    "FB": {"defending": .22, "pace": .22, "workrate": .20, "creativity": .18,
           "physical": .18},
    "GK": {"goalkeeping": .55, "composure": .25, "passing": .20},
}

ROLE_NAMES = {
    "ST": "Striker", "W": "Winger", "CM": "Central Midfielder",
    "CB": "Centre-Back", "FB": "Full-Back", "GK": "Goalkeeper",
}

STOPWORDS = {"The", "A", "It", "Now", "And", "But", "In", "Min", "From", "To",
             "He", "She", "His", "Her", "That", "This", "There", "They", "We",
             "Half", "Full", "Time", "Goal", "Match", "Both", "Up", "On", "At",
             "ISL", "FC", "United", "City", "Super", "League", "First", "Second"}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


@dataclass
class ScoutProfile:
    player: str
    mentions: int
    best_role: str
    role_rating: float
    verdict: str
    attributes: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    potential_flag: bool
    note: str = ""


class ScoutingEngine:
    def __init__(self, thresholds=None, min_mentions=2, model="offline"):
        # tunable verdict cut-offs (optimization panel can override)
        self.thresholds = thresholds or {"sign": 75, "monitor": 62, "develop": 48}
        self.min_mentions = min_mentions
        self.model_used = "offline (rule-based lexicon)"
        self.sent = OfflineSentimentAnalyzer()
        if model == "bert":
            # optional upgrade: RoBERTa sentiment from the full backend
            try:
                from penalty_predictor_backend import BERTSentimentAnalyzer
                self.sent = BERTSentimentAnalyzer()
                self.model_used = "BERT (cardiffnlp/twitter-roberta-base-sentiment)"
            except Exception as e:
                self.model_used = f"offline (BERT unavailable: {type(e).__name__}; install transformers+torch)"

    # ---- find players -------------------------------------------------
    def detect_players(self, commentary: str, roster: Optional[List[str]] = None,
                       min_mentions: Optional[int] = None) -> Dict[str, List[str]]:
        """Return {player: [sentences mentioning them]}."""
        if min_mentions is None:
            min_mentions = self.min_mentions
        sentences = re.split(r"[.!?\n]", commentary)
        if roster:
            players = {}
            for name in roster:
                key = strip_accents(name).lower()
                toks = [t for t in key.split() if len(t) > 2]
                hits = [s.strip() for s in sentences
                        if any(t in strip_accents(s).lower() for t in toks)]
                if hits:
                    players[name] = hits
            return players

        # auto-detect capitalised names
        counts: Dict[str, List[str]] = {}
        for s in sentences:
            for nm in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", s):
                if nm.split()[0] in STOPWORDS:
                    continue
                counts.setdefault(nm, []).append(s.strip())
        return {n: v for n, v in counts.items() if len(v) >= min_mentions}

    # ---- attribute scoring -------------------------------------------
    def score_attributes(self, sentences: List[str]) -> Dict[str, float]:
        text = " ".join(sentences)
        tl = strip_accents(text).lower()
        attrs = {}
        for attr, kws in ATTRIBUTE_KEYWORDS.items():
            hits = sum(tl.count(k) for k in kws)
            if hits == 0:
                continue
            # sentiment of the sentences that mention this attribute
            rel = [s for s in sentences
                   if any(k in strip_accents(s).lower() for k in kws)]
            score, _ = self.sent.analyze_text(" ".join(rel)) if rel else (0.0, "")
            neg = sum(strip_accents(" ".join(rel)).lower().count(n) for n in NEGATIVE_HINTS)
            # base from evidence volume, modulated by sentiment & negatives
            base = 50 + min(30, hits * 6)        # more mentions -> stronger evidence
            base += score * 25                    # positive tone lifts, negative drops
            base -= neg * 8
            attrs[attr] = round(max(0, min(100, base)), 1)
        return attrs

    # ---- role fit -----------------------------------------------------
    def role_rating(self, attrs: Dict[str, float], role: str) -> float:
        weights = ROLE_WEIGHTS[role]
        total_w, acc = 0.0, 0.0
        for attr, w in weights.items():
            val = attrs.get(attr)
            if val is None:
                continue  # no evidence for this attribute -> skip (don't penalise to 0)
            acc += w * val
            total_w += w
        if total_w == 0:
            return 0.0
        # scale by coverage so players with evidence on more key attributes rank higher
        coverage = total_w / sum(weights.values())
        return round((acc / total_w) * (0.6 + 0.4 * coverage), 1)

    def best_role(self, attrs: Dict[str, float]) -> (str, float):
        best, best_score = None, -1
        for role in ROLE_WEIGHTS:
            r = self.role_rating(attrs, role)
            if r > best_score:
                best, best_score = role, r
        return best, best_score

    def verdict(self, rating: float, potential: bool) -> str:
        t = self.thresholds
        if rating >= t["sign"]:
            return "SIGN"
        if rating >= t["monitor"]:
            return "MONITOR"
        if rating >= t["develop"] or potential:
            return "DEVELOP"
        return "PASS"

    # ---- build profile ------------------------------------------------
    def profile_player(self, name: str, sentences: List[str],
                       target_role: Optional[str] = None) -> ScoutProfile:
        attrs = self.score_attributes(sentences)
        text_l = strip_accents(" ".join(sentences)).lower()
        potential = any(h in text_l for h in POTENTIAL_HINTS)

        if target_role:
            role, rating = target_role, self.role_rating(attrs, target_role)
        else:
            role, rating = self.best_role(attrs)

        ranked = sorted(attrs.items(), key=lambda x: x[1], reverse=True)
        strengths = [f"{a} ({v:.0f})" for a, v in ranked if v >= 65][:4]
        weaknesses = [f"{a} ({v:.0f})" for a, v in ranked if v < 50][-3:]

        note = "Flagged as a development prospect." if potential else ""
        return ScoutProfile(
            player=name, mentions=len(sentences),
            best_role=role, role_rating=rating,
            verdict=self.verdict(rating, potential),
            attributes=attrs, strengths=strengths, weaknesses=weaknesses,
            potential_flag=potential, note=note,
        )

    # ---- shortlist ----------------------------------------------------
    def shortlist(self, commentary: str, target_role: str,
                  roster: Optional[List[str]] = None, top_n: int = 10,
                  video_signals: Optional[Dict[str, Dict[str, float]]] = None) -> List[ScoutProfile]:
        """Rank players for a target role. video_signals optionally adds/overrides
        attributes per player, e.g. {'Sharma': {'pace': 88}}."""
        players = self.detect_players(commentary, roster=roster)
        profiles = []
        for name, sents in players.items():
            prof = self.profile_player(name, sents, target_role=target_role)
            if video_signals and name in video_signals:
                # blend video-derived attributes in
                for a, v in video_signals[name].items():
                    prior = prof.attributes.get(a, 50)
                    prof.attributes[a] = round(0.5 * prior + 0.5 * v, 1)
                prof.role_rating = self.role_rating(prof.attributes, target_role)
                prof.verdict = self.verdict(prof.role_rating, prof.potential_flag)
            profiles.append(prof)
        profiles.sort(key=lambda p: p.role_rating, reverse=True)
        return profiles[:top_n]


# ────────────────────────────────────────────────────────────────
# Improvement-plan generator
# ────────────────────────────────────────────────────────────────
TRAINING_LIBRARY = {
    "finishing":   ["Daily finishing reps: 50 strikes from the edge of the box, both feet",
                    "1v1-vs-keeper drills under fatigue at the end of sessions",
                    "Video review of shot selection and body shape before striking"],
    "dribbling":   ["Cone slalom and tight-space ball-mastery, 15 min daily",
                    "1v1 isolation drills vs a live defender",
                    "Work on change-of-pace and the first explosive touch"],
    "passing":     ["Rondos (5v2) to sharpen first-time passing under pressure",
                    "Long-range switch-of-play repetitions, 30 per session",
                    "Scan-and-pass drills to improve vision and weight of pass"],
    "pace":        ["Structured sprint & acceleration program (10–30m) with a coach",
                    "Plyometrics and resisted-sprint work for explosiveness",
                    "Recovery-run conditioning to sustain pace late in games"],
    "physical":    ["Strength & core program (supervised, age-appropriate)",
                    "Shielding and hold-up duels vs a stronger partner",
                    "Aerial timing drills for headers and jump strength"],
    "defending":   ["1v1 defending: jockeying, timing the tackle, body position",
                    "Positional shadow-play to learn covering and pressing triggers",
                    "Clearance and recovery-tackle repetitions under pressure"],
    "positioning": ["Tactical sessions on reading the game and marking zones",
                    "Watch-and-analyse clips of elite players in the same role",
                    "Small-sided games focused on staying compact and timing runs"],
    "workrate":    ["Interval conditioning to raise high-intensity capacity",
                    "Pressing-trigger drills to channel energy effectively",
                    "GPS/heart-rate tracked sessions to build a tireless engine"],
    "composure":   ["Pressure-scenario reps (penalties, final-third decisions)",
                    "Decision-making under fatigue at session end",
                    "Breathing/visualisation routines for big moments"],
    "creativity":  ["Final-third overload games to manufacture chances",
                    "Crossing and through-ball delivery repetitions",
                    "Freedom in small-sided games to try flair and key passes"],
    "goalkeeping": ["Shot-stopping and reaction-save drills, daily",
                    "Distribution work: throwing and kicking under press",
                    "Command-of-area and cross-claiming repetitions"],
}


def generate_improvement_plan(name, attributes, role, weak_threshold=58):
    """Turn a player's weak attributes into a targeted training plan.
    attributes: dict attr->score(0-100). role: target position code.
    Returns a dict with focus_areas (weakest first), strengths_to_leverage,
    and a summary line."""
    role_attrs = list(ROLE_WEIGHTS.get(role, {}).keys())
    scored = sorted(attributes.items(), key=lambda x: x[1])
    # focus = low attributes, prioritising ones important for the role
    focus = []
    for attr, val in scored:
        if val < weak_threshold:
            focus.append({
                "attribute": attr,
                "score": val,
                "role_relevant": attr in role_attrs,
                "drills": TRAINING_LIBRARY.get(attr, ["General technical work with a coach."]),
            })
    # surface role-relevant weaknesses first
    focus.sort(key=lambda f: (not f["role_relevant"], f["score"]))
    focus = focus[:4]

    strengths = [a for a, v in sorted(attributes.items(), key=lambda x: -x[1]) if v >= 65][:3]

    if not focus:
        summary = f"{name} shows no major weaknesses in the data — maintain strengths and add game time."
    else:
        areas = ", ".join(f["attribute"] for f in focus)
        summary = (f"Priority development areas for {name} ({ROLE_NAMES.get(role, role)}): "
                   f"{areas}. Leverage strengths in {', '.join(strengths) if strengths else 'overall play'}.")

    return {"player": name, "role": role, "role_name": ROLE_NAMES.get(role, role),
            "focus_areas": focus, "strengths_to_leverage": strengths, "summary": summary}


def save_results(profiles: List[ScoutProfile], role: str, base_path: str = "."):
    import os
    j = os.path.join(base_path, "scout_shortlist.json")
    c = os.path.join(base_path, "scout_shortlist.csv")
    with open(j, "w", encoding="utf-8") as f:
        json.dump({"role": role, "shortlist": [asdict(p) for p in profiles]}, f, indent=2)
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "player", "best_role", "role_rating", "verdict",
                    "potential", "strengths", "weaknesses"])
        for i, p in enumerate(profiles, 1):
            w.writerow([i, p.player, p.best_role, p.role_rating, p.verdict,
                        "Y" if p.potential_flag else "",
                        "; ".join(p.strengths), "; ".join(p.weaknesses)])
    return j, c
