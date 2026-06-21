"""
Continental Coaching Strategies — knowledge base
================================================
A curated map of how different football cultures develop the attributes a player
might be weak in. Used two ways:
  * as the OFFLINE engine for cross-continent development insight (no LLM needed)
  * as grounding CONTEXT in the LLM prompt (llm_insights.py) so the model's
    advice is anchored to real coaching philosophies, not hallucinated.

These are broad, well-documented generalisations about footballing cultures —
useful as development reference points, not absolute rules.
"""

# Per-continent / country coaching philosophy summaries
PHILOSOPHIES = {
    "Spain":       "Juego de posición (positional play): possession, positional rotations, "
                   "playing through the lines, technical short passing and spatial awareness.",
    "Netherlands": "Total Football roots: positional interchange, technical ball mastery, "
                   "playing out from the back, intelligent movement and width.",
    "Germany":     "Gegenpressing and counter-pressing: aggressive ball recovery, vertical "
                   "transitions, structured pressing triggers and high work-rate.",
    "England":     "High-intensity, end-to-end pressing and transitions; physical duels, "
                   "wide play and relentless tempo.",
    "Italy":       "Tactical defending and game management: zonal marking, defensive "
                   "organisation, reading the game and controlling tempo.",
    "Brazil":      "Technical flair and 1v1 mastery built on futsal: close control, "
                   "dribbling, creativity, improvisation and joga bonito.",
    "Argentina":   "Street-football grit plus technique: la gambeta (dribbling), tactical "
                   "fight, mentality and creative playmaking (la pausa).",
    "France":      "Athletic, technically complete academy model (Clairefontaine): pace, "
                   "power and individual quality within a balanced structure.",
    "Africa":      "Athleticism and physicality: explosive pace, power, aerial strength, "
                   "1v1 individual brilliance and resilience.",
    "Japan":       "Disciplined, technical, possession-based collective: precise passing, "
                   "tactical structure, tireless team work-rate and humility.",
    "South Korea": "Elite fitness and intensity: relentless running, pressing stamina, "
                   "discipline and never-say-die mentality.",
    "USA":         "Athletic, data-driven, sports-science model: fitness, athleticism and "
                   "structured, analytics-informed development pathways.",
}

# Which footballing cultures best develop each attribute, + a concrete idea
ATTRIBUTE_STRATEGY = {
    "dribbling": {
        "cultures": ["Brazil", "Argentina"],
        "approach": "Brazilian futsal and Argentine street-football develop close control and 1v1 dribbling.",
        "drill": "Add futsal sessions and small-court 1v1 games to build tight close control and change of pace.",
    },
    "creativity": {
        "cultures": ["Brazil", "Argentina", "Spain"],
        "approach": "South American flair plus Spanish positional play breed creative playmakers.",
        "drill": "Overload the final third in training and give freedom to try la pausa and key passes.",
    },
    "passing": {
        "cultures": ["Spain", "Netherlands", "Japan"],
        "approach": "Spanish/Dutch positional play and Japanese possession sharpen passing range and weight.",
        "drill": "Daily rondos and positional games (juego de posición) to pass through the lines under pressure.",
    },
    "positioning": {
        "cultures": ["Italy", "Spain"],
        "approach": "Italian tactical schooling and Spanish positional play teach reading space and timing.",
        "drill": "Shadow-play and zonal tactical sessions; analyse elite players in the same role.",
    },
    "defending": {
        "cultures": ["Italy", "Germany"],
        "approach": "Italian defensive organisation and German pressing structure build defensive quality.",
        "drill": "1v1 defending, zonal marking patterns and pressing-trigger recognition drills.",
    },
    "workrate": {
        "cultures": ["Germany", "South Korea", "Japan"],
        "approach": "German gegenpressing and Korean/Japanese intensity build a tireless engine.",
        "drill": "Interval conditioning and pressing-trigger games tracked with GPS/heart-rate.",
    },
    "pace": {
        "cultures": ["Africa", "France", "USA"],
        "approach": "African athletic development and French/US sports-science build explosive speed.",
        "drill": "Structured sprint, plyometric and resisted-running programs with a coach.",
    },
    "physical": {
        "cultures": ["Africa", "England", "France"],
        "approach": "African physicality and English duel-intensity develop strength and aerial power.",
        "drill": "Age-appropriate strength/core work plus shielding and aerial-timing drills.",
    },
    "finishing": {
        "cultures": ["Brazil", "Italy", "Germany"],
        "approach": "Brazilian technique and European clinical coaching sharpen finishing.",
        "drill": "High-volume finishing under fatigue; placement-over-power and 1v1-vs-keeper reps.",
    },
    "composure": {
        "cultures": ["Italy", "Spain", "Japan"],
        "approach": "Italian game-management and Spanish/Japanese calm-in-possession build composure.",
        "drill": "Decision-making under fatigue and pressure-scenario reps (final third, penalties).",
    },
    "goalkeeping": {
        "cultures": ["Germany", "Italy"],
        "approach": "German modern sweeper-keeper training and Italian shot-stopping schooling.",
        "drill": "Distribution-under-press work plus reaction-save and command-of-area drills.",
    },
}


def offline_insight(player, role, attributes, weaknesses=None, strengths=None):
    """Generate a cross-continent development insight WITHOUT an LLM.
    attributes: dict attr->score. Returns a structured dict."""
    attributes = attributes or {}
    # pick the 2-3 weakest attributes that we have a strategy for
    ranked = sorted(attributes.items(), key=lambda x: x[1])
    focus = [a for a, v in ranked if a in ATTRIBUTE_STRATEGY and v < 65][:3]
    if not focus:  # fall back to lowest known attributes
        focus = [a for a, _ in ranked if a in ATTRIBUTE_STRATEGY][:2]

    items, cultures = [], []
    for attr in focus:
        s = ATTRIBUTE_STRATEGY[attr]
        cultures += s["cultures"]
        items.append({
            "attribute": attr,
            "score": attributes.get(attr),
            "study": s["cultures"],
            "approach": s["approach"],
            "recommendation": s["drill"],
        })
    # de-dup cultures, keep order
    seen, culture_list = set(), []
    for c in cultures:
        if c not in seen:
            seen.add(c); culture_list.append(c)

    summary = (f"To develop {player} ({role}), study how "
               f"{', '.join(culture_list[:3]) if culture_list else 'top football cultures'} "
               f"coach the areas they most need to improve"
               + (f": {', '.join(i['attribute'] for i in items)}." if items else "."))
    return {
        "source": "offline",
        "player": player,
        "summary": summary,
        "focus_items": items,
        "cultures": [{"name": c, "philosophy": PHILOSOPHIES.get(c, "")} for c in culture_list[:4]],
    }
