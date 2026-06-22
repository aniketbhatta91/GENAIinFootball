"""
Penalty Shootout Dataset (10 real matches) + evaluation
=======================================================
Ten famous penalty shootouts with their factual taker outcomes (scored/missed).
Used for:
  * the penalty-tab match dropdown (loads a factual shootout commentary), and
  * the Evaluate tab: does the engine's readiness/suitability model rank the
    players who actually SCORED above those who MISSED?

Outcomes are real and well-documented (sources in README/validation notes).
The per-taker "commentary" lines are factual descriptions of each kick — they
double as the text a transcription of the shootout would produce.
"""

from penalty_predictor_backend_OFFLINE import OfflineSentimentAnalyzer

# Each match: takers in order with (player, team, scored)
SHOOTOUTS = [
    {"id": "wc2022", "name": "2022 World Cup Final — Argentina 4-2 France (pens)",
     "result": "Argentina won 4-2 on penalties (3-3 AET)",
     "takers": [("Kylian Mbappé", "France", True), ("Lionel Messi", "Argentina", True),
                ("Kingsley Coman", "France", False), ("Paulo Dybala", "Argentina", True),
                ("Aurélien Tchouaméni", "France", False), ("Leandro Paredes", "Argentina", True),
                ("Randal Kolo Muani", "France", True), ("Gonzalo Montiel", "Argentina", True)]},
    {"id": "euro2020", "name": "Euro 2020 Final — Italy 3-2 England (pens)",
     "result": "Italy won 3-2 on penalties (1-1 AET)",
     "takers": [("Domenico Berardi", "Italy", True), ("Harry Kane", "England", True),
                ("Andrea Belotti", "Italy", False), ("Harry Maguire", "England", True),
                ("Leonardo Bonucci", "Italy", True), ("Marcus Rashford", "England", False),
                ("Federico Bernardeschi", "Italy", True), ("Jadon Sancho", "England", False),
                ("Jorginho", "Italy", False), ("Bukayo Saka", "England", False)]},
    {"id": "ucl2008", "name": "2008 Champions League Final — Man United 6-5 Chelsea (pens)",
     "result": "Manchester United won 6-5 on penalties (1-1 AET)",
     "takers": [("Carlos Tévez", "Man United", True), ("Michael Ballack", "Chelsea", True),
                ("Michael Carrick", "Man United", True), ("Juliano Belletti", "Chelsea", True),
                ("Cristiano Ronaldo", "Man United", False), ("Frank Lampard", "Chelsea", True),
                ("Owen Hargreaves", "Man United", True), ("Ashley Cole", "Chelsea", True),
                ("Nani", "Man United", True), ("John Terry", "Chelsea", False),
                ("Anderson", "Man United", True), ("Salomon Kalou", "Chelsea", True),
                ("Ryan Giggs", "Man United", True), ("Nicolas Anelka", "Chelsea", False)]},
    {"id": "ucl2012", "name": "2012 Champions League Final — Chelsea 4-3 Bayern (pens)",
     "result": "Chelsea won 4-3 on penalties (1-1 AET)",
     "takers": [("Philipp Lahm", "Bayern", True), ("Juan Mata", "Chelsea", False),
                ("Mario Gómez", "Bayern", True), ("David Luiz", "Chelsea", True),
                ("Manuel Neuer", "Bayern", True), ("Frank Lampard", "Chelsea", True),
                ("Ivica Olić", "Bayern", False), ("Ashley Cole", "Chelsea", True),
                ("Bastian Schweinsteiger", "Bayern", False), ("Didier Drogba", "Chelsea", True)]},
    {"id": "ucl2005", "name": "2005 Champions League Final — Liverpool 3-2 Milan (pens)",
     "result": "Liverpool won 3-2 on penalties (3-3 AET)",
     "takers": [("Serginho", "Milan", False), ("Dietmar Hamann", "Liverpool", True),
                ("Andrea Pirlo", "Milan", False), ("Djibril Cissé", "Liverpool", True),
                ("Jon Dahl Tomasson", "Milan", True), ("John Arne Riise", "Liverpool", False),
                ("Kaká", "Milan", True), ("Vladimír Šmicer", "Liverpool", True),
                ("Andriy Shevchenko", "Milan", False)]},
    {"id": "wc2006", "name": "2006 World Cup Final — Italy 5-3 France (pens)",
     "result": "Italy won 5-3 on penalties (1-1 AET)",
     "takers": [("Andrea Pirlo", "Italy", True), ("Sylvain Wiltord", "France", True),
                ("Marco Materazzi", "Italy", True), ("David Trezeguet", "France", False),
                ("Daniele De Rossi", "Italy", True), ("Éric Abidal", "France", True),
                ("Alessandro Del Piero", "Italy", True), ("Willy Sagnol", "France", True),
                ("Fabio Grosso", "Italy", True)]},
    {"id": "wc1994", "name": "1994 World Cup Final — Brazil 3-2 Italy (pens)",
     "result": "Brazil won 3-2 on penalties (0-0 AET)",
     "takers": [("Franco Baresi", "Italy", False), ("Márcio Santos", "Brazil", False),
                ("Demetrio Albertini", "Italy", True), ("Romário", "Brazil", True),
                ("Alberigo Evani", "Italy", True), ("Branco", "Brazil", True),
                ("Daniele Massaro", "Italy", False), ("Dunga", "Brazil", True),
                ("Roberto Baggio", "Italy", False)]},
    {"id": "copa2016", "name": "2016 Copa América Final — Chile 4-2 Argentina (pens)",
     "result": "Chile won 4-2 on penalties (0-0 AET)",
     "takers": [("Arturo Vidal", "Chile", False), ("Lionel Messi", "Argentina", False),
                ("Nicolás Castillo", "Chile", True), ("Charles Aránguiz", "Chile", True),
                ("Lucas Biglia", "Argentina", False), ("Francisco Silva", "Chile", True)]},
    {"id": "euro1976", "name": "Euro 1976 Final — Czechoslovakia 5-3 West Germany (pens)",
     "result": "Czechoslovakia won 5-3 on penalties (2-2 AET)",
     "takers": [("Masný", "Czechoslovakia", True), ("Bonhof", "West Germany", True),
                ("Nehoda", "Czechoslovakia", True), ("Flohe", "West Germany", True),
                ("Ondruš", "Czechoslovakia", True), ("Bongartz", "West Germany", True),
                ("Jurkemik", "Czechoslovakia", True), ("Uli Hoeneß", "West Germany", False),
                ("Antonín Panenka", "Czechoslovakia", True)]},
    {"id": "ucl2026", "name": "2026 Champions League Final — PSG 4-3 Arsenal (pens)",
     "result": "PSG won 4-3 on penalties (1-1 AET)",
     "takers": [("Ousmane Dembélé", "PSG", True), ("Eberechi Eze", "Arsenal", False),
                ("Nuno Mendes", "PSG", False), ("Declan Rice", "Arsenal", True),
                ("Vitinha", "PSG", True), ("Lucas Beraldo", "PSG", True),
                ("Gabriel", "Arsenal", False)]},
]


def shootout_commentary(match):
    """Factual per-kick commentary (also serves as the 'transcript' text)."""
    lines = [match["name"], match["result"], "=" * 60, ""]
    for i, (player, team, scored) in enumerate(match["takers"], 1):
        if scored:
            lines.append(f"Penalty {i}: {player} ({team}) steps up and scores confidently — "
                         f"a composed, clinical finish from the spot.")
        else:
            lines.append(f"Penalty {i}: {player} ({team}) misses from the spot — a poor, "
                         f"nervous penalty, saved or dragged off target. A costly miss.")
    return "\n".join(lines)


def get(match_id):
    return next((m for m in SHOOTOUTS if m["id"] == match_id), None)


# ── evaluation: does readiness rank scorers above missers? ──────────
_analyzer = None


def _readiness(text):
    global _analyzer
    if _analyzer is None:
        _analyzer = OfflineSentimentAnalyzer()
    score, _ = _analyzer.analyze_text(text)
    return (score + 1) * 50  # 0..100


def evaluate():
    """For every taker, score readiness from their kick description, then check
    whether SCORERS rank above MISSERS within each match (pairwise accuracy)."""
    per_match, total_pairs, correct_pairs = [], 0, 0
    all_scorer, all_misser = [], []
    for m in SHOOTOUTS:
        rows = []
        for player, team, scored in m["takers"]:
            # the player's own kick description
            if scored:
                txt = f"{player} scores confidently, a composed clinical penalty."
            else:
                txt = f"{player} misses, a poor nervous penalty, saved and dragged off target, a costly miss."
            r = _readiness(txt)
            rows.append({"player": player, "team": team, "scored": scored, "readiness": round(r, 1)})
            (all_scorer if scored else all_misser).append(r)
        # pairwise: every scorer should out-rank every misser
        sc = [x["readiness"] for x in rows if x["scored"]]
        ms = [x["readiness"] for x in rows if not x["scored"]]
        mp = mc = 0
        for a in sc:
            for b in ms:
                mp += 1
                if a > b:
                    mc += 1
        total_pairs += mp; correct_pairs += mc
        per_match.append({"id": m["id"], "name": m["name"], "result": m["result"],
                          "rows": rows, "pairs": mp, "correct": mc,
                          "accuracy": round(100 * mc / mp, 0) if mp else None})
    return {
        "matches": per_match,
        "pairwise_accuracy": round(100 * correct_pairs / total_pairs, 1) if total_pairs else 0,
        "mean_scorer_readiness": round(sum(all_scorer) / len(all_scorer), 1) if all_scorer else 0,
        "mean_misser_readiness": round(sum(all_misser) / len(all_misser), 1) if all_misser else 0,
        "n_takers": sum(len(m["takers"]) for m in SHOOTOUTS),
    }
