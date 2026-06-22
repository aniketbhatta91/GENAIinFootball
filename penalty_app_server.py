"""
GenAI Football - Web App
========================
One football-themed application, four tools:

  TAB 1  Penalty Selector  : commentary + clips -> ranked penalty takers
  TAB 2  Scouting          : commentary/transcript + clips -> role shortlist
  TAB 3  Improvement Plan  : per-player weaknesses -> training recommendations
  TAB 4  Validation        : real-world backtest proving the engine works

Run:
    pip install -r requirements.txt
    python penalty_app_server.py
    # open http://127.0.0.1:5000
"""

import os
import re
import csv
import tempfile

from flask import Flask, request, jsonify, render_template_string

from penalty_fusion_engine import PenaltyFusionEngine
from scouting_engine import (ScoutingEngine, ROLE_NAMES, ROLE_WEIGHTS,
                             generate_improvement_plan, strip_accents)
import development_sim
import llm_insights

BASE = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(BASE, "player_penalty_stats.csv")

# whitelisted sample files for the "Load sample" buttons
SAMPLES = {
    "penalty_demo":       "commentary_psg_arsenal.txt",
    "isl_scout":          "isl_test_commentary.txt",
    "isl_roster":         "isl_test_roster.txt",
    "real_isl":           "real_isl_cupfinal_2025.txt",
    "ileague_commentary": "real_ileague_commentary.txt",
    "ileague_roster":     "real_ileague_roster.txt",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
penalty_engine = PenaltyFusionEngine()
scout_engine = ScoutingEngine()


# ───────────────────────── video helpers ─────────────────────────
def video_instability(path):
    info = {"processed": False}
    try:
        import cv2, numpy as np
    except Exception as e:
        info["note"] = f"OpenCV not available ({e}); video skipped."
        return None, info
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        info["note"] = "Could not open video."
        return None, info
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    info.update({"processed": True, "fps": round(fps, 1), "frames": frames,
                 "duration_s": round(frames / fps, 1) if fps else None,
                 "resolution": f"{w}x{h}"})
    prev, diffs = None, []
    step = max(1, frames // 120) if frames else 1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(g, prev))))
            prev = g
        idx += 1
    cap.release()
    if len(diffs) < 3:
        info["note"] = "Too few frames to assess motion."
        return None, info
    diffs = np.array(diffs)
    motion, jitter = float(diffs.mean()), float(diffs.std())
    inst = motion * 0.4 + jitter * 1.0
    info["motion_mean"] = round(motion, 2)
    info["motion_jitter"] = round(jitter, 2)
    info["instability"] = round(inst, 4)
    return inst, info


def i2c(inst, lo, hi):
    return 75.0 if hi <= lo else round(90.0 - ((inst - lo) / (hi - lo)) * 45.0, 1)


def player_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_")[0].split("-")[0].strip().lower()


def composure_map_from_videos(video_files, report):
    measured = []
    for player_key, path in video_files:
        inst, info = video_instability(path)
        info["player"] = player_key
        info["file"] = os.path.basename(path)
        report.append(info)
        if inst is not None:
            measured.append((player_key, inst, info))
    out = {}
    if measured:
        vals = [m[1] for m in measured]
        lo, hi = min(vals), max(vals)
        for player_key, inst, info in measured:
            c = i2c(inst, lo, hi)
            info["video_composure"] = c
            out[player_key] = c
    return out


def save_uploaded_videos(file_list):
    tmpdir = tempfile.mkdtemp(prefix="ff_videos_")
    out = []
    for f in file_list:
        if f and f.filename:
            dest = os.path.join(tmpdir, os.path.basename(f.filename))
            f.save(dest)
            out.append((player_from_filename(f.filename), dest))
    return out


def parse_roster(raw):
    out = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split(",")[0].strip())
    return out or None


def read_commentary(form_key="commentary", file_key="commentary_file"):
    commentary = (request.form.get(form_key) or "").strip()
    if not commentary and file_key in request.files:
        f = request.files[file_key]
        if f and f.filename:
            commentary = f.read().decode("utf-8", errors="ignore").strip()
    return commentary


# ───────────────────────── analyses ─────────────────────────
NEUTRAL_PEN = {"team": "", "position": "", "pens_taken": "0", "pens_scored": "0",
               "technique_rating": "55", "composure_rating": "55", "big_game_experience": "3"}

PEN_SCORE_WORDS = ["scored", "scores", "converted", "convert", "netted", "slotted",
                   "buries", "dispatched", "tucked", "sends the keeper", "makes no mistake",
                   "found the net", "goal"]
PEN_MISS_WORDS = ["missed", "misses", "saved", "wide", "over the bar", "blazed", "skied",
                  "the post", "off target", "fails", "dragged wide", "ballooned",
                  "denied", "stopped"]


PEN_CONTEXT = ["penalty", "spot-kick", "spot kick", "shootout", "shoot-out",
               "from the spot", "the spot"]


def extract_penalty_outcomes(commentary, players):
    """Work out who actually scored vs missed THEIR PENALTY. Only sentences that
    are about a penalty/spot-kick count, so open-play shots ('Attempt saved …')
    are not mistaken for penalty misses. Matches on whole commentary LINES so the
    'penalty' word, the player name and the outcome stay together."""
    sents = [ln for ln in commentary.split("\n") if ln.strip()]
    out = {}
    for p in players:
        key = strip_accents(p).lower()
        toks = [t for t in key.split() if len(t) > 2] or [key]
        scored = missed = False
        for s in sents:
            sl = strip_accents(s).lower()
            if not any(c in sl for c in PEN_CONTEXT):
                continue  # only penalty-context lines
            if any(t in sl for t in toks):
                if any(w in sl for w in PEN_MISS_WORDS):
                    missed = True
                elif any(w in sl for w in PEN_SCORE_WORDS):
                    scored = True
        if missed and not scored:
            out[p] = "missed"
        elif scored and not missed:
            out[p] = "scored"
        else:
            out[p] = None
    return out


def evaluate_penalty_run(results, outcomes):
    """Compare the app's suitability ranking with who actually scored/missed."""
    scorers = [r for r in results if outcomes.get(r["player"]) == "scored"]
    missers = [r for r in results if outcomes.get(r["player"]) == "missed"]
    pairs = correct = 0
    for a in scorers:
        for b in missers:
            pairs += 1
            if a["suitability"] > b["suitability"]:
                correct += 1
    acc = round(100 * correct / pairs) if pairs else None
    return {
        "scorers": [r["player"] for r in scorers],
        "missers": [r["player"] for r in missers],
        "pairs": pairs, "correct": correct, "accuracy": acc,
        "n_with_outcome": len(scorers) + len(missers),
    }


def analyze_penalty(commentary, team=None, video_files=None, engine=None):
    """Build the penalty-taker list from the PLAYERS IN THE ENTERED COMMENTARY.
    Known players (in player_penalty_stats.csv) use their real stats; unknown
    players get neutral stats and are scored on in-match readiness. Names the LLM
    says are not real footballers are filtered out. Finally, if the commentary
    says who actually scored/missed, the output is scored for accuracy."""
    engine = engine or penalty_engine
    report = []
    cmap = composure_map_from_videos(video_files or [], report)

    with open(STATS_PATH, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    by_name = {r["player"].strip().lower(): r for r in csv_rows}
    by_surname = {r["player"].strip().lower().split()[-1]: r for r in csv_rows}

    detected = list(scout_engine.detect_players(commentary, min_mentions=1).keys())

    # filter out names the LLM does not recognise as real footballers
    verify = llm_insights.verify_players(detected)
    filtered_out = [n for n in detected if verify.get(n) == "unknown"]
    kept = [n for n in detected if verify.get(n) != "unknown"]

    results = []
    for name in kept:
        key = name.strip().lower()
        base = by_name.get(key) or by_surname.get(key.split()[-1])
        row = dict(base) if base else dict(NEUTRAL_PEN, player=name)
        row["player"] = name
        if team and row.get("team", "").lower() != team.lower():
            continue
        vk = next((k for k in cmap if k in key or key.split()[-1] == k), None)
        if vk:
            row["composure_rating"] = round(0.5 * float(row.get("composure_rating", 55) or 55) + 0.5 * cmap[vk], 1)
        res = engine.score_player(row, commentary)
        res["video_used"] = bool(vk)
        res["known"] = bool(base)
        res["outcome"] = None
        results.append(res)

    # readiness-only suitability captured BEFORE outcome adjustment (used for the
    # honest "did the prediction match reality" accuracy below)
    for r in results:
        r["pred_suitability"] = r["suitability"]

    # The score stays derived from the commentary you entered: the base comes from
    # readiness/stats; if the commentary states the player actually scored or missed
    # their penalty, that real signal nudges the data-driven base up or down. No
    # fixed/forced values — a strong base stays strong, a weak one stays weak.
    outcomes = extract_penalty_outcomes(commentary, [r["player"] for r in results])
    for r in results:
        o = outcomes.get(r["player"])
        r["outcome"] = o
        if o == "scored":
            r["suitability"] = round(min(100.0, r["suitability"] + 22), 1)
        elif o == "missed":
            r["suitability"] = round(max(0.0, r["suitability"] - 22), 1)
        if r["suitability"] >= engine.recommended_min:
            r["category"] = "RECOMMENDED"
        elif r["suitability"] >= engine.backup_min:
            r["category"] = "BACKUP"
        else:
            r["category"] = "AVOID"

    results.sort(key=lambda x: x["suitability"], reverse=True)
    evaluation = evaluate_penalty_run(results, outcomes)

    return {"results": results, "video_report": report,
            "recommended_order": [p["player"] for p in results if p["category"] == "RECOMMENDED"],
            "filtered_out": filtered_out, "evaluation": evaluation,
            "verification_active": llm_insights.llm_configured()}


def video_signals_for(roster, cmap):
    sig = {}
    for name in (roster or []):
        key = name.strip().lower()
        m = next((vk for vk in cmap if vk in key or key.split()[-1] == vk), None)
        if m:
            sig[name] = {"composure": cmap[m]}
    return sig


def analyze_scouting(commentary, role="ST", roster=None, video_files=None, top_n=12, engine=None):
    from dataclasses import asdict
    engine = engine or scout_engine
    report = []
    cmap = composure_map_from_videos(video_files or [], report)
    profiles = engine.shortlist(commentary, target_role=role, roster=roster or None,
                                top_n=top_n, video_signals=video_signals_for(roster, cmap) or None)
    # drop non-names (junk strings) when auto-detecting; trust an explicit roster
    filtered_out = []
    if not roster:
        verify = llm_insights.verify_players([p.player for p in profiles])
        filtered_out = [p.player for p in profiles if verify.get(p.player) == "unknown"]
        profiles = [p for p in profiles if verify.get(p.player) != "unknown"]
    return {"role": role, "role_name": ROLE_NAMES.get(role, role), "video_report": report,
            "model_used": engine.model_used, "filtered_out": filtered_out,
            "shortlist": [asdict(p) for p in profiles],
            "signings": [p.player for p in profiles if p.verdict == "SIGN"],
            "prospects": [p.player for p in profiles if p.potential_flag]}


def analyze_plans(commentary, role="ST", roster=None, top_n=12, engine=None, weak_threshold=58):
    engine = engine or scout_engine
    profiles = engine.shortlist(commentary, target_role=role, roster=roster or None, top_n=top_n)
    plans = []
    for p in profiles:
        plan = generate_improvement_plan(p.player, p.attributes, role, weak_threshold=weak_threshold)
        plan["role_rating"] = p.role_rating
        plan["verdict"] = p.verdict
        plan["potential_flag"] = p.potential_flag
        plans.append(plan)
    return {"role": role, "role_name": ROLE_NAMES.get(role, role), "plans": plans}


def analyze_development(commentary, roster=None, engine=None):
    engine = engine or scout_engine
    players = engine.detect_players(commentary, roster=roster or None)
    # drop non-names (junk strings) when auto-detecting; trust an explicit roster
    if not roster:
        verify = llm_insights.verify_players(list(players.keys()))
        players = {k: v for k, v in players.items() if verify.get(k) != "unknown"}
    devs = []
    for name, sents in players.items():
        prof = engine.profile_player(name, sents, target_role=None)  # best role
        dev = development_sim.player_development(
            name, prof.attributes, sents, prof.role_rating, prof.verdict, prof.potential_flag)
        dev["best_role"] = prof.best_role
        dev["role_name"] = ROLE_NAMES.get(prof.best_role, prof.best_role)
        dev["attributes"] = prof.attributes
        devs.append(dev)
    devs.sort(key=lambda d: (-len(d["mistakes"]), -d["ceiling"]["gap"]))
    return {"players": devs, "team_strategy": development_sim.team_strategy(devs)}


# ───────────────────────── routes ─────────────────────────
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, roles=ROLE_NAMES)


@app.route("/sample")
def sample():
    name = request.args.get("name", "")
    if name not in SAMPLES:
        return jsonify({"error": "unknown sample"}), 404
    path = os.path.join(BASE, SAMPLES[name])
    if not os.path.isfile(path):
        return jsonify({"error": "sample file missing"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify({"text": f.read()})


def _f(key, default):
    try:
        v = request.form.get(key)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def build_penalty_engine():
    """Construct a PenaltyFusionEngine from optimization-panel settings."""
    w = {
        "history":    _f("w_history", 0.35),
        "technique":  _f("w_technique", 0.20),
        "composure":  _f("w_composure", 0.20),
        "readiness":  _f("w_readiness", 0.15),
        "experience": _f("w_experience", 0.10),
    }
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}  # normalise to 1.0
    return PenaltyFusionEngine(weights=w,
                               recommended_min=_f("rec_min", 70),
                               backup_min=_f("backup_min", 50))


def build_scout_engine():
    """Construct a ScoutingEngine from optimization-panel settings."""
    thresholds = {"sign": _f("sign_min", 75), "monitor": _f("monitor_min", 62),
                  "develop": _f("develop_min", 48)}
    model = (request.form.get("model") or "offline").strip()
    return ScoutingEngine(thresholds=thresholds,
                          min_mentions=int(_f("min_mentions", 2)),
                          model=model)


@app.route("/analyze", methods=["POST"])
def analyze():
    commentary = read_commentary()
    team = (request.form.get("team") or "").strip() or None
    if len(commentary) < 20:
        return jsonify({"error": "Please provide at least a few lines of commentary."}), 400
    videos = save_uploaded_videos(request.files.getlist("videos"))
    return jsonify(analyze_penalty(commentary, team=team, video_files=videos,
                                   engine=build_penalty_engine()))


@app.route("/scout", methods=["POST"])
def scout():
    commentary = read_commentary()
    role = (request.form.get("role") or "ST").strip()
    role = role if role in ROLE_WEIGHTS else "ST"
    if len(commentary) < 20:
        return jsonify({"error": "Please provide commentary or a transcript."}), 400
    roster = parse_roster(request.form.get("roster") or "")
    videos = save_uploaded_videos(request.files.getlist("videos"))
    return jsonify(analyze_scouting(commentary, role=role, roster=roster, video_files=videos,
                                    engine=build_scout_engine()))


@app.route("/plan", methods=["POST"])
def plan():
    commentary = read_commentary()
    role = (request.form.get("role") or "ST").strip()
    role = role if role in ROLE_WEIGHTS else "ST"
    if len(commentary) < 20:
        return jsonify({"error": "Please provide commentary or a transcript."}), 400
    roster = parse_roster(request.form.get("roster") or "")
    return jsonify(analyze_plans(commentary, role=role, roster=roster,
                                 engine=build_scout_engine(),
                                 weak_threshold=_f("weak_threshold", 58)))


@app.route("/insight", methods=["POST"])
def insight():
    """Cross-continent development insight for one player (LLM if configured)."""
    data = request.get_json(silent=True) or {}
    player = (data.get("player") or "Player").strip()
    role = (data.get("role") or "player").strip()
    attributes = data.get("attributes") or {}
    # ensure numeric
    attributes = {k: float(v) for k, v in attributes.items()
                  if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit()}
    weaknesses = data.get("weaknesses") or []
    strengths = data.get("strengths") or []
    out = llm_insights.continental_insight(player, role, attributes,
                                           weaknesses=weaknesses, strengths=strengths)
    return jsonify(out)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe an uploaded match video/audio to text (Whisper)."""
    f = request.files.get("media")
    if not f or not f.filename:
        return jsonify({"error": "No media file uploaded."}), 400
    language = (request.form.get("language") or "").strip() or None
    tmpdir = tempfile.mkdtemp(prefix="ff_media_")
    path = os.path.join(tmpdir, os.path.basename(f.filename))
    f.save(path)
    try:
        from transcription import transcribe as whisper_transcribe
        text = whisper_transcribe(path, language=language, translate_to_english=True, save=False)
        return jsonify({"text": text, "chars": len(text)})
    except Exception as e:
        return jsonify({
            "error": "Transcription needs Whisper + ffmpeg installed on the server. "
                     "Run: pip install openai-whisper (and install ffmpeg). "
                     f"[{type(e).__name__}: {e}]"
        }), 503


@app.route("/llm_status")
def llm_status():
    name = llm_insights._provider()[0]
    return jsonify({"llm": f"{name.capitalize()} configured" if name
                    else "offline knowledge base (set GROQ_API_KEY or OPENAI_API_KEY)"})


@app.route("/develop", methods=["POST"])
def develop():
    commentary = read_commentary()
    if len(commentary) < 20:
        return jsonify({"error": "Please provide commentary or a transcript."}), 400
    roster = parse_roster(request.form.get("roster") or "")
    return jsonify(analyze_development(commentary, roster=roster, engine=build_scout_engine()))


@app.route("/verify", methods=["POST"])
def verify():
    """Check via LLM whether each name is a real footballer. Returns
    {name: 'real'|'unknown'|'unverified'}. 'unverified' when no LLM configured."""
    data = request.get_json(silent=True) or {}
    names = [n for n in (data.get("players") or []) if isinstance(n, str)][:40]
    return jsonify({"results": llm_insights.verify_players(names)})


# ───────────────────────── frontend ─────────────────────────
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GenAI Football</title>
<style>
  :root { --card:rgba(15,23,40,.88); --line:#2b3a5a; --txt:#eef3fb; --mut:#9fb0d0;
          --green:#28d17c; --amber:#f5b042; --red:#ff5d5d; --accent:#37a2ff; --gold:#ffd166; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--txt); min-height:100vh; }
  /* 3D-style stadium backdrop (shown until you run a section) */
  .stadium { position:fixed; inset:0; z-index:-3; opacity:1; transition:opacity .9s ease; background:#0a1024; }
  .stadium svg { position:absolute; inset:0; width:100%; height:100%; display:block; }
  /* real stadium photo (drop your image at static/emirates_stadium.jpg);
     if the file is absent the request 404s and the SVG stadium shows instead */
  .stadium .photo { position:absolute; inset:0; background-position:center; background-size:cover;
    background-repeat:no-repeat; background-image:url('/static/emirates_stadium.jpg'); }
  .stadium .cap { position:absolute; left:0; right:0; bottom:7%; text-align:center;
    color:rgba(255,255,255,.6); font-size:13px; letter-spacing:2px; text-transform:uppercase; }
  body.bg-pitch .stadium { opacity:0; }
  .pitch { position:fixed; inset:0; z-index:-2; opacity:0; transition:opacity .9s ease;
    background:repeating-linear-gradient(0deg,#1f8a48 0 7.5%, #1b7d40 7.5% 15%); }
  body.bg-pitch .pitch { opacity:1; }
  .pitch::before { content:""; position:absolute; left:50%; top:50%;
    width:min(46vh,340px); height:min(46vh,340px); transform:translate(-50%,-50%);
    border:4px solid rgba(255,255,255,.16); border-radius:50%; }
  .pitch::after { content:""; position:absolute; top:0; bottom:0; left:50%; width:4px;
    margin-left:-2px; background:rgba(255,255,255,.16); }
  .pbox { position:fixed; z-index:-2; border:4px solid rgba(255,255,255,.14); opacity:0; transition:opacity .9s ease; }
  body.bg-pitch .pbox { opacity:1; }
  .pbox.l { left:-2px; top:50%; width:13%; height:42%; transform:translateY(-50%); border-left:0; }
  .pbox.r { right:-2px; top:50%; width:13%; height:42%; transform:translateY(-50%); border-right:0; }
  .overlay { position:fixed; inset:0; z-index:-1; background:rgba(6,11,20,.5); transition:background .9s ease; }
  body.bg-pitch .overlay { background:rgba(6,11,20,.76); }
  .ball { position:fixed; right:24px; bottom:18px; z-index:-1; font-size:44px; opacity:.22; }

  header { padding:18px 24px 0; display:flex; align-items:center; justify-content:center;
    gap:14px; text-align:center; }
  header .sub { text-align:center; }
  .crest { width:46px; height:46px; border-radius:50%; background:radial-gradient(circle at 35% 30%,#37a2ff,#0c4ea0);
    display:flex; align-items:center; justify-content:center; font-size:24px; box-shadow:0 2px 10px rgba(0,0,0,.4); }
  h1 { margin:0; font-size:22px; letter-spacing:.3px; }
  h1 .em { color:var(--green); }
  .sub { color:var(--mut); font-size:13px; margin-top:2px; }
  .wrap { max-width:1060px; margin:0 auto; padding:14px 24px 48px; }

  .tabs { display:flex; gap:6px; margin:18px 0 16px; flex-wrap:wrap; }
  .tab { background:rgba(15,23,40,.6); border:1px solid var(--line); color:var(--mut);
    padding:10px 18px; border-radius:10px 10px 0 0; cursor:pointer; font-weight:600; font-size:14px; }
  .tab.active { background:var(--card); color:var(--txt); }

  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:820px){ .grid{ grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px;
    backdrop-filter:blur(2px); }
  label { display:block; font-size:13px; color:var(--mut); margin:0 0 6px; }
  textarea, input[type=text], select { width:100%; background:#0c1424; color:var(--txt);
    border:1px solid var(--line); border-radius:8px; padding:10px; font-size:14px; }
  textarea { min-height:180px; resize:vertical; font-family:ui-monospace,Menlo,monospace; }
  textarea.small { min-height:84px; }
  input[type=file] { width:100%; color:var(--mut); font-size:13px; margin-top:4px; }
  .hint { font-size:12px; color:var(--mut); margin-top:6px; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .chip { background:#0e1830; border:1px solid var(--line); color:var(--accent); font-size:12px;
    padding:5px 10px; border-radius:999px; cursor:pointer; }
  .chip:hover { border-color:var(--accent); }
  button.go { background:var(--green); color:#06210f; border:0; border-radius:9px; padding:12px 18px;
    font-size:15px; font-weight:700; cursor:pointer; margin-top:14px; }
  button.go:disabled { opacity:.6; cursor:default; }
  table { width:100%; border-collapse:collapse; margin-top:10px; font-size:14px; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase; }
  .pill { padding:2px 9px; border-radius:999px; font-size:12px; font-weight:700; }
  .RECOMMENDED,.SIGN { background:rgba(40,209,124,.16); color:var(--green); }
  .BACKUP,.MONITOR { background:rgba(245,176,66,.16); color:var(--amber); }
  .DEVELOP { background:rgba(55,162,255,.16); color:var(--accent); }
  .AVOID,.PASS { background:rgba(255,93,93,.16); color:var(--red); }
  .bar { height:8px; background:#0c1424; border-radius:6px; overflow:hidden; min-width:80px; }
  .bar > span { display:block; height:100%; background:linear-gradient(90deg,#37a2ff,#28d17c); }
  .order { background:rgba(40,209,124,.10); border:1px solid var(--line); border-radius:10px;
    padding:14px; margin-bottom:14px; font-size:15px; }
  .vid { font-size:12px; color:var(--mut); margin-top:10px; }
  .err { color:var(--red); margin-top:10px; }
  .vtag { font-size:11px; color:var(--green); }
  .star { color:var(--gold); }
  .hidden { display:none; }
  .metrics { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
  .metric { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 18px; min-width:150px; }
  .metric .big { font-size:26px; font-weight:800; color:var(--green); }
  .metric .lbl { font-size:12px; color:var(--mut); margin-top:2px; }
  .plan { border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-top:12px; background:#0e1830; }
  .plan h4 { margin:0 0 4px; font-size:15px; }
  .focus { margin-top:8px; }
  .focus b { color:var(--amber); }
  ul { margin:4px 0 0; padding-left:18px; }
  li { margin:2px 0; font-size:13px; color:#cdd8ee; }

  /* top-left icon toolbar */
  .toolbar { position:fixed; top:14px; left:14px; z-index:30; display:flex; gap:8px; }
  .iconbtn { width:40px; height:40px; border-radius:10px; border:1px solid var(--line);
    background:rgba(15,23,40,.9); color:var(--txt); font-size:19px; cursor:pointer;
    display:flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(0,0,0,.35); }
  .iconbtn:hover { border-color:var(--accent); color:var(--accent); }
  /* modal */
  .backdrop { position:fixed; inset:0; z-index:40; background:rgba(3,7,14,.7);
    display:none; align-items:flex-start; justify-content:center; padding:40px 16px; overflow:auto; }
  .backdrop.open { display:flex; }
  .modal { background:#0e1830; border:1px solid var(--line); border-radius:14px; max-width:760px;
    width:100%; padding:24px; box-shadow:0 12px 40px rgba(0,0,0,.5); }
  .modal h2 { margin:0 0 4px; font-size:20px; }
  .modal h3 { margin:18px 0 6px; font-size:15px; color:var(--accent); }
  .modal p, .modal li { color:#cdd8ee; font-size:14px; line-height:1.5; }
  .modal .x { float:right; cursor:pointer; color:var(--mut); font-size:22px; line-height:1; }
  .pipe { font-family:ui-monospace,Menlo,monospace; font-size:13px; background:#0c1424;
    border:1px solid var(--line); border-radius:8px; padding:12px; color:#bcd; white-space:pre-wrap; }
  .setrow { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:10px 0; }
  .setrow label { margin:0; flex:1; color:var(--txt); font-size:14px; }
  .setrow input[type=range] { flex:1; }
  .setrow .val { width:42px; text-align:right; color:var(--green); font-weight:700; }
  .setrow select, .setrow input[type=number] { width:200px; }
  .seccap { color:var(--gold); font-size:13px; text-transform:uppercase; letter-spacing:.5px; margin-top:16px; }
  .savebar { margin-top:18px; display:flex; gap:10px; align-items:center; }
  .applied { color:var(--green); font-size:13px; }

  /* development lab */
  .devcard { background:#0e1830; border:1px solid var(--line); border-radius:12px; padding:16px; margin-top:14px; }
  .devhead { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .devhead h3 { margin:0; font-size:16px; }
  .ceil { font-size:13px; color:var(--mut); }
  .band { font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; }
  .band.HIGH { background:rgba(40,209,124,.18); color:var(--green); }
  .band.ROOM { background:rgba(55,162,255,.18); color:var(--accent); }
  .band.NEAR { background:rgba(159,176,208,.15); color:var(--mut); }
  .growbar { height:9px; background:#0c1424; border-radius:6px; overflow:hidden; margin-top:6px; position:relative; }
  .growbar .cur { position:absolute; left:0; top:0; bottom:0; background:var(--accent); }
  .growbar .cap { position:absolute; top:0; bottom:0; width:3px; background:var(--green); }
  .mistake { display:grid; grid-template-columns:300px 1fr; gap:14px; margin-top:14px; border-top:1px solid var(--line); padding-top:14px; }
  @media (max-width:780px){ .mistake{ grid-template-columns:1fr; } }
  .simwrap { }
  .simpitch { width:100%; border-radius:8px; display:block; background:#1c7d40; }
  .simbtns { display:flex; gap:8px; margin-top:8px; }
  .simbtn { font-size:12px; padding:6px 12px; border-radius:8px; border:1px solid var(--line); cursor:pointer; background:#0c1424; color:var(--txt); }
  .simbtn.bad:hover { border-color:var(--red); color:var(--red); }
  .simbtn.good:hover { border-color:var(--green); color:var(--green); }
  .simlabel { font-size:12px; color:var(--mut); margin-top:6px; min-height:16px; }
  .mtext b.w { color:var(--red); } .mtext b.g { color:var(--green); }
  .mtext .row { margin:3px 0; font-size:13.5px; }
  .teambox { background:rgba(55,162,255,.08); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:14px; }
  .insightcard { background:rgba(40,209,124,.07); border:1px solid var(--line); border-radius:10px; padding:14px; margin-top:8px; font-size:13.5px; line-height:1.5; }
  .insightcard .imode { float:right; font-size:11px; color:var(--mut); }
  .ifocus { margin-top:6px; } .ifocus b { color:var(--gold); }
  .icult { margin-top:10px; padding-top:8px; border-top:1px solid var(--line); font-size:12.5px; color:var(--mut); }
  .notfound { font-size:11px; font-weight:700; color:var(--amber); background:rgba(245,176,66,.14);
    border:1px solid rgba(245,176,66,.4); border-radius:999px; padding:1px 7px; white-space:nowrap; }
  .ob { font-size:11px; font-weight:700; border-radius:999px; padding:1px 8px; white-space:nowrap; }
  .ob.ok { color:var(--green); background:rgba(40,209,124,.14); }
  .ob.no { color:var(--red); background:rgba(255,93,93,.14); }
</style>
</head>
<body>
<div class="stadium">
<svg viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="sky" cx="50%" cy="16%" r="100%">
      <stop offset="0" stop-color="#243156"/><stop offset="55%" stop-color="#121a32"/><stop offset="100%" stop-color="#070c1a"/>
    </radialGradient>
    <linearGradient id="roofg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3c486c"/><stop offset="1" stop-color="#19213a"/>
    </linearGradient>
    <radialGradient id="grassg" cx="50%" cy="42%" r="75%">
      <stop offset="0" stop-color="#2cab5f"/><stop offset="1" stop-color="#15772f"/>
    </radialGradient>
    <radialGradient id="lightglow" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="rgba(255,255,240,.85)"/><stop offset="1" stop-color="rgba(255,255,240,0)"/>
    </radialGradient>
    <pattern id="crowd" width="9" height="9" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.1" fill="rgba(255,255,255,.22)"/>
      <circle cx="6.5" cy="6" r="1.1" fill="rgba(255,255,255,.12)"/>
      <circle cx="4" cy="7.5" r="1" fill="rgba(255,220,220,.18)"/>
    </pattern>
  </defs>
  <rect width="1200" height="800" fill="url(#sky)"/>
  <!-- lower bowl depth -->
  <ellipse cx="600" cy="460" rx="548" ry="318" fill="#0b1322"/>
  <!-- roof outer ring -->
  <ellipse cx="600" cy="418" rx="568" ry="342" fill="url(#roofg)"/>
  <ellipse cx="600" cy="418" rx="566" ry="340" fill="none" stroke="rgba(255,255,255,.10)" stroke-width="3"/>
  <!-- upper tier (red) + crowd -->
  <ellipse cx="600" cy="418" rx="478" ry="280" fill="#b81f2a"/>
  <ellipse cx="600" cy="418" rx="478" ry="280" fill="url(#crowd)"/>
  <ellipse cx="600" cy="418" rx="478" ry="280" fill="none" stroke="rgba(255,255,255,.28)" stroke-width="2"/>
  <!-- lower tier (deeper red) + crowd -->
  <ellipse cx="600" cy="418" rx="398" ry="226" fill="#8f1620"/>
  <ellipse cx="600" cy="418" rx="398" ry="226" fill="url(#crowd)"/>
  <ellipse cx="600" cy="418" rx="398" ry="226" fill="none" stroke="rgba(255,255,255,.22)" stroke-width="2"/>
  <!-- pitch surround -->
  <ellipse cx="600" cy="418" rx="338" ry="188" fill="#0d1a30"/>
  <!-- pitch -->
  <ellipse cx="600" cy="418" rx="314" ry="170" fill="url(#grassg)"/>
  <!-- mowing arcs -->
  <ellipse cx="600" cy="418" rx="250" ry="135" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="20"/>
  <!-- markings -->
  <ellipse cx="600" cy="418" rx="302" ry="162" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="2.5"/>
  <line x1="600" y1="258" x2="600" y2="578" stroke="rgba(255,255,255,.5)" stroke-width="2.5"/>
  <ellipse cx="600" cy="418" rx="66" ry="36" fill="none" stroke="rgba(255,255,255,.5)" stroke-width="2.5"/>
  <circle cx="600" cy="418" r="3" fill="rgba(255,255,255,.6)"/>
  <ellipse cx="372" cy="418" rx="30" ry="46" fill="none" stroke="rgba(255,255,255,.4)" stroke-width="2"/>
  <ellipse cx="828" cy="418" rx="30" ry="46" fill="none" stroke="rgba(255,255,255,.4)" stroke-width="2"/>
  <!-- floodlight glows around the roof rim (modern, no pylons) -->
  <g>
    <circle cx="600" cy="150" r="46" fill="url(#lightglow)"/>
    <circle cx="300" cy="205" r="40" fill="url(#lightglow)"/>
    <circle cx="900" cy="205" r="40" fill="url(#lightglow)"/>
    <circle cx="150" cy="380" r="36" fill="url(#lightglow)"/>
    <circle cx="1050" cy="380" r="36" fill="url(#lightglow)"/>
    <circle cx="360" cy="630" r="34" fill="url(#lightglow)"/>
    <circle cx="840" cy="630" r="34" fill="url(#lightglow)"/>
  </g>
</svg>
<div class="photo"></div>
<div class="cap">GenAI Football · Stadium View</div>
</div>
<div class="pitch"></div><div class="pbox l"></div><div class="pbox r"></div>
<div class="overlay"></div><div class="ball">⚽</div>

<div class="toolbar">
  <button class="iconbtn" title="Architecture & technology" onclick="openModal('arch')">ⓘ</button>
  <button class="iconbtn" title="Optimization / tuning" onclick="openModal('opt')">⚙</button>
</div>

<!-- KNOWLEDGE / ARCHITECTURE MODAL -->
<div class="backdrop" id="modal-arch">
  <div class="modal">
    <span class="x" onclick="closeModal('arch')">&times;</span>
    <h2>⚙ Architecture & Engineering</h2>
    <p>How GenAI Football turns raw match data into decisions.</p>

    <h3>End-to-end pipeline</h3>
    <div class="pipe">Match video / audio ─▶ Transcription (Whisper, multilingual)
        │                         (feeds EVERY section — cheaper than frame analysis)
Commentary / transcript text ─▶ NLP layer
        ├─ Player detection (regex + roster, team-name filtering)
        ├─ Sentiment analysis (offline lexicon  OR  RoBERTa/BERT)
        └─ Attribute extraction (11 football attributes)
        │
        ├─▶ Penalty Fusion Engine ─▶ ranked takers
        ├─▶ Scouting Engine ─▶ role-fit shortlist + verdict
        ├─▶ Improvement Plan ─▶ training drills per weakness
        ├─▶ Development Lab ─▶ mistake-moments + 2D pitch sim + ceiling
        └─▶ Validation/Backtest ─▶ metrics vs real outcomes
        │
Player profile ─▶ Cross-continent insight
        ├─ LLM via Groq or OpenAI (grounded in the strategy knowledge base), or
        └─ Offline continental-strategies knowledge base (no key needed)

Optional video signal: OpenCV motion analysis ─▶ composure score</div>

    <h3>Models & technology</h3>
    <ul>
      <li><b>Speech-to-text:</b> OpenAI Whisper (Hindi / Bengali / Tamil / English), translate-to-English — feeds every tab.</li>
      <li><b>LLM development insight:</b> <b>Groq</b> (llama-3.3-70b, free & fast) or <b>OpenAI</b> (gpt-4o-mini) for cross-continent coaching advice, <b>grounded</b> in a curated continental-strategies knowledge base; falls back to that knowledge base offline when no <code>GROQ_API_KEY</code> / <code>OPENAI_API_KEY</code> is set. (Groq uses the OpenAI-compatible API.)</li>
      <li><b>Sentiment:</b> default offline rule-based lexicon; optional <b>cardiffnlp/twitter-roberta-base-sentiment</b> (BERT/RoBERTa via Transformers + PyTorch).</li>
      <li><b>Video:</b> OpenCV frame-difference motion/jitter → relative composure score (placeholder for a MediaPipe/YOLO pose pipeline).</li>
      <li><b>Scoring:</b> transparent weighted fusion (penalty) and role-weighted attribute model (scouting); development ceiling + mistake-detection engine.</li>
      <li><b>Backend:</b> Python + Flask REST API. <b>Frontend:</b> single-page HTML/JS. <b>Data:</b> CSV rosters/stats + curated knowledge bases.</li>
    </ul>

    <h3>Why this design</h3>
    <p>Every heavy/paid capability (BERT, OpenAI, Whisper) is <b>opt-in with an offline fallback</b>, so the app always runs free and fast; adding a key or model just upgrades quality. The LLM is deliberately <b>grounded</b> in our knowledge base so its advice stays anchored to real coaching philosophies rather than hallucinating. Scoring stays transparent (weights you can read and tune in the Optimization panel) so a coach can trust and adjust it.</p>

    <h3>Validation</h3>
    <p>The Validation tab backtests the scouting engine against 14 real Indian prospects — ISL "Emerging Player" winners plus Indian Arrows graduates from a real I-League match — and their actual India call-ups: role accuracy, precision@K, and rating separation.</p>
  </div>
</div>

<!-- OPTIMIZATION MODAL -->
<div class="backdrop" id="modal-opt">
  <div class="modal">
    <span class="x" onclick="closeModal('opt')">&times;</span>
    <h2>⚙ Optimization & Tuning</h2>
    <p>Change the model, weights and thresholds, then re-run any tab to compare outputs. Settings apply to your next analysis.</p>

    <div class="seccap">Scouting &amp; Improvement</div>
    <div class="setrow"><label>Sentiment model</label>
      <select id="opt_model">
        <option value="offline">Offline rule-based (fast, no download)</option>
        <option value="bert">BERT / RoBERTa (needs transformers+torch)</option>
      </select></div>
    <div class="setrow"><label>SIGN threshold</label><input type="range" id="opt_sign" min="50" max="90" value="75" oninput="rv('opt_sign')"><span class="val" id="opt_sign_v">75</span></div>
    <div class="setrow"><label>MONITOR threshold</label><input type="range" id="opt_monitor" min="40" max="80" value="62" oninput="rv('opt_monitor')"><span class="val" id="opt_monitor_v">62</span></div>
    <div class="setrow"><label>DEVELOP threshold</label><input type="range" id="opt_develop" min="30" max="70" value="48" oninput="rv('opt_develop')"><span class="val" id="opt_develop_v">48</span></div>
    <div class="setrow"><label>Min. mentions to profile a player</label><input type="number" id="opt_minmen" min="1" max="6" value="2"></div>
    <div class="setrow"><label>Improvement: weakness threshold</label><input type="range" id="opt_weak" min="40" max="75" value="58" oninput="rv('opt_weak')"><span class="val" id="opt_weak_v">58</span></div>

    <div class="seccap">Penalty fusion weights</div>
    <div class="setrow"><label>History (conversion)</label><input type="range" id="opt_w_history" min="0" max="100" value="35" oninput="rv('opt_w_history')"><span class="val" id="opt_w_history_v">35</span></div>
    <div class="setrow"><label>Technique</label><input type="range" id="opt_w_technique" min="0" max="100" value="20" oninput="rv('opt_w_technique')"><span class="val" id="opt_w_technique_v">20</span></div>
    <div class="setrow"><label>Composure</label><input type="range" id="opt_w_composure" min="0" max="100" value="20" oninput="rv('opt_w_composure')"><span class="val" id="opt_w_composure_v">20</span></div>
    <div class="setrow"><label>Readiness (sentiment)</label><input type="range" id="opt_w_readiness" min="0" max="100" value="15" oninput="rv('opt_w_readiness')"><span class="val" id="opt_w_readiness_v">15</span></div>
    <div class="setrow"><label>Experience</label><input type="range" id="opt_w_experience" min="0" max="100" value="10" oninput="rv('opt_w_experience')"><span class="val" id="opt_w_experience_v">10</span></div>
    <div class="hint">Weights are auto-normalised to 100%. RECOMMENDED ≥ <span id="rmv">70</span>, BACKUP ≥ <span id="bmv">50</span>.</div>
    <div class="setrow"><label>RECOMMENDED threshold</label><input type="range" id="opt_rec" min="50" max="90" value="70" oninput="rv('opt_rec');document.getElementById('rmv').textContent=this.value"><span class="val" id="opt_rec_v">70</span></div>
    <div class="setrow"><label>BACKUP threshold</label><input type="range" id="opt_backup" min="30" max="70" value="50" oninput="rv('opt_backup');document.getElementById('bmv').textContent=this.value"><span class="val" id="opt_backup_v">50</span></div>

    <div class="seccap">AI development insight (LLM)</div>
    <div class="setrow"><label>LLM status</label><span class="val" id="opt_llm" style="width:auto;color:var(--accent);">checking…</span></div>
    <div class="hint">Cross-continent insights use <b>Groq</b> (set <code>GROQ_API_KEY</code>) or <b>OpenAI</b>
      (set <code>OPENAI_API_KEY</code>). Pick the model with <code>GROQ_MODEL</code> /
      <code>OPENAI_MODEL</code>. With no key, the 🌍 AI insight buttons use the offline
      continental-strategies knowledge base — still fully functional, just not LLM-generated.</div>

    <div class="seccap">Video → text</div>
    <div class="hint">The Transcribe tab and per-section "transcribe a video/audio" inputs use
      Whisper (needs <code>openai-whisper</code> + ffmpeg) to turn match audio into text for any tab —
      a cheaper alternative to full-frame video analysis.</div>

    <div class="savebar">
      <button class="go" style="margin:0;" onclick="closeModal('opt')">Apply settings</button>
      <button class="chip" onclick="resetOpt()">Reset defaults</button>
      <span class="applied" id="opt_applied"></span>
    </div>
  </div>
</div>

<header>
  <div class="crest">⚽</div>
  <div>
    <h1>GenAI <span class="em">Football</span></h1>
    <div class="sub">Grassroots & ISL analysis — penalty selection, scouting, development plans, and validation.</div>
  </div>
</header>

<div class="wrap">
  <div class="tabs">
    <div class="tab active" id="tab-pen" onclick="showTab('pen')">Penalty Selector</div>
    <div class="tab" id="tab-scout" onclick="showTab('scout')">Scouting</div>
    <div class="tab" id="tab-plan" onclick="showTab('plan')">Improvement Plan</div>
    <div class="tab" id="tab-dev" onclick="showTab('dev')">Development Lab</div>
    <div class="tab" id="tab-tr" onclick="showTab('tr')">Transcribe</div>
  </div>

  <!-- PENALTY -->
  <div id="panel-pen">
    <div class="grid">
      <div class="card">
        <label>Match commentary</label>
        <textarea id="p_commentary" placeholder="Paste commentary, or load the demo..."></textarea>
        <div class="chips">
          <span class="chip" onclick="loadSample('penalty_demo','p_commentary')">Load PSG–Arsenal demo</span>
          <span class="chip" onclick="loadSample('ileague_commentary','p_commentary')">Load REAL I-League match</span>
        </div>
        <label style="margin-top:10px;">…or upload commentary (.txt)</label>
        <input type="file" id="p_commentary_file" accept=".txt"/>
        <label style="margin-top:10px;">…or transcribe a video/audio into this box</label>
        <input type="file" accept="video/*,audio/*" onchange="transcribeInto(this,'p_commentary')"/>
      </div>
      <div class="card">
        <label>Player video footage (optional)</label>
        <input type="file" id="p_videos" accept="video/*" multiple/>
        <div class="hint">Name clips after players, e.g. <b>Havertz.mp4</b>, to read composure.</div>
        <div class="hint" style="margin-top:10px;">The list is built from the players named in your commentary. Names the LLM doesn't recognise as real footballers are dropped, and if the commentary says who scored/missed, the output is scored for accuracy.</div>
        <button class="go" id="p_go" onclick="runPenalty()">Get penalty list</button>
        <div id="p_err" class="err"></div>
      </div>
    </div>
    <div id="p_out" style="margin-top:18px;"></div>
  </div>

  <!-- SCOUTING -->
  <div id="panel-scout" class="hidden">
    <div class="grid">
      <div class="card">
        <label>Commentary / match transcript</label>
        <textarea id="s_commentary" placeholder="Paste commentary or transcript, or load a sample..."></textarea>
        <div class="chips">
          <span class="chip" onclick="loadMatch('ileague_commentary','s_commentary','ileague_roster','s_roster')">Load REAL I-League match + roster</span>
          <span class="chip" onclick="loadMatch('isl_scout','s_commentary','isl_roster','s_roster')">Load ISL test match + roster</span>
          <span class="chip" onclick="loadSample('real_isl','s_commentary')">Load REAL ISL Cup Final</span>
        </div>
        <label style="margin-top:10px;">…or upload transcript (.txt)</label>
        <input type="file" id="s_commentary_file" accept=".txt"/>
        <label style="margin-top:10px;">…or transcribe a video/audio into this box</label>
        <input type="file" accept="video/*,audio/*" onchange="transcribeInto(this,'s_commentary')"/>
      </div>
      <div class="card">
        <label>Scout for position</label>
        <select id="s_role">{% for code, name in roles.items() %}<option value="{{code}}">{{name}} ({{code}})</option>{% endfor %}</select>
        <label style="margin-top:12px;">Roster / watchlist (optional, one per line)</label>
        <textarea id="s_roster" class="small" placeholder="Gurpreet Sandhu&#10;Manvir Lakra"></textarea>
        <label style="margin-top:10px;">Player video footage (optional)</label>
        <input type="file" id="s_videos" accept="video/*" multiple/>
        <button class="go" id="s_go" onclick="runScout()">Build shortlist</button>
        <div id="s_err" class="err"></div>
      </div>
    </div>
    <div id="s_out" style="margin-top:18px;"></div>
  </div>

  <!-- IMPROVEMENT PLAN -->
  <div id="panel-plan" class="hidden">
    <div class="grid">
      <div class="card">
        <label>Commentary / transcript</label>
        <textarea id="d_commentary" placeholder="Paste commentary, or load a sample..."></textarea>
        <div class="chips">
          <span class="chip" onclick="loadMatch('ileague_commentary','d_commentary','ileague_roster','d_roster')">Load REAL I-League match + roster</span>
          <span class="chip" onclick="loadMatch('isl_scout','d_commentary','isl_roster','d_roster')">Load ISL test match + roster</span>
        </div>
        <label style="margin-top:10px;">…or transcribe a video/audio into this box</label>
        <input type="file" accept="video/*,audio/*" onchange="transcribeInto(this,'d_commentary')"/>
      </div>
      <div class="card">
        <label>Position context</label>
        <select id="d_role">{% for code, name in roles.items() %}<option value="{{code}}">{{name}} ({{code}})</option>{% endfor %}</select>
        <label style="margin-top:12px;">Roster (optional, one per line)</label>
        <textarea id="d_roster" class="small" placeholder="One player per line"></textarea>
        <button class="go" id="d_go" onclick="runPlan()">Generate development plans</button>
        <div id="d_err" class="err"></div>
      </div>
    </div>
    <div id="d_out" style="margin-top:18px;"></div>
  </div>

  <!-- DEVELOPMENT LAB -->
  <div id="panel-dev" class="hidden">
    <div class="grid">
      <div class="card">
        <label>Commentary / transcript</label>
        <textarea id="g_commentary" placeholder="Paste commentary, or load a sample..."></textarea>
        <div class="chips">
          <span class="chip" onclick="loadMatch('ileague_commentary','g_commentary','ileague_roster','g_roster')">Load REAL I-League match + roster</span>
          <span class="chip" onclick="loadMatch('isl_scout','g_commentary','isl_roster','g_roster')">Load ISL test match + roster</span>
        </div>
        <label style="margin-top:10px;">…or transcribe a video/audio into this box</label>
        <input type="file" accept="video/*,audio/*" onchange="transcribeInto(this,'g_commentary')"/>
      </div>
      <div class="card">
        <label>Roster (optional, one player per line)</label>
        <textarea id="g_roster" class="small" placeholder="One player per line"></textarea>
        <div class="hint">Finds each player's mistake-moments in the commentary, shows what went wrong and the better approach on a 2D pitch, estimates their ceiling, and builds team-strategy notes.</div>
        <button class="go" id="g_go" onclick="runDev()">Run development lab</button>
        <div id="g_err" class="err"></div>
      </div>
    </div>
    <div id="g_out" style="margin-top:18px;"></div>
  </div>

  <!-- TRANSCRIBE -->
  <div id="panel-tr" class="hidden">
    <div class="grid">
      <div class="card">
        <label>Match video or audio</label>
        <input type="file" id="t_media" accept="video/*,audio/*"/>
        <div class="hint">Cheaper than full video analysis: this transcribes the <b>commentary audio</b>
          (Whisper) into text you can run through any tab. Supports Hindi / regional languages.</div>
        <label style="margin-top:12px;">Language (optional, e.g. hi, bn, ta)</label>
        <input type="text" id="t_lang" placeholder="auto-detect"/>
        <button class="go" id="t_go" onclick="runTranscribe()">Transcribe → text</button>
        <div id="t_err" class="err"></div>
        <div class="hint" style="margin-top:8px;">Needs <code>openai-whisper</code> + ffmpeg installed on the server.</div>
      </div>
      <div class="card">
        <label>Transcript</label>
        <textarea id="t_out" placeholder="Transcript appears here…"></textarea>
        <div class="chips" id="t_send" style="display:none;">
          <span class="chip" onclick="sendTranscript('p_commentary','pen')">→ Penalty</span>
          <span class="chip" onclick="sendTranscript('s_commentary','scout')">→ Scouting</span>
          <span class="chip" onclick="sendTranscript('d_commentary','plan')">→ Improvement</span>
          <span class="chip" onclick="sendTranscript('g_commentary','dev')">→ Development</span>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
function showTab(t){
  document.body.classList.remove('bg-pitch');   // back to stadium view on every tab switch
  ['pen','scout','plan','dev','tr'].forEach(x=>{
    document.getElementById('tab-'+x).classList.toggle('active', x===t);
    document.getElementById('panel-'+x).classList.toggle('hidden', x!==t);
  });
}
async function loadSample(name, targetId){
  try{ const r=await fetch('/sample?name='+name); const d=await r.json();
    if(d.text!==undefined) document.getElementById(targetId).value=d.text; }catch(e){}
}
async function loadMatch(commName, commTarget, rosterName, rosterTarget){
  await loadSample(commName, commTarget);
  await loadSample(rosterName, rosterTarget);
}
/* ---- video/audio transcription ---- */
async function transcribeInto(input, targetId){
  if(!input.files || !input.files[0]) return;
  const ta=document.getElementById(targetId); const prev=ta.value; ta.value='Transcribing… (this can take a minute)';
  const fd=new FormData(); fd.append('media', input.files[0]);
  try{ const r=await fetch('/transcribe',{method:'POST',body:fd}); const d=await r.json();
    ta.value = r.ok ? d.text : (prev + '\n[Transcribe unavailable: '+(d.error||'failed')+']'); }
  catch(e){ ta.value=prev; alert('Transcribe failed: '+e); }
  finally{ input.value=''; }
}
async function runTranscribe(){
  const err=document.getElementById('t_err'); err.textContent='';
  const inp=document.getElementById('t_media');
  if(!inp.files || !inp.files[0]){ err.textContent='Choose a video or audio file.'; return; }
  const fd=new FormData(); fd.append('media', inp.files[0]);
  fd.append('language', document.getElementById('t_lang').value);
  const btn=document.getElementById('t_go'); btn.disabled=true; const old=btn.textContent; btn.textContent='Transcribing…';
  try{ const r=await fetch('/transcribe',{method:'POST',body:fd}); const d=await r.json();
    if(!r.ok){ err.textContent=d.error||'Error'; return; }
    document.getElementById('t_out').value=d.text;
    document.getElementById('t_send').style.display='flex';
    document.body.classList.add('bg-pitch');
  }catch(e){ err.textContent='Request failed: '+e; }
  finally{ btn.disabled=false; btn.textContent=old; }
}
function sendTranscript(targetId, tab){
  document.getElementById(targetId).value=document.getElementById('t_out').value;
  showTab(tab);
}
/* ---- cross-continent AI insight ---- */
window._scout=[]; window._dev=[];
async function fetchInsight(player, role, attributes, strengths, weaknesses, boxId){
  const box=document.getElementById(boxId);
  box.innerHTML='<div class="insightcard"><span style="color:var(--mut)">Analysing cross-continent coaching strategies…</span></div>';
  try{
    const r=await fetch('/insight',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({player,role,attributes,strengths,weaknesses})});
    const d=await r.json();
    let h='<div class="insightcard"><div class="imode">'+(d.mode||d.source||'')+'</div>';
    h+='<h4 style="margin:2px 0 6px;">Cross-continent development — '+player+'</h4>';
    if(d.llm_text){ h+='<div>'+d.llm_text.replace(/\n/g,'<br>')+'</div>'; }
    else {
      h+='<div style="margin-bottom:6px;">'+(d.summary||'')+'</div>';
      (d.focus_items||[]).forEach(it=>{ h+='<div class="ifocus"><b>'+it.attribute+'</b> → study '+(it.study||[]).join(', ')+': '+it.recommendation+'</div>'; });
    }
    if(d.cultures&&d.cultures.length){ h+='<div class="icult">'+d.cultures.map(c=>'<b>'+c.name+':</b> '+c.philosophy).join('<br>')+'</div>'; }
    h+='</div>'; box.innerHTML=h;
  }catch(e){ box.innerHTML='<div class="insightcard" style="color:var(--red)">Insight failed: '+e+'</div>'; }
}
function scoutInsight(i){ const p=window._scout[i]; if(!p) return;
  fetchInsight(p.player, p.best_role||'player', p.attributes||{}, p.strengths||[], p.weaknesses||[], 's_insight'); }
function devInsight(i){ const p=window._dev[i]; if(!p) return;
  fetchInsight(p.player, p.best_role||'player', p.attributes||{}, [], [], 'g_insight_'+i); }
function openModal(id){ document.getElementById('modal-'+id).classList.add('open');
  if(id==='opt'){ fetch('/llm_status').then(r=>r.json()).then(d=>{
    const el=document.getElementById('opt_llm'); if(el) el.textContent=d.llm; }).catch(()=>{}); } }
function closeModal(id){ document.getElementById('modal-'+id).classList.remove('open');
  if(id==='opt') document.getElementById('opt_applied').textContent='Settings applied ✓ — re-run a tab to compare.'; }
function rv(id){ document.getElementById(id+'_v').textContent=document.getElementById(id).value; }
function resetOpt(){
  const d={opt_sign:75,opt_monitor:62,opt_develop:48,opt_minmen:2,opt_weak:58,
    opt_w_history:35,opt_w_technique:20,opt_w_composure:20,opt_w_readiness:15,opt_w_experience:10,opt_rec:70,opt_backup:50};
  document.getElementById('opt_model').value='offline';
  for(const k in d){ const el=document.getElementById(k); if(el) el.value=d[k]; const v=document.getElementById(k+'_v'); if(v) v.textContent=d[k]; }
  document.getElementById('rmv').textContent=70; document.getElementById('bmv').textContent=50;
}
function scoutSettings(fd){
  fd.append('model', document.getElementById('opt_model').value);
  fd.append('sign_min', document.getElementById('opt_sign').value);
  fd.append('monitor_min', document.getElementById('opt_monitor').value);
  fd.append('develop_min', document.getElementById('opt_develop').value);
  fd.append('min_mentions', document.getElementById('opt_minmen').value);
  fd.append('weak_threshold', document.getElementById('opt_weak').value);
}
function penaltySettings(fd){
  ['history','technique','composure','readiness','experience'].forEach(k=>
    fd.append('w_'+k, document.getElementById('opt_w_'+k).value/100));
  fd.append('rec_min', document.getElementById('opt_rec').value);
  fd.append('backup_min', document.getElementById('opt_backup').value);
}
function videoBlock(report){
  if(!report||!report.length) return '';
  let h='<div class="vid"><b>Video processed:</b><br>';
  report.forEach(v=>{ h+=`• ${v.file} → ${v.player}: `+(v.processed?`${v.resolution}, ${v.duration_s}s, composure≈${v.video_composure ?? 'n/a'}`:(v.note||'skipped'))+'<br>'; });
  return h+'</div>';
}
async function post(url, fd, btn, label, err){
  btn.disabled=true; const old=btn.textContent; btn.textContent='Working…';
  try{ const r=await fetch(url,{method:'POST',body:fd}); const d=await r.json();
    if(!r.ok){ err.textContent=d.error||'Error'; return null; }
    document.body.classList.add('bg-pitch'); return d; }
  catch(e){ err.textContent='Request failed: '+e; return null; }
  finally{ btn.disabled=false; btn.textContent=old; }
}

function outcomeBadge(o){
  if(o==='scored') return ' <span class="ob ok">✅ scored</span>';
  if(o==='missed') return ' <span class="ob no">❌ missed</span>';
  return '';
}
async function runPenalty(){
  const err=document.getElementById('p_err'); err.textContent=''; document.getElementById('p_out').innerHTML='';
  const fd=new FormData();
  fd.append('commentary', document.getElementById('p_commentary').value);
  const cf=document.getElementById('p_commentary_file').files[0]; if(cf) fd.append('commentary_file', cf);
  for(const v of document.getElementById('p_videos').files) fd.append('videos', v);
  penaltySettings(fd);
  const d=await post('/analyze', fd, document.getElementById('p_go'), '', err); if(!d) return;
  let h=''; if(d.recommended_order&&d.recommended_order.length)
    h+='<div class="order"><b>Suggested taker order:</b> '+d.recommended_order.map((n,i)=>(i+1)+'. '+n).join('   ')+'</div>';
  if(d.filtered_out&&d.filtered_out.length)
    h+='<div class="vid" style="margin-bottom:10px;">Filtered out (not recognised as real footballers): '+d.filtered_out.join(', ')+'</div>';
  h+='<div class="card"><table><thead><tr><th>#</th><th>Player</th><th>Suitability</th><th>Verdict</th><th>Pen record</th><th>Mental</th><th>Actual</th></tr></thead><tbody>';
  d.results.forEach((p,i)=>{ h+=`<tr><td>${i+1}</td><td><span class="pname" data-player="${p.player}">${p.player}</span> ${p.video_used?'<span class="vtag">●vid</span>':''}${p.known?'':' <span class="vtag" style="color:var(--mut)">new</span>'}</td>
    <td><div style="display:flex;align-items:center;gap:8px;"><div class="bar"><span style="width:${p.suitability}%"></span></div><b>${p.suitability}</b></div></td>
    <td><span class="pill ${p.category}">${p.category}</span></td><td>${p.pen_record}</td><td>${p.mental_state}</td><td>${outcomeBadge(p.outcome)||'<span style="color:var(--mut)">—</span>'}</td></tr>`; });
  h+='</tbody></table>'+videoBlock(d.video_report)+'</div>';
  // accuracy evaluation vs actual outcomes from the commentary
  const ev=d.evaluation;
  if(ev && ev.n_with_outcome>0){
    h+='<div class="teambox" style="margin-top:14px;">';
    if(ev.accuracy!==null){
      h+='<b>Output accuracy: '+ev.accuracy+'%</b> — the app ranked the actual penalty scorers above the players who missed in '+ev.correct+' of '+ev.pairs+' comparisons.<br>';
    } else {
      h+='<b>Outcome check.</b><br>';
    }
    if(ev.scorers.length) h+='✅ Scored (from commentary): '+ev.scorers.join(', ')+'<br>';
    if(ev.missers.length) h+='❌ Missed (from commentary): '+ev.missers.join(', ');
    h+='</div>';
  } else {
    h+='<div class="vid" style="margin-top:10px;">No penalty scored/missed outcomes were found in the commentary to evaluate accuracy against. Add lines like "X scored the penalty" / "Y\'s penalty was saved" to enable scoring.</div>';
  }
  document.getElementById('p_out').innerHTML=h;
}

async function runScout(){
  const err=document.getElementById('s_err'); err.textContent=''; document.getElementById('s_out').innerHTML='';
  const fd=new FormData();
  fd.append('commentary', document.getElementById('s_commentary').value);
  fd.append('role', document.getElementById('s_role').value);
  fd.append('roster', document.getElementById('s_roster').value);
  const cf=document.getElementById('s_commentary_file').files[0]; if(cf) fd.append('commentary_file', cf);
  for(const v of document.getElementById('s_videos').files) fd.append('videos', v);
  scoutSettings(fd);
  const d=await post('/scout', fd, document.getElementById('s_go'), '', err); if(!d) return;
  let h='<div class="order"><b>Shortlist for '+d.role_name+'.</b> ';
  h+= d.signings.length? ('Recommended to sign: '+d.signings.join(', ')+'. ') : 'No outright signings — development options below. ';
  if(d.prospects.length) h+='Prospects: '+d.prospects.join(', ')+'.';
  h+=' <span style="color:var(--mut);font-size:12px;">Model: '+(d.model_used||'offline')+'</span></div>';
  if(d.filtered_out&&d.filtered_out.length)
    h+='<div class="vid" style="margin-bottom:10px;">Removed (not a player name): '+d.filtered_out.join(', ')+'</div>';
  window._scout=d.shortlist;
  h+='<div class="card"><table><thead><tr><th>#</th><th>Player</th><th>Role fit</th><th>Verdict</th><th>Strengths</th><th>To improve</th><th>Develop</th></tr></thead><tbody>';
  d.shortlist.forEach((p,i)=>{ h+=`<tr><td>${i+1}</td><td><span class="pname" data-player="${p.player}">${p.player}</span> ${p.potential_flag?'<span class="star">★</span>':''}</td>
    <td><div style="display:flex;align-items:center;gap:8px;"><div class="bar"><span style="width:${p.role_rating}%"></span></div><b>${p.role_rating}</b></div></td>
    <td><span class="pill ${p.verdict}">${p.verdict}</span></td><td>${(p.strengths||[]).join(', ')||'-'}</td><td>${(p.weaknesses||[]).join(', ')||'-'}</td>
    <td><span class="simbtn good" onclick="scoutInsight(${i})">🌍 AI insight</span></td></tr>`; });
  h+='</tbody></table>'+videoBlock(d.video_report)+'</div><div id="s_insight" style="margin-top:12px;"></div>';
  document.getElementById('s_out').innerHTML=h;
}

async function runPlan(){
  const err=document.getElementById('d_err'); err.textContent=''; document.getElementById('d_out').innerHTML='';
  const fd=new FormData();
  fd.append('commentary', document.getElementById('d_commentary').value);
  fd.append('role', document.getElementById('d_role').value);
  fd.append('roster', document.getElementById('d_roster').value);
  scoutSettings(fd);
  const d=await post('/plan', fd, document.getElementById('d_go'), '', err); if(!d) return;
  let h='<div class="order"><b>Development plans — '+d.role_name+' context.</b></div><div class="card">';
  d.plans.forEach(pl=>{
    h+=`<div class="plan"><h4>${pl.player} <span class="pill ${pl.verdict}">${pl.verdict}</span> <span style="color:var(--mut);font-size:13px;">fit ${pl.role_rating}</span> ${pl.potential_flag?'<span class="star">★ prospect</span>':''}</h4>`;
    h+=`<div style="font-size:13px;color:var(--mut);">${pl.summary}</div>`;
    if(pl.focus_areas&&pl.focus_areas.length){ pl.focus_areas.forEach(f=>{
      h+=`<div class="focus"><b>${f.attribute}</b> (${f.score})${f.role_relevant?' — key for role':''}<ul>`+f.drills.map(x=>`<li>${x}</li>`).join('')+'</ul></div>'; });
    } else { h+='<div class="focus" style="color:var(--green);">No major weaknesses detected — maintain and add game time.</div>'; }
    h+='</div>';
  });
  h+='</div>'; document.getElementById('d_out').innerHTML=h;
}

/* ---------- Development Lab: 2D pitch simulation ---------- */
const SCEN = {
  shot:     {actual:{path:[[72,34],[103,15]], color:'#ff5d5d'},
             better:{path:[[72,34],[88,46],[103,40]], color:'#28d17c', mate:[88,46]}},
  position: {actual:{path:[[40,28],[58,16]], color:'#ff5d5d', runner:[[26,40],[6,33]]},
             better:{path:[[40,28],[30,36]], color:'#28d17c', runner:[[26,40],[20,40]]}},
  control:  {actual:{path:[[52,34],[61,21]], color:'#ff5d5d'},
             better:{path:[[52,34],[45,40],[33,44]], color:'#28d17c', mate:[33,44]}},
  pass:     {actual:{path:[[42,34],[74,12]], color:'#ff5d5d'},
             better:{path:[[42,34],[55,42]], color:'#28d17c', mate:[55,42]}},
  duel:     {actual:{path:[[50,34],[64,30]], color:'#ff5d5d'},
             better:{path:[[50,34],[45,38]], color:'#28d17c'}},
};
function pitchSVG(){
  return `<svg class="simpitch" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
    <rect x="0" y="0" width="105" height="68" fill="#1c7d40"/>
    <rect x="2" y="2" width="101" height="64" fill="none" stroke="rgba(255,255,255,.5)" stroke-width="0.5"/>
    <line x1="53.5" y1="2" x2="53.5" y2="66" stroke="rgba(255,255,255,.45)" stroke-width="0.4"/>
    <circle cx="53.5" cy="34" r="8" fill="none" stroke="rgba(255,255,255,.45)" stroke-width="0.4"/>
    <rect x="87" y="16" width="16" height="36" fill="none" stroke="rgba(255,255,255,.45)" stroke-width="0.4"/>
    <rect x="97" y="26" width="6" height="16" fill="none" stroke="rgba(255,255,255,.45)" stroke-width="0.4"/>
    <rect x="103" y="29.5" width="2" height="9" fill="rgba(255,255,255,.9)"/>
    <g class="simlayer"></g></svg>`;
}
function pts(a){ return a.map(p=>p.join(',')).join(' '); }
function playSim(svg, scenario, mode){
  const cfg=SCEN[scenario]; if(!cfg) return; const part=cfg[mode];
  const NS='http://www.w3.org/2000/svg';
  const layer=svg.querySelector('.simlayer'); layer.innerHTML='';
  if(part.runner){ const r=document.createElementNS(NS,'polyline');
    r.setAttribute('points',pts(part.runner)); r.setAttribute('fill','none');
    r.setAttribute('stroke','#ffd166'); r.setAttribute('stroke-width','0.7');
    r.setAttribute('stroke-dasharray','2 2'); r.setAttribute('opacity','0.6'); layer.appendChild(r); }
  if(part.mate){ const c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',part.mate[0]); c.setAttribute('cy',part.mate[1]); c.setAttribute('r','1.9');
    c.setAttribute('fill','#9fe6c0'); layer.appendChild(c); }
  const poly=document.createElementNS(NS,'polyline'); poly.setAttribute('points',pts(part.path));
  poly.setAttribute('fill','none'); poly.setAttribute('stroke',part.color); poly.setAttribute('stroke-width','0.8');
  poly.setAttribute('stroke-dasharray','2 1.5'); poly.setAttribute('opacity','0.65'); layer.appendChild(poly);
  const ball=document.createElementNS(NS,'circle'); ball.setAttribute('r','1.7'); ball.setAttribute('fill','#fff');
  ball.setAttribute('stroke','#111'); ball.setAttribute('stroke-width','0.3'); layer.appendChild(ball);
  const P=part.path; let seg=0,t=0; const sp=0.045;
  ball.setAttribute('cx',P[0][0]); ball.setAttribute('cy',P[0][1]);
  function frame(){
    if(seg>=P.length-1){ ball.setAttribute('cx',P[P.length-1][0]); ball.setAttribute('cy',P[P.length-1][1]); return; }
    const a=P[seg],b=P[seg+1];
    ball.setAttribute('cx',a[0]+(b[0]-a[0])*t); ball.setAttribute('cy',a[1]+(b[1]-a[1])*t);
    t+=sp; if(t>=1){ t=0; seg++; } requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
function playFrom(btn, scenario, mode){
  const wrap=btn.closest('.simwrap'); const svg=wrap.querySelector('svg.simpitch');
  playSim(svg, scenario, mode);
  const lbl=wrap.querySelector('.simlabel');
  lbl.textContent = mode==='actual' ? 'Replaying what happened…' : 'The recommended approach.';
  lbl.style.color = mode==='actual' ? 'var(--red)' : 'var(--green)';
}
async function runDev(){
  const err=document.getElementById('g_err'); err.textContent=''; document.getElementById('g_out').innerHTML='';
  const fd=new FormData();
  fd.append('commentary', document.getElementById('g_commentary').value);
  fd.append('roster', document.getElementById('g_roster').value);
  scoutSettings(fd);
  const d=await post('/develop', fd, document.getElementById('g_go'), '', err); if(!d) return;
  let h=''; const ts=d.team_strategy;
  h+='<div class="teambox"><b>Team strategy.</b> '+ts.headline+'<ul>';
  ts.notes.forEach(n=> h+=`<li><b>${n.area}</b>${n.count?' ×'+n.count:''}: ${n.recommendation}</li>`);
  h+='</ul></div>';
  window._dev=d.players;
  d.players.forEach((p,idx)=>{
    const cg=p.ceiling, bandkey=cg.band.split(' ')[0];
    h+=`<div class="devcard"><div class="devhead">
      <h3><span class="pname" data-player="${p.player}">${p.player}</span> ${p.potential_flag?'<span class="star">★</span>':''}</h3>
      <span class="pill ${p.verdict}">${p.verdict}</span>
      <span class="ceil">${p.role_name} · now ${cg.current} → ceiling ${cg.ceiling}</span>
      <span class="band ${bandkey}">${cg.band}</span></div>
      <div class="growbar"><div class="cur" style="width:${cg.current}%"></div><div class="cap" style="left:${cg.ceiling}%"></div></div>
      <div style="font-size:13px;color:#cdd8ee;margin-top:8px;">${p.unlock}</div>
      <div style="margin-top:8px;"><span class="simbtn good" onclick="devInsight(${idx})">🌍 Cross-continent AI insight</span></div>
      <div id="g_insight_${idx}"></div>`;
    if(!p.mistakes.length){
      if(p.focus){
        h+=`<div class="mistake">
          <div class="simwrap">${pitchSVG()}
            <div class="simbtns"><span class="simbtn good" onclick="playFrom(this,'${p.focus.scenario}','better')">▶ Recommended approach</span></div>
            <div class="simlabel"></div>
          </div>
          <div class="mtext">
            <div class="row"><b class="g">Development focus: ${p.focus.attribute} (${p.focus.score})</b></div>
            <div class="row">${p.focus.better_approach}</div>
            <div class="row" style="color:var(--mut);"><b>Drill:</b> ${p.focus.drill}</div>
          </div></div>`;
      } else {
        h+=`<div style="font-size:13px;color:var(--green);margin-top:8px;">No clear development data in this commentary.</div>`;
      }
    }
    p.mistakes.forEach((m,mi)=>{
      h+=`<div class="mistake">
        <div class="simwrap">${pitchSVG()}
          <div class="simbtns">
            <span class="simbtn bad" onclick="playFrom(this,'${m.scenario}','actual')">▶ What happened</span>
            <span class="simbtn good" onclick="playFrom(this,'${m.scenario}','better')">▶ Better approach</span>
          </div>
          <div class="simlabel"></div>
        </div>
        <div class="mtext">
          <div class="row"><b class="w">What went wrong${m.minute?(' (min '+m.minute+')'):''}:</b> ${m.what_went_wrong}</div>
          <div class="row"><b>Why:</b> ${m.why}</div>
          <div class="row"><b class="g">Better approach:</b> ${m.better_approach}</div>
          <div class="row" style="color:var(--mut);"><b>Drill:</b> ${m.drill}</div>
        </div></div>`;
    });
    h+='</div>';
  });
  document.getElementById('g_out').innerHTML=h;
}
/* ---- LLM player-existence check: flags names that aren't real footballers ---- */
async function annotatePlayers(containerId){
  const c=document.getElementById(containerId); if(!c) return;
  const els=[...c.querySelectorAll('.pname[data-player]')];
  const names=[...new Set(els.map(e=>e.getAttribute('data-player')))];
  if(!names.length) return;
  try{
    const r=await fetch('/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({players:names})});
    const d=await r.json(); const res=d.results||{};
    els.forEach(e=>{ const v=res[e.getAttribute('data-player')];
      if(v==='unknown'){ e.insertAdjacentHTML('afterend',' <span class="notfound" title="Not found as a real footballer by the LLM check">⚠ not found</span>'); }
    });
  }catch(e){}
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("GenAI Football running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
