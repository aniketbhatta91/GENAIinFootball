# GenAI in Football ⚽

AI-assisted football analysis for grassroots and Indian football (ISL / I-League).
The app reads match **commentary / transcripts** (and optionally **video footage**)
and turns them into decisions a coach or scout can use:

1. **Penalty Selector** — ranks the best penalty-shootout takers from match commentary + player clips.
2. **Scouting** — builds a role-based shortlist (ST / W / CM / CB / FB / GK) with SIGN / MONITOR / DEVELOP / PASS verdicts.
3. **Improvement Plan** — turns each player's weaknesses into targeted training drills.
4. **Validation** — backtests the engine against real ISL "Emerging Player" winners and their actual India call-ups.

All wrapped in a football-themed web app with an **Architecture** panel and a live **Optimization** panel for tuning models, weights, and thresholds.

## Quick start

```bash
pip install -r requirements.txt
python penalty_app_server.py
# open http://127.0.0.1:5000
```

Optional extras:
- **BERT/RoBERTa sentiment:** `pip install transformers torch` (otherwise the app uses a fast offline rule-based analyzer).
- **Video footage analysis:** `opencv-python` (in requirements).
- **Transcription of match audio/video:** `pip install openai-whisper` + ffmpeg.

## Architecture

```
Audio/Video ─▶ Transcription (Whisper, multilingual: Hindi/Bengali/Tamil/English)
        │
Commentary / transcript ─▶ NLP layer
        ├─ Player detection (regex + roster matching)
        ├─ Sentiment (offline lexicon  OR  RoBERTa/BERT)
        └─ Attribute extraction (11 football attributes)
        │
        ├─▶ Penalty Fusion Engine ─▶ ranked takers
        ├─▶ Scouting Engine ─▶ role-fit shortlist + verdict
        ├─▶ Improvement Plan ─▶ training drills per weakness
        └─▶ Validation / Backtest ─▶ metrics vs real outcomes

Optional video signal: OpenCV motion analysis ─▶ composure score
```

**Tech:** Python · Flask · OpenCV · (optional) HuggingFace Transformers + PyTorch · Whisper · single-page HTML/JS frontend.

## Validation result

Backtested on 11 real ISL Emerging Player winners (2014–2024-25), reading only performance descriptions:

| Metric | Result |
|---|---|
| Role-classification accuracy | 100% (11/11) |
| Precision@5 (top picks are internationals) | 100% |
| Internationals flagged (not "PASS") | 100% |
| Rating gap (internationals vs not) | +33 pts |
| Spearman (rating vs India call-up) | +0.67 |

See `validation/validation_report.md` for methodology, data, and sources.

## Key files

| File | Purpose |
|---|---|
| `penalty_app_server.py` | Flask web app (all four tools + architecture/optimization panels) |
| `penalty_fusion_engine.py` | Penalty-taker suitability scoring |
| `scouting_engine.py` | Attribute extraction, role-fit, verdicts, improvement plans |
| `penalty_predictor_backend_OFFLINE.py` | Offline rule-based sentiment analyzer |
| `transcription.py` | Whisper speech-to-text layer |
| `backtest.py` | Validation harness + metrics |
| `scout_shortlist.py` | CLI scouting runner |
| `validation/` | Ground-truth data + validation report |
| `*_commentary.txt`, `*_roster.txt` | Sample and real data for testing |

## CLI usage

```bash
# Scouting shortlist for a position
python scout_shortlist.py --commentary isl_test_commentary.txt --roster isl_test_roster.txt --role W

# Run the validation backtest
python backtest.py
```

## Notes & limitations

- Scoring is deliberately **transparent** (readable, tunable weights) rather than a black box.
- Validation is a modest sample (n=11) on a strong cohort — encouraging, not yet statistically robust.
- Video composure is a lightweight OpenCV motion heuristic, a placeholder for a future MediaPipe/YOLO pose pipeline.
- Some sample commentary uses fictional players for demonstration; real ISL data is factual match events.

---
*Research/educational project. Not affiliated with the AIFF, ISL, or any club.*
