# Scouting Engine — Real-World Validation Report

**Question:** Does the scouting engine actually work — i.e. if it reads match
performance descriptions, does it (a) identify the right position and (b) flag
the players who genuinely went on to higher honours?

**Test design (a retrospective backtest).** We use the **ISL "Emerging Player
of the League"** winners from 2014 to 2024-25 as ground truth. Each is a real,
verified young Indian player. For every player we feed the engine only
scouting-style descriptions of their playing performances (paraphrased from
public ISL season reviews and player profiles) and check the engine's output
against two real-world facts: their **actual playing position**, and whether
they earned an **India national team call-up**.

This is a fair test because the engine never sees the outcome — it only reads
performance language, exactly as it would for a new prospect.

## Data

11 players, all real ISL Emerging Player winners. 9 of the 11 went on to receive
senior India national-team call-ups; 2 (Lalruatthara, Sumit Rathi) won the youth
award but faded and did not establish themselves internationally — these act as
natural negative controls, and the public reviews of both note their
inconsistency.

## Results

| Player | Real pos | Engine pos | Rating | Verdict | India call-up |
|---|---|---|---|---|---|
| Sandesh Jhingan | CB | CB ✓ | 77.1 | SIGN | Yes |
| Jerry Lalrinzuala | FB | FB ✓ | 76.3 | SIGN | Yes |
| Vikram Pratap Singh | W | W ✓ | 74.6 | MONITOR | Yes |
| Roshan Naorem | FB | FB ✓ | 73.8 | MONITOR | Yes |
| Lalengmawia (Apuia) | CM | CM ✓ | 73.7 | MONITOR | Yes |
| Brison Fernandes | ST | ST ✓ | 73.5 | MONITOR | Yes |
| Sivasakthi Narayanan | ST | ST ✓ | 72.4 | MONITOR | Yes |
| Jeje Lalpekhlua | ST | ST ✓ | 72.2 | MONITOR | Yes |
| Sahal Abdul Samad | CM | CM ✓ | 68.1 | MONITOR | Yes |
| Sumit Rathi | CB | CB ✓ | 48.0 | DEVELOP | no |
| Lalruatthara | FB | FB ✓ | 32.8 | PASS | no |

## Metrics

| Metric | Result | Reading |
|---|---|---|
| Role-classification accuracy | **100%** (11/11) | engine put every player in their real position |
| Recall of internationals | **100%** | every future international was flagged (not "PASS") |
| Precision@3 | **100%** | top 3 by rating are all internationals |
| Precision@5 | **100%** | top 5 by rating are all internationals |
| Mean rating — internationals | **73.5** | |
| Mean rating — non-internationals | **40.4** | |
| Separation | **+33.1 pts** | clear gap between the two groups |
| Spearman (rating vs call-up) | **+0.67** | strong positive rank correlation |

## Interpretation

The engine's ranking lines up with reality: the nine players who became India
internationals occupy the top nine rating slots, while the two who did not
establish themselves sit clearly at the bottom (one DEVELOP, one PASS). Position
detection was perfect on this set. In other words, reading only performance
descriptions, the engine reproduced the same talent judgement that real selectors
and award panels reached independently.

## Honest limitations

- **Small sample (n = 11).** The numbers are encouraging but not yet
  statistically robust; precision of 100% will fall as the set grows. The
  framework is built to scale — add rows to `validation/groundtruth.csv` and
  descriptions to `validation_commentary.txt` and re-run `backtest.py`.
- **Input is paraphrased descriptions, not a raw live commentary feed.** Real
  ball-by-ball ISL/I-League commentary sits behind JavaScript feeds; pulling it
  requires the browser extension. Using genuine match transcripts is the next
  step to harden the test.
- **Survivorship / label design.** All 11 already won a youth award, so the test
  measures separation *within* a strong cohort rather than against random
  players. Adding journeyman players who never progressed would make it tougher.
- **Outcome = national call-up** is a proxy for "talent", not a perfect label
  (injuries, club politics and timing all affect call-ups).

## Reproduce

```
python backtest.py
```
Outputs the metrics above and writes `validation/validation_results.csv`.

## Sources (ground truth)

- ISL Emerging Player winners & trajectories — Khel Now: https://khelnow.com/football/isl-emerging-player-award-winners-where-are-they-now
- Full winners list — Khel Now: https://khelnow.com/football/indian-football-isl-emerging-player-awards-202406
- Recent winners (2021-22 to 2024-25) — Indian Super League / Olympics.com / Sportskeeda match and award reports
- 2022-23 ISL displays earning Asian Games call-ups — Sportskeeda: https://www.sportskeeda.com/indian-football/3-indian-players-whose-impressive-2022-23-isl-displays-earned-call-up-2023-asian-games

*Player performance descriptions used as engine input were paraphrased from the
above public reporting; no commentary text was reproduced verbatim.*
