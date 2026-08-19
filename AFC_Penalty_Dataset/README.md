# AFC Penalty-Shootout Commentary Dataset

A growing database of AFC competition matches decided by a penalty shootout, from 2002 onward, stored as detailed 0–120 minute commentary transcripts with full penalty-shootout details. Built for the GenAI Football research project (penalty-taker selection and sentiment-strategy comparison).

## Important — how the commentary was produced

Genuine ball-by-ball commentary is only archived online for recent editions. For older matches it does not exist to collect. So every transcript here is a **reconstructed, clearly-labelled** file:

- **Real / verified in every file:** the teams, competition, stage, date, venue, final score (incl. extra time), and the **penalty shootout kick-by-kick outcome** (who scored, missed, or was saved).
- **Reconstructed:** the minute-by-minute open-play narrative is AI-generated realistic filler so the app has commentary text to read. It is illustrative, not an authentic record.
- **Goal scorers/minutes** are included only where reliably known (e.g. the 2023 edition); otherwise goals are marked generic.

Each file carries a header stating this. Do not cite the open-play narrative as fact — only the header facts and the shootout are real.

## Contents

### Phase 1 — AFC Asian Cup (complete): `asian_cup/`
16 shootout matches, 2004–2023: 2004 (Bahrain–Uzbekistan, Japan–Jordan, China–Iran); 2007 (Japan–Australia, South Korea–Iran, Iraq–South Korea, South Korea–Japan 3rd place); 2011 (Japan–South Korea); 2015 (Iraq–Iran, UAE–Japan); 2019 (Vietnam–Jordan, Australia–Uzbekistan); 2023 (Tajikistan–UAE, South Korea–Saudi Arabia, Iran–Syria, Qatar–Uzbekistan). See `index_asian_cup.csv`.

### Planned phases (scope: all AFC competitions)
- **Phase 2 — AFC Champions League** knockout shootouts since 2002/03.
- **Phase 3 — AFC Cup** knockout shootouts.
- **Phase 4 — AFC World Cup qualifiers / playoffs** decided on penalties.

These are large sets; they will be added incrementally, each with the same verified-facts + reconstructed-narrative approach and its own index file.

## File format

Plain `.txt`, UTF-8. Structure: header block (verified facts + reconstruction notice) → `[minute]` commentary lines 0–120 → `PENALTY SHOOTOUT` section listing each kick as `Penalty N: <taker> (<Team>) scores/MISSES/penalty SAVED  (running score)`. This matches what the GenAI Football penalty selector reads.

## Using it in the app

Paste a file into the Penalty tab, or add it to the match dropdown. The penalty scorer reads the shootout lines to rank takers; the LLM layer verifies real players (generic placeholder kick labels for matches where individual takers aren't documented will be filtered out).

## Sources (for verified facts)

- List of AFC Asian Cup penalty shoot-outs — EverybodyWiki
- 2023 AFC Asian Cup knockout stage — Wikipedia
- Individual edition pages (2004–2019 AFC Asian Cup) — Wikipedia; ESPN match pages
