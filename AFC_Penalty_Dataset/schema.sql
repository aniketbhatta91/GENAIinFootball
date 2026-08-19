CREATE TABLE matches(
  match_id      TEXT PRIMARY KEY,
  competition   TEXT, edition_year INTEGER, stage TEXT, match_date TEXT, venue TEXT,
  home_team     TEXT, away_team TEXT,
  score_reg_et  TEXT, extra_time TEXT,
  shootout_score TEXT, winner TEXT,
  shootout_kicks INTEGER, commentary_type TEXT,
  transcript_file TEXT, commentary_text TEXT
);

CREATE TABLE shootout_kicks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id TEXT REFERENCES matches(match_id),
  kick_no  INTEGER, team TEXT, taker TEXT,
  result   TEXT,            -- scored | missed | saved
  scored   INTEGER          -- 1/0
);

CREATE TABLE sqlite_sequence(name,seq);

CREATE TABLE goals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id TEXT REFERENCES matches(match_id),
  minute   TEXT, team TEXT, scorer TEXT, description TEXT
);

CREATE INDEX ix_kicks_match ON shootout_kicks(match_id);

CREATE INDEX ix_goals_match ON goals(match_id);

CREATE TABLE worldcup_sources(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  year INTEGER, tournament TEXT, match_detail TEXT, match_date TEXT,
  owner TEXT, commentary_url TEXT, espn_game_id TEXT,
  commentary_collected TEXT DEFAULT 'No', commentary_text TEXT);