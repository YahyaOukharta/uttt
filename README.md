# Ultimate Tic-Tac-Toe bot

A bot for the [CodinGame Ultimate Tic-Tac-Toe arena](https://www.codingame.com/multiplayer/bot-programming/tic-tac-toe),
in one C++ file: bitboard engine, Monte Carlo tree search with a solver.

Work in progress. Starting point: rank 195 in Legend (score 20.23) with an older MCTS bot,
kept as `league/src/legacy.cpp` as the baseline every new version has to beat.

## Quick start

```sh
make            # build/bot (local build) and build/referee
make perft      # engine vs referee move counts: 81 720 6336 55080 473256 4020960 33782544 281067408
python3 tools/league.py add v0.1 bot.cpp
python3 tools/league.py match v0.1 legacy -g 400 -n 20000     # fixed iterations per move
python3 tools/kaggle_match.py push t1 --games 300 --ms 100 bot.cpp league/src/legacy.cpp   # real time, on Kaggle
```

## Layout

| Path | What |
|---|---|
| `bot.cpp` | Engine, search, protocol; local tools under `#ifdef LOCAL` (perft, bench, fixed iterations, openings) |
| `tools/referee.cpp` | Native referee, written independently of the engine (cross-checked by perft) |
| `tools/league.py` | Frozen versions, matches with Elo (draws count half), ratings |
| `tools/kaggle_match.py` | Real-time gate matches on Kaggle CPUs |
| `tools/cg.py` | CodinGame API: test games, submit (+ IDE mirror), progress, leaderboard |
| `docs/research.md` | What the strong bots do: forum, post-mortems, papers |
| `docs/statement.md` | The rules |
