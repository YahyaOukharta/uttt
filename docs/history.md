# History

"Local" = gate match against the previous version: fixed iterations per move on the Mac,
or real time (100 ms) on Kaggle CPUs. Self-play Elo overstates arena gains.

## Versions

| Version | Change | Local | Arena | Rank |
|---|---|---|---|---|
| legacy | 2025 MCTS (UCT c = 1.2, partial solver, tree reuse) | | 20.23 | 195 / 490 Legend |
| v0.1 | from scratch: bitboards, UCT c = 0.7, MCTS-Solver (win / loss / draw), tree reuse | = legacy at 100 ms (+3 ± 34) | | |
| v0.2 | c = 0.55, rollouts take a game-winning move | vs legacy at 100 ms: +166 =56 −78, +105 ± 37 | (not submitted) | |
| v0.3 | + expansion keeps only a game-winning child, FPU 0.8 | | submission 41716558 | |

## Measurements

| Experiment | Result (Elo, 400 games at 20k iterations unless noted) |
|---|---|
| UCT c 0.4 / 0.55 / 0.9 / 1.1 vs 0.7 | +35 / **+51** / −67 / −153 |
| Rollout: take a game-winning move | **+135** |
| + avoid sending the opponent to a board where it wins the game | +118 (no better than win-only) |
| + take a small-board win | +26 (worse than win-only) |
| + block a small-board win | −28 vs random |
| game win + board win + board block | +99 |
| all four | +63 |
| Expansion keeps only a game-winning child (node proven) | +33 ± 29 |
| FPU 0.4 / 0.6 vs "visit every child first" | +15 / +29 (noise) |
| Pilot net: 128 hidden, 443k positions (4k-iteration rollout self-play), expand-all search | 9 −84 =7 vs v0.3 at 100 ms (net too weak; 75k vs 60k iterations per turn) |
