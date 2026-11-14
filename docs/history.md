# History

"Local" = gate match against the previous version: fixed iterations per move on the Mac,
or real time (100 ms) on Kaggle CPUs. Self-play Elo overstates arena gains.

## Versions

| Version | Change | Local | Arena | Rank |
|---|---|---|---|---|
| legacy | 2025 MCTS (UCT c = 1.2, partial solver, tree reuse) | | 20.23 | 195 / 490 Legend |
| v0.1 | from scratch: bitboards, UCT c = 0.7, MCTS-Solver (win / loss / draw), tree reuse | = legacy at 100 ms (+3 ± 34) | | |
| v0.2 | c = 0.55, rollouts take a game-winning move | vs legacy at 100 ms: +166 =56 −78, +105 ± 37 | (not submitted) | |
| v0.3 | + expansion keeps only a game-winning child, FPU 0.8 | vs legacy at 100 ms: +197 =54 −49, +187 ± 40 | 22.6 at 84% (stopped) | ~92 |
| v1.0 | value net n1a (189→256×2, 10 buckets, 5.2M positions of 4k-iteration rollout self-play), expand-all search, c = 0.3 | vs v0.3 at 100 ms: +194 =46 −60, +166 ± 40 | 23.82 at 62% (replaced) | 63 |
| v1.1 | net n1b (256 hidden, 41M positions: 39M rollout self-play + 2M n1a self-play, 10 epochs on an A10G), c = 0.2, fused child evaluation (+44% iterations on CG), spare pool prefaulted (fixes rare second-turn timeouts) | vs v1.0 at 100 ms: +426 =135 −39, +266 ± 29 | submission 41716853 | |

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
| n1a search c 0.3 / 0.55 vs v0.3 (100 ms, Modal) | **+166** / +62 |
| Modal gate tool check: v0.3 vs legacy (100 ms) | +155 ± 39 (Kaggle: +187 ± 40) |
| n1a search c 0.2 / 0.3 (600 games each vs v0.3) | +172 / +147; c 0.12: +147 (300) |
| n1a, one net evaluation per leaf instead of scoring all children | +15 / +24 vs v0.3 (expand-all: +166) |
| Fused child evaluation vs accumulator copy (same net, 600 games) | +6 ± 16; CG iterations 308k → 443k first turn, 40k → 58k per turn |
| n1c (512 hidden) vs v1.0 | +259 ± 31 (n1b 256: +266 ± 29) |
| Pilot net: 128 hidden, 443k positions (4k-iteration rollout self-play), expand-all search | 9 −84 =7 vs v0.3 at 100 ms (net too weak; 75k vs 60k iterations per turn) |
