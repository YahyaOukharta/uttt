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
| v1.1 | net n1b (256 hidden, 41M positions: 39M rollout self-play + 2M n1a self-play, 10 epochs on an A10G), c = 0.2, fused child evaluation (+44% iterations on CG), spare pool prefaulted (fixes rare second-turn timeouts) | vs v1.0 at 100 ms: +426 =135 −39, +266 ± 29 | 27.89 at 70% (replaced) | 36 |
| v1.2 | net n1d (68M positions, 30 epochs) + PUCT selection: priors = softmax(first child values / 0.1), c = 1.0 | vs v1.1 at 100 ms: +186 =82 −32, +196 ± 36 | 31.13 (final) | 15 |
| v1.3 | net n1f (91M positions, λ 0.75), 4 virtual visits per first net value, compaction bounded to 2M nodes (fixes rare second-turn timeouts), first turn 920 ms | vs v1.2 at 100 ms: +200 =287 −113, +51 ± 20; 0 timeouts in 600 games | 31.43 (final) | 13 |
| v1.4 | net n3b: 640 hidden, 151M net self-play positions (v1.0–v1.3), λ 0.75, 20 epochs; accumulator cache; final move by value + 0.02·ln(visits); int8 output layer in the packed net | vs v1.3 at 100 ms: +252 =253 −95, +93 ± 21 (n3a 512: +80, n2c 512: +65) | submission 41717247 | |

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
| n1d (68M, 30 epochs) / n1e (λ 0.25) vs n1b (v1.1) | +25 ± 19 / −16 ± 19 |
| PUCT with value priors, n1b, vs UCT (v1.1): c 1.5 / 3 / 6 at T 0.1; c 3 at T 0.05 | **+144** / +64 / −50; +84 (control UCT: −6) |
| PUCT, n1d, vs v1.1: c 1.5 / 1.0 / 0.7 at T 0.1; c 1.0 at T 0.2 / 0.05; c 1.5 at T 0.2 | +156 / **+196** / +179; +132 / +171; +151 |
| n1f (λ 0.75) / n2a (net self-play only) / n2b (all 129M, λ 0.5) vs n1d, same search | +24 / +16 / +3 (± 19) |
| VN 2 / 4 (n1d) vs v1.2; LEAF 0.7 / 0.4 | +19 / **+47**; +13 / +2 |
| n1f with VN 4 / 8 / 12 / 8 + c 1.5 vs v1.2 | **+38** / +9 / +8 / +33 |
| Final move by value + FK·ln(visits), n1f, vs v1.3: FK 0 / 0.01 / 0.02 / 0.03 / 0.05 / 0.1 | +12 (cache only) / +59 / +51, +67 / +73 / +45 / +37 |
| Accumulator cache (x86, 512 hidden, 300k iterations) | 0.524 s → 0.477 s |
| Pilot net: 128 hidden, 443k positions (4k-iteration rollout self-play), expand-all search | 9 −84 =7 vs v0.3 at 100 ms (net too weak; 75k vs 60k iterations per turn) |
