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
| v1.4 | net n3b: 640 hidden, 151M net self-play positions (v1.0–v1.3), λ 0.75, 20 epochs; accumulator cache; final move by value + 0.02·ln(visits); int8 output layer in the packed net | vs v1.3 at 100 ms: +252 =253 −95, +93 ± 21 (n3a 512: +80, n2c 512: +65) | 31.44 (final) | 14 |
| v1.5 | net n4a: n3b fine-tuned 8 epochs on 139M unique positions (v1.2–v1.4 self-play); proven-draw contempt (draw worth 0.3 as first player, 0.7 as second); Huffman-coded weights (83.6k chars) | vs v1.4 at 100 ms: +204 =250 −146, +34 ± 21 (n4b from scratch on 186M: +33; no contempt: +25) | ~31.1 at 58% (replaced) | 15 |
| v1.6 | v1.5 + pondering (search continues during the opponent's turn until its move arrives on stdin; on CG +2.5k–9k iterations per opponent turn vs the default bot) + 11% faster iterations | vs v1.5 at 100 ms (2 cores per game): +261 =229 −110, +89 ± 22 (speedups alone: +24) | 31.41 (final) | 16 |
| v1.7 | net n7t: n4a widened with 18 small-board threat inputs, fine-tuned 6 epochs on the newest 137M unique positions; proven draws back to 0.5 (contempt lost arena games); pre-game ponder for the second player | n7t vs n4a, same code, no ponder: +202 =258 −140, +36 ± 21 (n6m without threats: +14) | 32.19 (final); vs the top 25: 51% (v1.6: 48%) | 11 |

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
| Proven-draw contempt (first / second player draw value), n3b, vs v1.4: 0.4 / 0.6, 0.3 / 0.7, 0.4 / 0.5, 0.5 / 0.6 | +19 / **+27** / +9 / +1 (± 20) |
| n4a with contempt 0.3 / 0.7, 0.2 / 0.8, none vs v1.4 | +34 / +35 / +25 |
| n4b (from scratch, 186M unique, 16 epochs) vs v1.4 / vs n4a | +33 / −9 |
| Trainer: 64-bit hash dedup, int16 inputs | load 3.8 → 1.0 s per million positions; 186M positions fit an A10G |
| Huffman-coded int8 weights | 6.73 bits per weight; 640 hidden 99.6k → 84.2k chars; 768 projected 96.4k |
| Search speed (x86, 300k first-turn iterations): fused accumulator copy + child eval without zero rows / + one division in PUCT | 0.531 s → 0.499 s → 0.473 s (−11%), same moves |
| n4d (768 hidden, 194M unique, 16 epochs) vs n4b (640), same code | +6 ± 22 (no gain: capacity offset by speed) |
| Draw head (n4a fine-tuned 4 epochs with draw loss 0.5) vs v1.5: eval contempt DE 0 / 1 / 2 | −13 / −54 / −104 (eval contempt hurts; draw loss also costs the value head) |
| Arena seat split, v1.6, 120 games (seat read from the game logs; the battle records list players in finishing order, not seat order) | first player: W 54 D 2 L 4 (92%); second: W 17 D 7 L 36 (34%) |
| Arena vs the top 25, v1.3 / v1.4 / v1.5 / v1.6 | 49% / 47% / 46% / 48%; Babebibobu 75% / 50% / 28% / 17%, MrSubZero 46% / 47% / 36% / 21% |
| Proven-draw contempt in the arena (v1.5, v1.6) | as first player the search left proven draws (valued 0.3) for lines just above 0.3 and lost them: reverted to 0.5 / 0.5 |
| Pondering vs v1.5 (Modal, 4 games on 8 CPUs) | +89 ± 22; without pondering +24 ± 21 |
| n6k (n4a fine-tuned on 15M fresh Kaggle positions only) vs n4a | −26 ± 22 |
| n6m (n4a fine-tuned on newest 137M) vs n4a | +14 ± 22 |
| Threat features (18 inputs: small-board win threats), val loss at 128 hidden, 8M positions | 0.6014 → 0.5920 (val-result 0.5921 → 0.5819) |
| n7t (n4a widened with threat inputs, same data and recipe as n6m) | val 0.5794 → 0.5769, val-result 0.5674 → 0.5647 |
| CG CPU | one core per bot (threads give no extra work) |
| Ponder on CG vs a top bot (Daporan) | ~4–5k iterations per opponent turn (~40 it/ms): the opponent's search takes the CPU; vs the default bot ~158 it/ms |
| Pre-game work at process start on CG | a 128 MB memset took 2.5 s before the first input was read: removed (first input now awaited 50 ms first) |
| Phase profile (x86, first turn, 300k iterations, n7t): descent / accumulator replay / expansion with child evaluation / backup | 29–34% / 13–15% / 50–53% / 1–3% of ~4k cycles per iteration; computing each prior's exp once: no change; division-free PUCT comparison: slower |
