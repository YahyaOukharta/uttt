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
| v1.8 | v1.7 engine (n7t) + opening book: 156 positions of non-won v1.7 arena games answered by 4M-iteration searches (won lines and second-player draws untouched) | replays of the 33 non-won top-25 games: 36% (first 83%, second 19%) vs v1.7 26% (50%, 17%); where a book move was played: 3 losses → draws, 1 draw → loss | 32.77 (final); vs the top 25: first 82%, second 11% | 10 |
| v1.9 | v1.8 + book grown to 386 positions: non-won games of v1.7 and v1.8 plus test games, 843 new positions searched (4M iterations, n8t) | local sanity only (book changes only positions of non-won lines) | 32.52 (final); vs the top 25: first 89%, second 11% (v1.8 on the same top 25: 84%, 14%) | 11 |
| v1.10 | book 572 entries: + v1.9 arena games and two book-loop cycles; **record entries**: where our own move varied and one move has the better record (≥ 3 games, or the others score ≤ 25%), replay it unless the long search rates it >0.05 worse (23 entries) | local sanity only | **34.04 (final)** | **5** |
| v1.11 | book 676 entries: + all v1.10 arena games (v1.10 vs the top 25: first 88%, second 16%; SupSon as second 10 / 10 draws, was all losses) and 16M searches of the new frequent positions | local sanity only | 33.55 (final); vs the top 25: first 90%, second 13% (v1.10 on the same top 25: 88%, 16%; overall 51.7% vs 52.2%: arena noise, Angecide resubmitted and played 40 of the games) | 7 |
| v1.12 | book 754 entries; **refuted moves**: a move of ours whose every observed opponent reply led to a position the long search proves lost is marked losing, propagated from the deepest positions up; the book plays the best other move (216 entries, 193 as second player, plies 20–40) | local sanity only | 33.79 (final); vs the top 25: first 90%, second 10% | 7 |
| v1.13 | book 1068 entries; positions that also occur in won games / second-player draws ("mixed") may change for a long-search gain ≥ 0.03 (e.g. vs Vingte as first player: 7 draws through a ply-24 move the search rates 0.08 worse) | local sanity only | submission 41720690 | |
| v1.14 | book 1638 entries: + 456 pre-emptive (opponent's likely replies to book moves, where a 35k-iteration search misplays) | 34.42, rank 5 / Legend (submission 41720886) |
| v1.15 | minified packing (90.6k chars); book 2549 entries (1279 pre-emptive from 4 rounds, 322 refuted, 47 by record) | **34.49, rank 4** (submission 41721094) |
| v1.16 | book 3146 entries (5 pre-emptive rounds: 1876) | 34.35, rank 6 (41721330) |
| v1.17 | book 3338 (+ v1.16 games, analysis on Kaggle after the Modal workspace paused) | 34.34, rank 7 (41721543) |
| v1.18 | + probed second-player lines (Apostolique win, Daporan draw: 37 forced entries) | **32.55, rank 11** (41721785): a record entry changed our first reply as second player (4 3 for 4 5 after 4 4, from 4 probe games vs 1447) and reshuffled every second-player game; mid-table draws lost (SupSon 0/9, RoboStac 0/7) |
| v1.19–v1.21 | same + record-margin fix (v1.20) + trunk protection (v1.21) | replaced within minutes |
| v1.15 again | user: keep the best rank unless sure | resubmitted (41721991): 34.20 at placement, 33.28 rank 8 the next morning after the top bots resubmitted (book lines stale) |
| v1.22 | v1.15's book + 133 entries, only at positions where v1.15 lost every arena game (`tools/refresh.py`), guard 0 | 34.75, rank 5 (41724565) |
| v1.23 | v1.22 + 85 (refresh loop, cycle 1) | 34.57, rank 7 (41724801); Angecide / Daporan resubmitted meanwhile |
| v1.24 | v1.23 + 116 (cycle 2) | **35.23, rank 3** (41725060), best so far |

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
| Extra inputs on top of threats, 128 hidden, 8M positions, 10 epochs: open boards each player can still win + dead open boards (27) | val 0.5802 → 0.5793 (macro threats earlier: −0.0011): too small to pay for the code and updates |
| Arena seat split, v1.7 vs the top 25 (116 games per seat) | first player W 96 D 17 L 3 (90%); second W 3 D 21 L 92 (12%); first-player draws drift from q 0.82 to 0.50 over moves 10–20 without a single blunder |
| CG CPU (probe) | Intel Haswell VM, AVX2 (no AVX-512, no VNNI), gcc 11.2 |
| Phase profile on CG (test game, n7t): descent / accumulator replay / expansion / backup | 22–31% / 14–19% / 51–60% / 1%; ~7.5–9k TSC cycles per iteration; 8.3 children scored per expansion |
| First-layer and output weights stored as int8 (sign-extended on load) instead of int16 | identical results; CG −10% iterations (35.6k → 31.9k per turn), Modal −9%: the evaluation is compute-bound, not cache-bound |
| n8t (n7t fine-tuned 6 epochs on 211M unique positions incl. g5/g6) vs n7t, same code, no ponder | +172 =275 −153, +11 ± 20 |
| Distillation, 128 hidden, 8M positions: teacher 1024 + hidden layer 32 (val 0.5725) → student on teacher targets, mu 1.0 / 0.5 | val 0.5802 → 0.5799 / 0.5798: the small net is capacity-bound, not target-noise-bound |
| Arena determinism (v1.7 as second player vs the top 25) | 113 of 116 games open 4 4, 4 5 (ours); 99 continue 4 8, 4 6 (ours) and lose 85; many opponents repeat whole games (sZoom: 12 losses, two distinct games) |
| Long searches (4M iterations, n8t) of our 1,364 positions in non-won v1.7 games | same move as played in most positions; 156 book entries (lost lines: any disagreement; first-player draws: gain ≥ 0.02; positions of won games and of second-player draws untouched) change 88 / 94 second-player losses, 3 / 3 first-player losses, 8 / 17 first-player draws and no won game |
| Replay suite of 33 non-won v1.7 games vs the top 25, v1.7 unchanged | 26% (first W2 D5 L2, second W2 D4 L18): replays reproduce only partly |
| Book hits in v1.8 arena games (156-entry book) | vs the top 25 as second player a book move came up in 73 of 104 games: still W2 D8 L63; the losses move deeper into the same lines (SupSon ×11 after ply 19, yoch ×9 after ply 27). After the 386-entry book, 114 of 135 distinct lost lines of v1.7/v1.8 are changed |
| Search parameter sweep, n8t, 30k iterations, 600 games each vs defaults (Modal) | CP 0.8 −13, CP 1.25 −17, PT 0.07 −39, PT 0.12 +14, PT 0.14 −6, VN 2 +10, VN 3 +13 / +6, VN 6 −17, VN 3 + PT 0.12 −2, FK 0.01 +1, FK 0.04 −9, LEAF 0.85 −4, null 0 (all ± 19–20); CN has no effect with PUCT (identical games): defaults kept |
| Book analysis depth: 16M vs 4M iterations on the 482 positions of ≥ 3 non-won games | best move differs in 40 / 482, mean root q change 0.026: the deep result is preferred (400 book entries) |
| Replays of v1.8's 25 lost / drawn top-25 battles with v1.9 (first 10 games; Angecide's agent had been replaced) | first W1 D2 L1, second W0 D0 L6: opponents leave the book lines later, so the book has to grow with every round |
| Book loop | arena + test-game positions → 4M searches on Modal (~$0.03 per 100 positions) → book → packed bot → replays of the lost battles (25 test games, ~25 min per cycle, 600 test games a day) |
| Second-player draws protected from the book (v1.9) | vs Babebibobu at ply 23 the v1.8 book move 6 6 drew 6 / 6; the draws protected the position, so v1.9 left it to the search, which played 7 8 / 6 7 and lost 17 / 17 (v1.9: Babebibobu second L8, v1.8 D8 / 13). Our search is not deterministic: protection must replay the move that drew, hence record entries |
| n9t: n8t fine-tuned 6 epochs on 319M unique positions (g3a–g8m, incl. g7 Kaggle and g8m Modal self-play of n7t / n8t), val 0.5697 → 0.5679 | vs n7t at 100 ms: +172 =282 −146, +15 ± 20; vs n8t: +159 =282 −159, 0 ± 20: the net has plateaued at this size and data |
| CReLU instead of SCReLU (one multiply per 16 neurons instead of two; Haswell has one multiply port), speed only, CG test game | first turn 224k vs 207k iterations (+8.6%), but the same build varies ±20% between CG runs (255k / 207k): ~5–10% at best, not worth a retrained net |
| Board symmetry in the book | of 9,547 position classes in our arena games only 374 occur in more than one orientation: a symmetric book would add ~4% hits |
| Why second-player lines are still lost (v1.11 vs Angecide, 10 identical losses) | the long searches rate the line a draw (q 0.50, unproven at 16M iterations) from ply 37 to 49; after Angecide's reply ply 51 is proven lost. The opponents see deeper in these endgames than a 4M–16M search, so their actual replies are used as refutations |
| Book storage | 3 characters per entry → sorted 30-bit hashes with Rice-coded gaps (k = 19) + 7-bit move: 1068 entries in 2,027 characters instead of 3,204 (decoder +440 characters); all entries verified by decoding |
| Loss report (`tools/losses.py`), v1.12's 129 non-won games vs the top 25: value drop split into our errors (long-search best − our move) and surprises (opponent reply better than the long search expected) | errors 20.3, surprises 100.2. Second player: start at 0.09–0.18 after 4 4 and slide to 0 by surprises (our errors ~0.05–0.13 per game): no blunders, the first player is just stronger. First player: real blunders at plies 18–38 (67 moves ≥ 0.15 in v1.10–v1.12, e.g. Daporan ply 38: 0.60 vs 0.98), none a proven win (no tactical solver would find them), all now answered by the book |
| Pre-emptive book (`tools/modal_pre.py`), round 1: for the 1,068 book moves, the opponent's 2 most likely replies (1M-iteration search) → 2,020 positions searched 4M and 35k iterations (~one turn of ours) | the short search picks another move in 692 (34%), worse by ≥ 0.03 in 209 and by ≥ 0.1 in 114: these become book entries before any arena game reaches them |
| Minified packing (`pack.py`, default): whitespace between tokens dropped where the tokens stay apart, code lines joined, frequent long identifiers replaced by two-character macros | 97,883 → 89,019 characters (−8.9k, ~4,500 more book entries); assembly identical to the unminified build (same md5); compiles and plays on CodinGame |
| Deep head, val loss at 60M positions / 10 epochs (Modal A10G, ~3 min per run) | from n8t: plain 0.5601, deep-16 0.5582, deep-32 0.5567; from scratch: plain 0.5634, deep-8 0.5619, deep-16 0.5610, deep-32 0.5599. Clear capacity gain, but dense inference costs ~3-4x per child evaluation |
| Second-seat probing (`tools/probe.py`), first 45 test games vs the top 6 | W1 D2: deviations at positions only reached in lost games; kept lines forced in the book. Angecide's old battles unreplayable (resubmits); Vingte / Daporan often leave the old line |
| Deep head in the engine (float test version, NN_DEEP) | 20k iterations 6.3 s vs 0.08 s plain: every game of the Kaggle equal-iteration match faulted on time. Integer estimate on Haswell (vpmaddwd on one port): deep-32 scoring every child ~3x per iteration and +18k chars (over the limit); deep-16 on the expanded node only ~15-30% slower and +9k chars for a val gain worth maybe +10-20 Elo. Parked |
| Replay check of the probed lines (`probe.py verify`): the battle of every kept probe line replayed with the candidate | Apostolique W → L, Daporan D ×3 → L ×3: the top bots repeat their openings but not whole games, so a single probe result does not reproduce. Probed lines are not used for submits |
| Book refresh after opponent resubmits (`tools/refresh.py`, `tools/refresh_loop.sh`) | v1.15 dropped 34.49 → 33.28 overnight with no change of ours (opponents resubmitted). Refreshing only positions the live version lost every game in: v1.22 34.75, v1.24 35.23 (rank 3). v1.15 vs the top 6: first player 0.83, second 0.01; vs the rest: 0.97 / 0.45 |
