# Research notes

What the strong Ultimate Tic-Tac-Toe bots do, collected on 2026-11-14. Items marked
(unverified) come from snippets or the author's own claims.

## The state of the arena

- **Pure random-rollout MCTS tops out around rank 50.** TomAlard (#2 in Sept 2026):
  "with excellent performance engineering, it is possible to reach about top 50"; the
  TrueSkill gap #1 → #50 is ~9, i.e. #1 wins >90% even when #50 plays X.
  [Fighting for #1 in the UTTT arena](https://tomalard.github.io/posts/fighting-for-1-in-the-ultimate-tic-tac-toe-arena/)
- **The top runs an NNUE value net + search**, trained on self-play:
  - TomAlard: (189 → 1024)×2 perspectives → 10 output buckets (next board: 9 boards + "any"),
    SCReLU, int16 + AVX2, ~3 inputs change per move (incremental), 300M+ self-play
    positions. Starter net 199 → 256 → 1 (~50k params). Search: "jacekmax" (below).
    Alpha-beta with the same net scored only ~40% against it.
  - MrSubZero (#3): alpha-beta + NNUE.
  - jacek: value-only MLP, one-hot inputs 81 × 15 (cell state, active board, "X/O would win
    the small board here"), incremental. [inputs article](https://tech.io/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/ultimate-tic-tac-toe)
  - reCurse: AlphaZero-style CNN ("TotoZero"), briefly #1 around 2021.
- **"Python3" top bots ship a compiled binary**: a Python stub writes a UTF-16-encoded
  (~15.9 bits/char) stripped binary and runs it (TomAlard; tool:
  [Agade09/CG-Send-Binary](https://github.com/Agade09/CG-Send-Binary/tree/base65536)).
  Treated as cheating in contests since 2017; tolerated in this arena (unverified policy).
  Gain claimed: clang + PGO ≈ +20% sims/s and more room for weights.
- **Self-play at the top: X wins 73%, O 3.5%, draws 23.5%.** Only good first move: centre of
  the centre board. Opening books "don't really help" (MrSubZero).

## Search: jacekmax / UBFM

Best-first minimax driven by UCT ([jacek, tech.io 55004](https://tech.io/playgrounds/55004)):
on reaching a leaf, expand **all** children and evaluate each with the net; back up
negamax (parent = −max child), select with value + C·sqrt(ln N / n), final move by
value + log(visits). TomAlard builds the leaf accumulator once and updates it per child, keeps
accumulator checkpoints in high-visit nodes, and found selection on the plain average of
backed-up values best (minimax weight drifted to 0).

## Pure MCTS: what is known to matter

| Technique | Evidence |
|---|---|
| MCTS-Solver | needed for top 50 even at 80–90k sims (darkhorse64); 65% vs plain in LOA (Winands 2008) |
| Win + block in playouts (decisive / anti-decisive) | "much better", little speed loss (MrSubZero). Win-only is *worse* (36% in Connect-4, Baier & Winands 2015) |
| Score-bounded MCTS (draws) | 57% vs solver in Connect-4 (Cazenave & Saffidine 2010); MrSubZero climbed with it + FPU tuning |
| Low C | "much less than √2" (jacek); C = 0.5 with 0 / ½ / 1 rewards (Di_Masta) |
| Speed | 100k+ rollouts/turn at the top of Legend (reCurse 2019); ~3 µs per playout; k-th free cell by index, xorshift, logf/LUT |
| Tree reuse, node pool | small gains, standard |
| RAVE / AMAF | low: a move's value depends on where it sends the opponent |
| Pondering / threads | ~10–20% at most, single core |

Static pruning ([minimax.dev](https://minimax.dev/docs/ultimate/pruning/)): never send the
opponent to a board they can win, or to a closed board, unless the move wins.

## Small nets and training

- KataGo ([Wu 2019](https://arxiv.org/abs/1902.10565)): playout cap randomization ×1.37,
  auxiliary ownership targets ×1.65 → for UTTT: small-board owners and final board margin.
- Gumbel AlphaZero (Danihelka 2022): learns with 2–16 simulations; useful for cheap data.
- CGZero ([marchete](https://github.com/marchete/CGZero)): value target k·z + (1−k)·q,
  duplicate averaging, 55% gate.
- uttt.ai ([ar-nowaczynski/utttai](https://github.com/ar-nowaczynski/utttai)): neural MCTS at
  1k sims beats rollout MCTS at 10M (5M-param net, too big for us).
- Arthur-Vat/codingame-bots (rank ~59, Oct 2026): 1.28M random playouts/s, 793k MCTS it/s on a
  2.1 GHz Xeon; NNUE at leaves +38 Elo over own rollouts; small 217→64→16 value net lost
  at equal time. [notes](https://github.com/Arthur-Vat/codingame-bots/tree/main/games/uttt)

## Rules / platform

- Theory: Bertholon et al. 2020 proved a first-player win, but for the variant where won
  boards stay playable. The CodinGame variant (closed boards, board-count tie-break) is unsolved.
- 768 MB memory, one vCPU (2.2–3.0 GHz), clock starts at the first input read, keep 5–10 ms
  margin. CodinGame compiles with g++ (no -O flag) → pragmas.
- Engine tricks: 512-entry win table, k-th set bit table, 2^18 tables over (mine, theirs) for
  "cells that win now", fill or mask closed boards.
