// Ultimate Tic-Tac-Toe (CodinGame "tic-tac-toe") - Monte Carlo tree search.
//
// Board: 9 small boards of 9 cells. A move is m = board * 9 + cell (cells row-major).
// The side to move plays in the small board named by the previous move's cell, unless that
// board is closed (won or full): then anywhere. Won boards are closed. Three won boards in a
// line win; if every board is closed without a line, more won boards wins (else a draw).
#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")
#if defined(__x86_64__) || defined(__i386__)
#pragma GCC target("avx2,bmi,bmi2,popcnt,lzcnt")
#include <immintrin.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <poll.h>
using namespace std;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#ifndef EXPWIN
#define EXPWIN 1   // expansion keeps only a game-winning move when there is one
#endif

// ------------------------------------------------------------------------------------------
// Parameters (overridable locally through UT_PARAMS="C=0.6,...")
// ------------------------------------------------------------------------------------------
static float P_C = 0.55f;         // UCT exploration constant (values in [0,1])
static bool netOn = false;        // a value net is loaded
static float P_CN = 0.2f;         // exploration constant with the net
static int P_NNSEL = 1;           // 1: PUCT with priors = softmax of the children's first net values
static float P_CP = 1.0f, P_PT = 0.1f;
static int P_CLIM = 1 << 21;      // compaction keeps at most this many nodes
static float P_DX = 0.5f, P_DO = 0.5f;  // value of a proven draw for us as first / second player
#ifdef LOCAL
static bool checkAcc = false;     // debug: verify incremental accumulators against a refresh
static long long checkCount = 0;
#endif
static int P_PONDER = 1;           // search during the opponent's turn
static float P_DE = 0.f;   // scale of the draw-head contempt in evaluations (needs a net with a draw head)
static float P_FK = 0.02f;         // final move: 0 = most visits; > 0: value + FK * ln(visits)
static int P_CT = 32;             // cache the accumulator of nodes with at least this many visits (0: off)
static int P_VN = 4;              // virtual visits given to a child's first net value
static float P_LEAF = 1.f;        // leaf value: LEAF * best child + (1 - LEAF) * prior-weighted mean
static int P_NNMODE = 0;          // 0: expand all children and score them, leaf = best child; 1: one eval per leaf
static float P_FPU = 0.8f;        // value of unvisited children (< 0: visit every child first)
static int P_ROLL = 1;            // rollout policy bits: 1 game win, 2 avoid sending to a game win, 4 board win, 8 board block
static float P_MS = 95.0f;        // per-turn budget, ms
static float P_FIRST_MS = 920.0f; // first-turn budget, ms

// ------------------------------------------------------------------------------------------
// Tables
// ------------------------------------------------------------------------------------------
static bool WIN[512];
static u8 POP[512];
static u8 NTH[512][9];  // NTH[mask][k] = index of the k-th set bit
static u16 TW[512];     // TW[mask] = cells c not in mask with WIN[mask | 1 << c]
static u8 MB[81], MC[81], MROW[81], MCOL[81];
static u8 GRID2M[81];   // row * 9 + col -> m

static void initTables() {
    static const int L[8] = {0x7, 0x38, 0x1C0, 0x49, 0x92, 0x124, 0x111, 0x54};
    for (int x = 0; x < 512; x++) {
        WIN[x] = false;
        for (int l : L) if ((x & l) == l) WIN[x] = true;
        POP[x] = (u8)__builtin_popcount(x);
        int k = 0;
        for (int i = 0; i < 9; i++) if (x >> i & 1) NTH[x][k++] = (u8)i;
    }
    for (int x = 0; x < 512; x++) {
        TW[x] = 0;
        for (int c = 0; c < 9; c++) if (!(x >> c & 1) && WIN[x | 1 << c]) TW[x] |= (u16)(1u << c);
    }
    for (int m = 0; m < 81; m++) {
        int b = m / 9, c = m % 9;
        MB[m] = (u8)b;
        MC[m] = (u8)c;
        MROW[m] = (u8)(b / 3 * 3 + c / 3);
        MCOL[m] = (u8)(b % 3 * 3 + c % 3);
        GRID2M[MROW[m] * 9 + MCOL[m]] = (u8)m;
    }
}

// ------------------------------------------------------------------------------------------
// Random
// ------------------------------------------------------------------------------------------
struct Rng {
    u64 s;
    explicit Rng(u64 seed = 0x9E3779B97F4A7C15ULL) : s(seed ? seed : 1) {}
    inline u32 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (u32)(s >> 32);
    }
    inline u32 below(u32 n) { return (u32)(((u64)next() * n) >> 32); }
};

// ------------------------------------------------------------------------------------------
// Position
// ------------------------------------------------------------------------------------------
enum { ONGOING = 0, WIN0 = 1, WIN1 = 2, DRAW = 3 };

struct State {
    u16 b[2][9];   // cells of each player, per small board
    u16 macro[2];  // small boards won by each player
    u16 closed;    // small boards won or full
    int8_t next;   // forced small board, -1 = any open board
    u8 side;       // player to move (0 = first player)
    u8 result;     // ONGOING / WIN0 / WIN1 / DRAW
    u8 ply;

    void reset() { memset(this, 0, sizeof *this); next = -1; }
    inline u16 empty(int bd) const { return (u16)(~(b[0][bd] | b[1][bd]) & 0x1FF); }
    inline void play(int m) {
        const int bd = MB[m], c = MC[m], p = side;
        const u16 nb = b[p][bd] | (u16)(1u << c);
        b[p][bd] = nb;
        if (WIN[nb]) {
            macro[p] |= (u16)(1u << bd);
            closed |= (u16)(1u << bd);
            if (WIN[macro[p]]) result = (u8)(WIN0 + p);
        } else if ((nb | b[p ^ 1][bd]) == 0x1FF) {
            closed |= (u16)(1u << bd);
        }
        if (!result && closed == 0x1FF) {
            int a = POP[macro[0]], o = POP[macro[1]];
            result = a > o ? WIN0 : (o > a ? WIN1 : DRAW);
        }
        next = (closed >> c & 1) ? -1 : (int8_t)c;
        side ^= 1;
        ply++;
    }
    // Legal moves in ascending m order. Returns count.
    inline int gen(u8* out) const {
        int n = 0;
        if (result) return 0;
        if (next >= 0) {
            u16 e = empty(next);
            const int base = next * 9;
            while (e) { out[n++] = (u8)(base + __builtin_ctz(e)); e &= e - 1; }
        } else {
            u16 open = (u16)(~closed & 0x1FF);
            while (open) {
                const int bd = __builtin_ctz(open);
                open &= open - 1;
                u16 e = empty(bd);
                const int base = bd * 9;
                while (e) { out[n++] = (u8)(base + __builtin_ctz(e)); e &= e - 1; }
            }
        }
        return n;
    }
};

static inline int pickBit(u16 m, Rng& rng) { return NTH[m][rng.below(POP[m])]; }

// Playout to the end. F = policy bits (see P_ROLL). Returns the result.
template <int F>
static inline int rolloutT(State s, Rng& rng) {
    while (!s.result) {
        const int p = s.side, o = p ^ 1;
        int bd = s.next, c = -1;
        if (F & 1) {  // a move that wins the game
            u16 tb = (u16)(TW[s.macro[p]] & ~s.closed & 0x1FF);
            if (bd >= 0) tb &= (u16)(1u << bd);
            while (tb) {
                const int x = __builtin_ctz(tb);
                tb &= tb - 1;
                const u16 wc = TW[s.b[p][x]] & s.empty(x);
                if (wc) { bd = x; c = __builtin_ctz(wc); break; }
            }
        }
        if (c < 0 && bd >= 0) {
            const u16 e = s.empty(bd);
            if (F & 4) { const u16 w = TW[s.b[p][bd]] & e; if (w) c = pickBit(w, rng); }
            if (c < 0 && (F & 8)) { const u16 w = TW[s.b[o][bd]] & e; if (w) c = pickBit(w, rng); }
            if (c < 0 && (F & 2)) {  // do not send the opponent where it wins the game
                const u16 open = (u16)(~s.closed & 0x1FF);
                u16 danger = (u16)(TW[s.macro[o]] & open), bad = 0;
                while (danger) {
                    const int x = __builtin_ctz(danger);
                    danger &= danger - 1;
                    if (TW[s.b[o][x]] & s.empty(x)) bad |= (u16)(1u << x);
                }
                if (bad) {
                    const u16 safe = e & open & (u16)~bad;
                    if (safe) c = pickBit(safe, rng);
                }
            }
            if (c < 0) c = pickBit(e, rng);
        } else if (c < 0) {
            u8 cnt[9];
            int tot = 0;
            for (int i = 0; i < 9; i++) {
                cnt[i] = (s.closed >> i & 1) ? 0 : POP[s.empty(i)];
                tot += cnt[i];
            }
            int k = (int)rng.below((u32)tot);
            bd = 0;
            while (k >= cnt[bd]) k -= cnt[bd++];
            c = NTH[s.empty(bd)][k];
        }
        s.play(bd * 9 + c);
    }
    return s.result;
}

static inline int rollout(const State& s, Rng& rng) {
    switch (P_ROLL) {
        case 1: return rolloutT<1>(s, rng);
        case 2: return rolloutT<2>(s, rng);
        case 3: return rolloutT<3>(s, rng);
        case 4: return rolloutT<4>(s, rng);
        case 5: return rolloutT<5>(s, rng);
        case 7: return rolloutT<7>(s, rng);
        case 9: return rolloutT<9>(s, rng);
        case 11: return rolloutT<11>(s, rng);
        case 12: return rolloutT<12>(s, rng);
        case 13: return rolloutT<13>(s, rng);
        case 15: return rolloutT<15>(s, rng);
        default: return rolloutT<0>(s, rng);
    }
}

// ------------------------------------------------------------------------------------------
// Tree
// ------------------------------------------------------------------------------------------
// Node values are from the point of view of the player who made the move into the node.
enum { PR_UNK = 0, PR_WIN = 1, PR_DRAW = 2, PR_LOSS = 3 };

struct Node {
    int32_t first;  // index of the first child, -1 = not expanded
    int32_t N;
    float W;
    u8 n;           // number of children
    u8 move;
    u8 proven;
    u8 pad;
};

static int POOL = 1 << 23;
static Node* pool[2] = {nullptr, nullptr};
static int cur = 0;      // active pool
static int used = 0;     // nodes used in the active pool
static int root = 0;
static State rootState;
static Rng rng(0x1234567ULL);
static float LOGT[1 << 16];

static inline float fastlog(int n) { return n < (1 << 16) ? LOGT[n] : logf((float)n); }

static uint32_t treeGen = 1;  // bumped whenever node indices are reassigned

static inline int newRoot() {
    Node* P = pool[cur];
    treeGen++;
    used = 0;
    P[0] = Node{-1, 0, 0.f, 0, 0, PR_UNK, 0};
    used = 1;
    return 0;
}

static inline bool expand(int idx, const State& s) {
    u8 mv[81];
#if EXPWIN
    {   // a move that wins the game: keep only that child, the node is proven
        const int p = s.side;
        u16 tb = (u16)(TW[s.macro[p]] & ~s.closed & 0x1FF);
        if (s.next >= 0) tb &= (u16)(1u << s.next);
        while (tb) {
            const int x = __builtin_ctz(tb);
            tb &= tb - 1;
            const u16 wc = TW[s.b[p][x]] & s.empty(x);
            if (wc) {
                if (used + 1 > POOL) return false;
                Node* P = pool[cur];
                P[used] = Node{-1, 0, 0.f, 0, (u8)(x * 9 + __builtin_ctz(wc)), PR_WIN, 0};
                P[idx].first = used++;
                P[idx].n = 1;
                P[idx].proven = PR_LOSS;
                return true;
            }
        }
    }
#endif
    const int n = s.gen(mv);
    if (used + n > POOL) return false;
    Node* P = pool[cur];
    const int f = used;
    for (int i = 0; i < n; i++) P[f + i] = Node{-1, 0, 0.f, 0, mv[i], PR_UNK, 0};
    used += n;
    P[idx].first = f;
    P[idx].n = (u8)n;
    return true;
}

static inline float provenValue(int p) { return p == PR_WIN ? 1.f : (p == PR_DRAW ? 0.5f : 0.f); }
// Proven-draw contempt: a draw is worth P_DX to us when we move first, P_DO when we move second.
static int rootSide = 0;
static inline float provenValueM(int p, int mover) {
    if (p != PR_DRAW) return p == PR_WIN ? 1.f : 0.f;
    const float d = rootSide == 0 ? P_DX : P_DO;
    return mover == rootSide ? d : 1.f - d;
}

// Re-check the proof status of node idx from its children (children moved by the opponent of
// idx's mover). Returns true if idx became proven.
static inline bool tryProve(int idx) {
    Node* P = pool[cur];
    Node& nd = P[idx];
    if (nd.first < 0) return false;
    bool all = true, draw = false;
    for (int i = 0; i < nd.n; i++) {
        const u8 pr = P[nd.first + i].proven;
        if (pr == PR_WIN) { nd.proven = PR_LOSS; return true; }
        if (pr == PR_UNK) all = false;
        else if (pr == PR_DRAW) draw = true;
    }
    if (all) { nd.proven = draw ? PR_DRAW : PR_WIN; return true; }
    return false;
}

static inline int selectChild(int idx) {
    Node* P = pool[cur];
    const Node& nd = P[idx];
    if (netOn && P_NNSEL) {
        const float c = P_CP * sqrtf((float)nd.N);
        int best = -1;
        float bv = -1e30f;
        for (int i = 0; i < nd.n; i++) {
            const int ch = nd.first + i;
            const Node& x = P[ch];
            if (x.proven == PR_LOSS) continue;
            if (x.proven == PR_WIN) return ch;
            const float u = c * ((float)x.pad * (1.f / 255.f) + 0.002f);
            const float v = x.N ? (x.W + u) / (float)x.N : 0.5f + u;
            if (v > bv) { bv = v; best = ch; }
        }
        return best < 0 ? nd.first : best;
    }
    const float c = (netOn ? P_CN : P_C) * sqrtf(fastlog(nd.N));
    int best = -1;
    float bv = -1e30f;
    for (int i = 0; i < nd.n; i++) {
        const int ch = nd.first + i;
        const Node& x = P[ch];
        if (x.N == 0) {
            if (P_FPU < 0) return ch;
            if (P_FPU + c > bv) { bv = P_FPU + c; best = ch; }
            continue;
        }
        if (x.proven == PR_LOSS) continue;
        if (x.proven == PR_WIN) return ch;
        const float v = x.W / x.N + c / sqrtf((float)x.N);
        if (v > bv) { bv = v; best = ch; }
    }
    return best < 0 ? nd.first : best;
}

static int pathBuf[96];

static void iterate() {
    Node* P = pool[cur];
    State s = rootState;
    int depth = 0, idx = root;
    pathBuf[depth++] = idx;
    for (;;) {
        Node& nd = P[idx];
        if (nd.proven || s.result) break;
        if (nd.first < 0) {
            if (nd.N == 0 && idx != root) break;
            if (!expand(idx, s)) break;
            if (nd.proven) break;
        }
        idx = selectChild(idx);
        s.play(P[idx].move);
        pathBuf[depth++] = idx;
        if (s.result && !P[idx].proven) {
            const int mover = s.side ^ 1;
            P[idx].proven = s.result == DRAW ? PR_DRAW : (s.result == WIN0 + mover ? PR_WIN : PR_LOSS);
        }
    }
    float v;
    bool proof = false;
    if (P[idx].proven) {
        v = provenValue(P[idx].proven);
        proof = depth > 1;
    } else {
        const int r = rollout(s, rng);
        const int mover = s.side ^ 1;
        v = r == DRAW ? 0.5f : (r == WIN0 + mover ? 1.f : 0.f);
    }
    for (int i = depth - 1; i >= 0; i--) {
        Node& nd = P[pathBuf[i]];
        nd.N++;
        nd.W += v;
        v = 1.f - v;
        if (proof && i < depth - 1) proof = tryProve(pathBuf[i]);
    }
}

// ------------------------------------------------------------------------------------------
// Value network (NNUE style): 189 one-hot inputs per perspective -> NN_H, SCReLU, both
// perspectives (side to move first) -> 10 heads chosen by the next board.
// ------------------------------------------------------------------------------------------
#ifndef NN_H
#define NN_H 0
#endif
#if NN_H
#ifdef NN_THR
static const int NQA = 255, NQB = 64, NIN = 207;  // + 18 inputs: small-board win threats
#else
static const int NQA = 255, NQB = 64, NIN = 189;
#endif
alignas(32) static int16_t NW[NIN][NN_H];
#ifdef NN_THR
alignas(32) static int16_t NWN[18][NN_H];  // negated threat rows (a lost threat is added as a row)
#endif
alignas(32) static int16_t NBIAS[NN_H];
alignas(32) static int16_t NNZ[NN_H];
alignas(32) static int16_t NO[10][2 * NN_H];
static int32_t NOB[10];
alignas(32) static int16_t ND[10][2 * NN_H];  // optional second head: P(draw)
static int32_t NDB[10];
static bool hasDraw = false, useDraw = false;
static float drawK[2] = {0.f, 0.f};  // eval contempt: value += drawK[side to move] * P(draw)
struct Acc { alignas(32) int16_t v[2][NN_H]; };
static Acc rootAcc;

static inline void addF(int16_t* __restrict a, int f) {
    const int16_t* __restrict w = NW[f];
    for (int i = 0; i < NN_H; i++) a[i] += w[i];
}
static inline void addF2(int16_t* __restrict a, int f, int g) {
    const int16_t* __restrict w = NW[f];
    const int16_t* __restrict x = NW[g];
    for (int i = 0; i < NN_H; i++) a[i] += w[i] + x[i];
}
#ifdef NN_THR
// Player q threatens board bd: two in a line with the third cell empty, board still open.
static inline int thr(const State& s, int q, int bd) {
    return !(s.closed >> bd & 1) && (TW[s.b[q][bd]] & s.empty(bd)) ? 1 : 0;
}
static inline void subF(int16_t* __restrict a, int f) {
    const int16_t* __restrict w = NW[f];
    for (int i = 0; i < NN_H; i++) a[i] -= w[i];
}
// Threat changes on the board of move m (perspective P: 189 + bd = own threat, 198 + bd = opponent's).
static inline void accThreats(Acc& A, const State& before, const State& after, int m) {
    const int bd = MB[m];
    for (int q = 0; q < 2; q++) {
        const int d = thr(after, q, bd) - thr(before, q, bd);
        if (d > 0) { addF(A.v[q], 189 + bd); addF(A.v[q ^ 1], 198 + bd); }
        else if (d < 0) { subF(A.v[q], 189 + bd); subF(A.v[q ^ 1], 198 + bd); }
    }
}
#endif
static inline void addFTo(int16_t* __restrict d, const int16_t* __restrict a, int f) {
    const int16_t* __restrict w = NW[f];
    for (int i = 0; i < NN_H; i++) d[i] = (int16_t)(a[i] + w[i]);
}
static inline void addF2To(int16_t* __restrict d, const int16_t* __restrict a, int f, int g) {
    const int16_t* __restrict w = NW[f];
    const int16_t* __restrict x = NW[g];
    for (int i = 0; i < NN_H; i++) d[i] = (int16_t)(a[i] + w[i] + x[i]);
}
// D = S updated for move m (copy and update in one pass).
static inline void accMoveTo(Acc& D, const Acc& S, const State& before, const State& after, int m) {
    const int p = before.side, bd = MB[m];
    if ((after.closed ^ before.closed) >> bd & 1) {
        if (after.macro[p] >> bd & 1) { addF2To(D.v[p], S.v[p], m, 162 + bd); addF2To(D.v[p ^ 1], S.v[p ^ 1], 81 + m, 171 + bd); }
        else { addF2To(D.v[p], S.v[p], m, 180 + bd); addF2To(D.v[p ^ 1], S.v[p ^ 1], 81 + m, 180 + bd); }
    } else {
        addFTo(D.v[p], S.v[p], m);
        addFTo(D.v[p ^ 1], S.v[p ^ 1], 81 + m);
    }
#ifdef NN_THR
    accThreats(D, before, after, m);
#endif
}
// Update A for move m that took `before` to `after`.
static inline void accMove(Acc& A, const State& before, const State& after, int m) {
    const int p = before.side, bd = MB[m];
    if ((after.closed ^ before.closed) >> bd & 1) {
        if (after.macro[p] >> bd & 1) { addF2(A.v[p], m, 162 + bd); addF2(A.v[p ^ 1], 81 + m, 171 + bd); }
        else { addF2(A.v[p], m, 180 + bd); addF2(A.v[p ^ 1], 81 + m, 180 + bd); }
    } else {
        addF(A.v[p], m);
        addF(A.v[p ^ 1], 81 + m);
    }
#ifdef NN_THR
    accThreats(A, before, after, m);
#endif
}
static void accRefresh(Acc& A, const State& s) {
    for (int P = 0; P < 2; P++) {
        memcpy(A.v[P], NBIAS, sizeof NBIAS);
        for (int bd = 0; bd < 9; bd++) {
            for (int q = 0; q < 2; q++) {
                u16 x = s.b[q][bd];
                while (x) { const int c = __builtin_ctz(x); x &= x - 1; addF(A.v[P], (q == P ? 0 : 81) + bd * 9 + c); }
                if (s.macro[q] >> bd & 1) addF(A.v[P], (q == P ? 162 : 171) + bd);
            }
            if ((s.closed >> bd & 1) && !((s.macro[0] | s.macro[1]) >> bd & 1)) addF(A.v[P], 180 + bd);
#ifdef NN_THR
            for (int q = 0; q < 2; q++)
                if (thr(s, q, bd)) addF(A.v[P], (q == P ? 189 : 198) + bd);
#endif
        }
    }
}
// Value of the position for the side to move, in [0, 1].
#if defined(__x86_64__) && NN_H % 16 == 0
static inline int32_t hsum8(__m256i acc) {
    __m128i x = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4E));
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1));
    return _mm_cvtsi128_si32(x);
}
#define UT_LD(p) _mm256_load_si256((const __m256i*)(p))
// clipped activation c of 16 neurons -> acc += c*c*o (and accd += c*c*od for the draw head)
#define UT_STEP(C, OFF) { const __m256i c_ = _mm256_min_epi16(_mm256_max_epi16(C, zero), qa); \
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_mullo_epi16(c_, UT_LD(o + (OFF))), c_)); \
        if (D) accd = _mm256_add_epi32(accd, _mm256_madd_epi16(_mm256_mullo_epi16(c_, UT_LD(od + (OFF))), c_)); }
#endif
template <bool D>
static inline float nnOut(int32_t sum, int32_t sumd, int bk, int stm) {
    const float logit = ((float)sum / NQA + (float)NOB[bk]) / (float)(NQA * NQB);
    float v = 1.f / (1.f + expf(-logit));
    if (D) {
        const float ld = ((float)sumd / NQA + (float)NDB[bk]) / (float)(NQA * NQB);
        v += drawK[stm] / (1.f + expf(-ld));
    }
    return v;
}
template <bool D>
static inline float nnEvalT(const Acc& A, const State& s) {
    const int stm = s.side, bk = s.next < 0 ? 9 : s.next;
    const int16_t* __restrict o = NO[bk];
    const int16_t* __restrict od = ND[bk];
    const int16_t* __restrict a = A.v[stm];
    const int16_t* __restrict b = A.v[stm ^ 1];
    int32_t sum = 0, sumd = 0;
#if defined(__x86_64__) && NN_H % 16 == 0
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(NQA);
    __m256i acc = zero, accd = zero;
    for (int i = 0; i < NN_H; i += 16) {
        UT_STEP(UT_LD(a + i), i)
        UT_STEP(UT_LD(b + i), NN_H + i)
    }
    sum = hsum8(acc);
    if (D) sumd = hsum8(accd);
#else
    for (int i = 0; i < NN_H; i++) {
        const int16_t c = (int16_t)min(max((int)a[i], 0), NQA);
        sum += (int32_t)(int16_t)(c * o[i]) * c;
        if (D) sumd += (int32_t)(int16_t)(c * od[i]) * c;
    }
    for (int i = 0; i < NN_H; i++) {
        const int16_t c = (int16_t)min(max((int)b[i], 0), NQA);
        sum += (int32_t)(int16_t)(c * o[NN_H + i]) * c;
        if (D) sumd += (int32_t)(int16_t)(c * od[NN_H + i]) * c;
    }
#endif
    return nnOut<D>(sum, sumd, bk, stm);
}
static inline float nnEval(const Acc& A, const State& s) {
    return useDraw ? nnEvalT<true>(A, s) : nnEvalT<false>(A, s);
}
//@NET_DATA@
#ifdef NN_EMBED
// Embedded net: a bit stream at 15 bits per character; first layer and biases int8, output int16.
#ifdef NN_HUFF
static void loadEmbedded() {  // canonical Huffman-coded int8 weights (see tools/pack.py)
    const char16_t* p = NN_DATA;
    u32 acc = 0;
    int nb = 0;
    auto bit = [&]() -> int {
        if (!nb) { acc = (u32)(*p++ - NN_OFF); nb = 15; }
        const int b = (int)(acc & 1);
        acc >>= 1;
        nb--;
        return b;
    };
    auto bits = [&](int n) { int v = 0; for (int i = 0; i < n; i++) v |= bit() << i; return v; };
    for (int k = 0; k < 10; k++) { const u32 hi = (u32)bits(16); NOB[k] = (int32_t)(hi << 16 | (u32)bits(16)); }
#ifdef NN_DRAW
    for (int k = 0; k < 10; k++) { const u32 hi = (u32)bits(16); NDB[k] = (int32_t)(hi << 16 | (u32)bits(16)); }
    hasDraw = true;
#endif
    int len[256], cnt[32] = {0}, pos[33] = {0}, sym[256];
    for (int i = 0; i < 256; i++) cnt[len[i] = bits(5)]++;
    cnt[0] = 0;
    for (int l = 1; l < 32; l++) pos[l + 1] = pos[l] + cnt[l];
    for (int i = 0; i < 256; i++) if (len[i]) sym[pos[len[i]]++] = i;
    auto dec = [&]() -> int8_t {
        int code = 0, first = 0, index = 0;
        for (int l = 1; l < 32; l++) {
            code |= bit();
            if (code - first < cnt[l]) return (int8_t)sym[index + code - first];
            index += cnt[l];
            first = (first + cnt[l]) << 1;
            code <<= 1;
        }
        return 0;
    };
    for (int f = 0; f < NIN; f++)
        for (int i = 0; i < NN_H; i++) NW[f][i] = dec();
    for (int i = 0; i < NN_H; i++) NBIAS[i] = dec();
    for (int k = 0; k < 10; k++)
        for (int i = 0; i < 2 * NN_H; i++) NO[k][i] = dec();
#ifdef NN_DRAW
    for (int k = 0; k < 10; k++)
        for (int i = 0; i < 2 * NN_H; i++) ND[k][i] = dec();
#endif
}
#else
static void loadEmbedded() {
    const char16_t* p = NN_DATA;
    u32 acc = 0;
    int nb = 0;
    auto byte = [&]() -> int {
        while (nb < 8) { acc |= (u32)(*p++ - NN_OFF) << nb; nb += 15; }
        const int b = (int)(acc & 255);
        acc >>= 8;
        nb -= 8;
        return b;
    };
    auto word = [&]() -> int { const int lo = byte(); return lo | byte() << 8; };
    for (int f = 0; f < NIN; f++)
        for (int i = 0; i < NN_H; i++) NW[f][i] = (int8_t)byte();
    for (int i = 0; i < NN_H; i++) NBIAS[i] = (int8_t)byte();
    for (int k = 0; k < 10; k++)
        for (int i = 0; i < 2 * NN_H; i++) NO[k][i] = (int8_t)byte();
    for (int k = 0; k < 10; k++) { const u32 hi = (u32)word(); NOB[k] = (int32_t)(hi << 16 | (u32)word()); }
}
#endif
#endif
static void finishNet() {
#ifdef NN_THR
    for (int k = 0; k < 9; k++)
        for (int i = 0; i < NN_H; i++) { NWN[k][i] = (int16_t)-NW[189 + k][i]; NWN[9 + k][i] = (int16_t)-NW[198 + k][i]; }
#endif
}
static bool loadNet(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    bool ok = fread(NW, sizeof NW, 1, f) == 1 && fread(NBIAS, sizeof NBIAS, 1, f) == 1 &&
              fread(NO, sizeof NO, 1, f) == 1 && fread(NOB, sizeof NOB, 1, f) == 1;
    hasDraw = ok && fread(ND, sizeof ND, 1, f) == 1 && fread(NDB, sizeof NDB, 1, f) == 1;
    fclose(f);
    return ok;
}

// One iteration with the net: descend by UCT, expand every child of the leaf and score each
// with the net; the leaf's value is the best child's (one ply of minimax), backed up as an average.
// Value (side to move after m) of the position reached by move m from `before`, computed from
// the accumulator of `before` without materialising the child accumulator. Same arithmetic as
// accMove + nnEval.
template <bool D>
static inline float nnEvalMoveT(const Acc& A, const State& before, const State& after, int m) {
    const int p = before.side, bd = MB[m];
    const int16_t *wp1 = NW[m], *wq1 = NW[81 + m], *wp2 = NNZ, *wq2 = NNZ;
    if ((after.closed ^ before.closed) >> bd & 1) {
        if (after.macro[p] >> bd & 1) { wp2 = NW[162 + bd]; wq2 = NW[171 + bd]; }
        else { wp2 = NW[180 + bd]; wq2 = NW[180 + bd]; }
    }
    const int16_t *wp3 = NNZ, *wq3 = NNZ, *wp4 = NNZ, *wq4 = NNZ;
#ifdef NN_THR
    {   // threat changes: rows for perspective p (wp*) and p ^ 1 (wq*)
        const int dp = thr(after, p, bd) - thr(before, p, bd), dq = thr(after, p ^ 1, bd) - thr(before, p ^ 1, bd);
        if (dp) { wp3 = dp > 0 ? NW[189 + bd] : NWN[bd]; wq3 = dp > 0 ? NW[198 + bd] : NWN[9 + bd]; }
        if (dq) { wq4 = dq > 0 ? NW[189 + bd] : NWN[bd]; wp4 = dq > 0 ? NW[198 + bd] : NWN[9 + bd]; }
    }
#endif
    const int bk = after.next < 0 ? 9 : after.next;
    const int16_t* __restrict o = NO[bk];
    const int16_t* __restrict od = ND[bk];
    const int16_t* __restrict a = A.v[p ^ 1];   // side to move after m
    const int16_t* __restrict b = A.v[p];
    int32_t sum = 0, sumd = 0;
#if defined(__x86_64__) && NN_H % 16 == 0
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(NQA);
    __m256i acc = zero, accd = zero;
    if (wp3 != NNZ || wp4 != NNZ) {  // a threat changed: up to four rows per perspective
        for (int i = 0; i < NN_H; i += 16) {
            UT_STEP(_mm256_add_epi16(_mm256_add_epi16(UT_LD(a + i), UT_LD(wq1 + i)),
                    _mm256_add_epi16(UT_LD(wq2 + i), _mm256_add_epi16(UT_LD(wq3 + i), UT_LD(wq4 + i)))), i)
            UT_STEP(_mm256_add_epi16(_mm256_add_epi16(UT_LD(b + i), UT_LD(wp1 + i)),
                    _mm256_add_epi16(UT_LD(wp2 + i), _mm256_add_epi16(UT_LD(wp3 + i), UT_LD(wp4 + i)))), NN_H + i)
        }
    } else if (wq2 == NNZ) {  // no board closed (most moves): skip the zero rows
        for (int i = 0; i < NN_H; i += 16) {
            UT_STEP(_mm256_add_epi16(UT_LD(a + i), UT_LD(wq1 + i)), i)
            UT_STEP(_mm256_add_epi16(UT_LD(b + i), UT_LD(wp1 + i)), NN_H + i)
        }
    } else {
        for (int i = 0; i < NN_H; i += 16) {
            UT_STEP(_mm256_add_epi16(UT_LD(a + i), _mm256_add_epi16(UT_LD(wq1 + i), UT_LD(wq2 + i))), i)
            UT_STEP(_mm256_add_epi16(UT_LD(b + i), _mm256_add_epi16(UT_LD(wp1 + i), UT_LD(wp2 + i))), NN_H + i)
        }
    }
    sum = hsum8(acc);
    if (D) sumd = hsum8(accd);
#else
    for (int i = 0; i < NN_H; i++) {
        const int16_t v = (int16_t)(a[i] + (int16_t)(wq1[i] + wq2[i] + wq3[i] + wq4[i]));
        const int16_t c = (int16_t)min(max((int)v, 0), NQA);
        sum += (int32_t)(int16_t)(c * o[i]) * c;
        if (D) sumd += (int32_t)(int16_t)(c * od[i]) * c;
    }
    for (int i = 0; i < NN_H; i++) {
        const int16_t v = (int16_t)(b[i] + (int16_t)(wp1[i] + wp2[i] + wp3[i] + wp4[i]));
        const int16_t c = (int16_t)min(max((int)v, 0), NQA);
        sum += (int32_t)(int16_t)(c * o[NN_H + i]) * c;
        if (D) sumd += (int32_t)(int16_t)(c * od[NN_H + i]) * c;
    }
#endif
    return nnOut<D>(sum, sumd, bk, p ^ 1);
}
static inline float nnEvalMove(const Acc& A, const State& before, const State& after, int m) {
    return useDraw ? nnEvalMoveT<true>(A, before, after, m) : nnEvalMoveT<false>(A, before, after, m);
}
// Accumulators of frequently visited nodes, direct-mapped by node index. The descent only
// replays the plies below the deepest cached ancestor.
static const int ACN = 1 << 13;
static Acc accCache[ACN];
static int32_t accKey[ACN];
static uint32_t accGen[ACN];
static inline const Acc* cacheGet(int idx) {
    const int h = idx & (ACN - 1);
    return accKey[h] == idx && accGen[h] == treeGen ? &accCache[h] : nullptr;
}
static inline void cachePut(int idx, const Acc& A) {
    const int h = idx & (ACN - 1);
    accKey[h] = idx;
    accGen[h] = treeGen;
    accCache[h] = A;
}
static State pendS[96];
static int pendI[96];

static void iterateNN() {
    Node* P = pool[cur];
    State s = rootState;
    const Acc* base = &rootAcc;
    int depth = 0, idx = root, np = 0;
    pathBuf[depth++] = idx;
    while (P[idx].first >= 0 && !P[idx].proven) {
        idx = selectChild(idx);
        const Acc* c = P_CT > 0 ? cacheGet(idx) : nullptr;
        if (c) { base = c; np = 0; }
        else { pendS[np] = s; pendI[np++] = idx; }
        s.play(P[idx].move);
        pathBuf[depth++] = idx;
    }
    Acc A;
    const Acc* Ap = base;  // accumulator of s
    for (int i = 0; i < np; i++) {
        if (i == 0) accMoveTo(A, *base, pendS[0], np > 1 ? pendS[1] : s, P[pendI[0]].move);
        else accMove(A, pendS[i], i + 1 < np ? pendS[i + 1] : s, P[pendI[i]].move);
        Ap = &A;
        if (P_CT > 0 && P[pendI[i]].N >= P_CT) cachePut(pendI[i], A);
    }
#ifdef LOCAL
    if (checkAcc && !s.result) {   // debug: incremental accumulator and fused child eval vs refresh
        Acc R;
        accRefresh(R, s);
        if (memcmp(&R, Ap, sizeof R)) { fprintf(stderr, "ACC MISMATCH ply %d\n", s.ply); exit(5); }
        u8 mv[81];
        const int n = s.gen(mv);
        for (int i = 0; i < n; i++) {
            State cs = s;
            cs.play(mv[i]);
            if (cs.result) continue;
            Acc C;
            accRefresh(C, cs);
            const float f = nnEvalMove(*Ap, s, cs, mv[i]), g = nnEval(C, cs);
            if (fabsf(f - g) > 1e-6f) { fprintf(stderr, "EVALMOVE MISMATCH %f %f ply %d\n", f, g, s.ply); exit(6); }
        }
        checkCount++;
    }
#endif
    float v;
    bool proof = false;
    if (P_NNMODE == 1 && !P[idx].proven && !s.result && (P[idx].N > 0 || idx == root)) {
        // classic value-net MCTS: expand without scoring, step into one child, score it
        if (expand(idx, s) && !P[idx].proven) {
            idx = selectChild(idx);
            const State b = s;
            s.play(P[idx].move);
            if (Ap != &A) { A = *Ap; Ap = &A; }
            accMove(A, b, s, P[idx].move);
            pathBuf[depth++] = idx;
        }
    }
    Node& L = P[idx];
    if (P_NNMODE == 1 && !L.proven && !s.result) {
        v = 1.f - nnEval(*Ap, s);
    } else if (!L.proven && !s.result && expand(idx, s) && !L.proven) {
        float best = -1.f;
        for (int i = 0; i < L.n; i++) {
            Node& ch = P[L.first + i];
            State cs = s;
            cs.play(ch.move);
            float cv;  // value for the player who made the move (s.side)
            if (cs.result) {
                ch.proven = cs.result == DRAW ? PR_DRAW : (cs.result == WIN0 + s.side ? PR_WIN : PR_LOSS);
                cv = provenValueM(ch.proven, s.side);
            } else {
                cv = 1.f - nnEvalMove(*Ap, s, cs, ch.move);
            }
            ch.N = P_VN;
            ch.W = cv * P_VN;
            if (cv > best) best = cv;
        }
        float lv = best;
        if (P_NNSEL) {  // priors: softmax of first values over siblings, stored in pad (1/255 units)
            float sum = 0.f, wsum = 0.f;
            for (int i = 0; i < L.n; i++) {
                const float cv = P[L.first + i].W / P_VN, e = expf((cv - best) / P_PT);
                sum += e;
                wsum += e * cv;
            }
            const float inv = 255.f / sum;
            for (int i = 0; i < L.n; i++) {
                Node& ch = P[L.first + i];
                ch.pad = (u8)min(255.f, expf((ch.W / P_VN - best) / P_PT) * inv + 0.5f);
            }
            lv = P_LEAF * best + (1.f - P_LEAF) * (wsum / sum);
        }
        tryProve(idx);
        v = L.proven ? provenValueM(L.proven, s.side ^ 1) : 1.f - lv;
        proof = L.proven && depth > 1;
    } else if (L.proven || s.result) {
        if (!L.proven) {
            const int mover = s.side ^ 1;
            L.proven = s.result == DRAW ? PR_DRAW : (s.result == WIN0 + mover ? PR_WIN : PR_LOSS);
        }
        v = provenValueM(L.proven, s.side ^ 1);
        proof = depth > 1;
    } else {  // pool full: score the leaf itself
        v = 1.f - nnEval(*Ap, s);
    }
    for (int i = depth - 1; i >= 0; i--) {
        Node& nd = P[pathBuf[i]];
        nd.N++;
        nd.W += v;
        v = 1.f - v;
        if (proof && i < depth - 1) proof = tryProve(pathBuf[i]);
    }
}
#endif

static inline void searchStep() {
#if NN_H
    if (netOn) { iterateNN(); return; }
#endif
    iterate();
}
static inline void turnStart(int us) {
#if NN_H
    rootSide = us;
    drawK[rootSide] = P_DE * ((rootSide == 0 ? P_DX : P_DO) - 0.5f);
    drawK[rootSide ^ 1] = -drawK[rootSide];
    useDraw = hasDraw && drawK[0] != 0.f;
    if (netOn) accRefresh(rootAcc, rootState);
#endif
}

// Copy the subtree under `root` into the other pool (breadth first, children stay contiguous).
static void compact() {
    Node* S = pool[cur];
    Node* D = pool[cur ^ 1];
    // BFS copy (the queue is implicit in the copy order). At most P_CLIM nodes are kept so the
    // copy stays well inside a turn; deeper nodes keep their statistics but lose their children
    // and are expanded again when the search reaches them.
    D[0] = S[root];
    int n = 1;
    for (int h = 0; h < n; h++) {
        Node& d = D[h];
        if (d.first >= 0) {
            if (n + d.n > P_CLIM && !d.proven) { d.first = -1; continue; }
            const int f = d.first;
            memcpy(D + n, S + f, sizeof(Node) * d.n);
            d.first = n;
            n += d.n;
        }
    }
    cur ^= 1;
    used = n;
    treeGen++;
    root = 0;
}

// Move the root down by move m (keeps the subtree when it exists).
static void advance(int m) {
    Node* P = pool[cur];
    const Node& nd = P[root];
    int nr = -1;
    if (nd.first >= 0)
        for (int i = 0; i < nd.n; i++)
            if (P[nd.first + i].move == m) { nr = nd.first + i; break; }
    rootState.play(m);
    if (nr < 0) root = newRoot();
    else root = nr;
}

static int bestMove() {
    Node* P = pool[cur];
    const Node& nd = P[root];
    int best = -1;
    float bs = -1e30f;
    for (int i = 0; i < nd.n; i++) {
        const Node& x = P[nd.first + i];
        float s;
        if (x.proven == PR_WIN) s = 1e20f;
        else if (x.proven == PR_LOSS) s = -1e10f + x.N;
        else if (P_FK > 0.f) s = (x.N ? x.W / x.N : 0.f) + P_FK * logf((float)x.N + 1.f);
        else s = (float)x.N + (x.N ? x.W / x.N : 0.f) * 0.5f;
        if (s > bs) { bs = s; best = i; }
    }
    return P[nd.first + best].move;
}

// ------------------------------------------------------------------------------------------
// Driver
// ------------------------------------------------------------------------------------------
typedef chrono::steady_clock Clock;
static double msSince(Clock::time_point t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}

static long long fixedIters = 0, fixedFirstIters = 0;

static void parseParams(const char* p) {
    if (!p) return;
    string s(p);
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == string::npos) j = s.size();
        string kv = s.substr(i, j - i);
        size_t e = kv.find('=');
        if (e != string::npos) {
            string k = kv.substr(0, e);
            float v = (float)atof(kv.c_str() + e + 1);
            if (k == "C") P_C = v;
            else if (k == "ROLL") P_ROLL = (int)v;
            else if (k == "FPU") P_FPU = v;
            else if (k == "CN") P_CN = v;
            else if (k == "NNMODE") P_NNMODE = (int)v;
            else if (k == "NNSEL") P_NNSEL = (int)v;
            else if (k == "CP") P_CP = v;
            else if (k == "PT") P_PT = v;
            else if (k == "VN") P_VN = (int)v;
            else if (k == "CLIM") P_CLIM = (int)v;
            else if (k == "CT") P_CT = (int)v;
            else if (k == "FK") P_FK = v;
            else if (k == "DX") P_DX = v;
            else if (k == "DO") P_DO = v;
            else if (k == "DE") P_DE = v;
            else if (k == "PONDER") P_PONDER = (int)v;
#ifdef LOCAL
            else if (k == "CHECK") checkAcc = v != 0;
#endif
            else if (k == "LEAF") P_LEAF = v;
            else if (k == "MS") P_MS = v;
            else if (k == "FMS") P_FIRST_MS = v;
            else fprintf(stderr, "unknown param %s\n", k.c_str());
        }
        i = j + 1;
    }
}

#ifdef LOCAL
static u64 perft(const State& s, int d) {
    if (d == 0) return 1;
    u8 mv[81];
    int n = s.gen(mv);
    if (d == 1) return (u64)n;
    u64 t = 0;
    for (int i = 0; i < n; i++) { State c = s; c.play(mv[i]); t += perft(c, d - 1); }
    return t;
}
#endif

#ifdef LOCAL
// Self-play data: UT_GEN="games:iters:seed:outfile". Each game starts with 0..6 random plies,
// samples moves by visit count for the next 10 plies, then plays the most visited move.
// Record (44 bytes, side-to-move frame): own/opp cells per small board, root value, next
// board, result for the side to move (0 loss, 1 draw, 2 win), ply, best move, flags (1 = root proven).
#pragma pack(push, 1)
struct GenRec { u16 own[9], opp[9]; int16_t q; int8_t next; u8 result, ply, best, flags, side; };
#pragma pack(pop)
static_assert(sizeof(GenRec) == 44, "record size");

static int runGen(const char* spec) {
    long games = 0, iters = 0;
    unsigned long long seed = 0;
    char out[512];
    if (sscanf(spec, "%ld:%ld:%llu:%511[^:]", &games, &iters, &seed, out) != 4) { fprintf(stderr, "bad UT_GEN\n"); return 1; }
    // optional ":fastIters:pFull": playout cap randomization (only fully searched moves are recorded)
    long fastIters = iters;
    float pFull = 1.f;
    {
        int colons = 0;
        for (const char* c = spec; *c; c++) colons += *c == ':';
        if (colons >= 5) {
            const char* c = spec;
            for (int k = 0; k < 4; k++) c = strchr(c, ':') + 1;
            sscanf(c, "%ld:%f", &fastIters, &pFull);
        }
    }
    rng = Rng(seed * 0x9E3779B97F4A7C15ULL + 12345);
    P_DX = P_DO = 0.5f;  // self-play: both sides use the same tree, so no side-dependent draw value
    POOL = 1 << 21;
    pool[0] = (Node*)malloc(sizeof(Node) * POOL);
    pool[1] = (Node*)malloc(sizeof(Node) * POOL);
    FILE* f = fopen(out, "ab");
    if (!f) return 1;
    long positions = 0;
    auto t0 = Clock::now();
    for (long g = 0; g < games; g++) {
        rootState.reset();
        root = newRoot();
        const int open = (int)rng.below(7);
        u8 mv[81];
        for (int k = 0; k < open && !rootState.result; k++) {
            int n = rootState.gen(mv);
            advance(mv[rng.below((u32)n)]);
        }
        if (rootState.result) continue;
        vector<GenRec> recs;
        const int sampleUntil = rootState.ply + 10;
        while (!rootState.result) {
            if (used > POOL / 2) compact();
            turnStart(rootState.side);
            const bool full = pFull >= 1.f || rng.below(1 << 20) < (u32)(pFull * (1 << 20));
            const long budget = full ? iters : fastIters;
            for (long it = 0; it < budget && !pool[cur][root].proven; it++) searchStep();
            if (pool[cur][root].first < 0) expand(root, rootState);
            const Node& rn = pool[cur][root];
            GenRec r;
            memset(&r, 0, sizeof r);
            const int p = rootState.side;
            for (int i = 0; i < 9; i++) { r.own[i] = rootState.b[p][i]; r.opp[i] = rootState.b[p ^ 1][i]; }
            float q = rn.N ? 1.f - rn.W / rn.N : 0.5f;
            if (rn.proven) q = 1.f - provenValue(rn.proven);
            r.q = (int16_t)lrintf(q * 20000.f - 10000.f);
            r.next = rootState.next;
            r.ply = rootState.ply;
            r.side = (u8)p;
            r.flags = rn.proven ? 1 : 0;
            int m = bestMove();
            r.best = (u8)m;
            if (full) recs.push_back(r);
            if (rootState.ply < sampleUntil && !rn.proven) {  // sample proportionally to visits
                Node* P = pool[cur];
                long tot = 0;
                for (int i = 0; i < rn.n; i++) if (P[rn.first + i].proven != PR_LOSS) tot += P[rn.first + i].N;
                if (tot > 0) {
                    long k = (long)(((u64)rng.next() * (u64)tot) >> 32);
                    for (int i = 0; i < rn.n; i++) {
                        const Node& x = P[rn.first + i];
                        if (x.proven == PR_LOSS) continue;
                        if (k < x.N) { m = x.move; break; }
                        k -= x.N;
                    }
                }
            }
            advance(m);
        }
        const int res = rootState.result;
        for (auto& r : recs) r.result = res == DRAW ? 1 : ((res == WIN0 + r.side) ? 2 : 0);
        fwrite(recs.data(), sizeof(GenRec), recs.size(), f);
        fflush(f);
        positions += (long)recs.size();
        if ((g + 1) % 50 == 0)
            fprintf(stderr, "gen %ld games %ld positions %.0f/s\n", g + 1, positions, positions / (msSince(t0) / 1000.0));
    }
    fclose(f);
    return 0;
}
#endif

int main() {
    initTables();
#ifdef NN_EMBED
    loadEmbedded();
    netOn = true;
#endif
    finishNet();
    for (int i = 1; i < (1 << 16); i++) LOGT[i] = logf((float)i);
    LOGT[0] = 0;
    rootState.reset();
#ifdef LOCAL
    parseParams(getenv("UT_PARAMS"));
    if (getenv("UT_ITERS")) fixedIters = atoll(getenv("UT_ITERS"));
    fixedFirstIters = getenv("UT_FIRST_ITERS") ? atoll(getenv("UT_FIRST_ITERS")) : fixedIters * 10;
    if (getenv("UT_MS")) P_MS = (float)atof(getenv("UT_MS")) - 5.f;
    if (getenv("UT_FIRST_MS")) P_FIRST_MS = (float)atof(getenv("UT_FIRST_MS")) - 40.f;
    if (getenv("UT_SEED")) rng = Rng(strtoull(getenv("UT_SEED"), 0, 10));
    else rng = Rng((u64)chrono::high_resolution_clock::now().time_since_epoch().count());
    if (getenv("UT_PERFT")) {
        int D = atoi(getenv("UT_PERFT"));
        for (int d = 1; d <= D; d++) {
            auto t0 = Clock::now();
            u64 r = perft(rootState, d);
            printf("perft %d %llu %.0fms\n", d, (unsigned long long)r, msSince(t0));
        }
        return 0;
    }
#if NN_H
    if (getenv("UT_NET")) {
        netOn = loadNet(getenv("UT_NET"));
        finishNet();
        if (!netOn) { fprintf(stderr, "cannot load net %s\n", getenv("UT_NET")); return 1; }
    }
#endif
#if NN_H
    if (getenv("UT_EVAL")) {  // print the net's value of the first records of a data file
        FILE* f = fopen(getenv("UT_EVAL"), "rb");
        GenRec r;
        for (int k = 0; k < (getenv("UT_EVALN") ? atoi(getenv("UT_EVALN")) : 20) && f && fread(&r, sizeof r, 1, f) == 1; k++) {
            State s;
            s.reset();
            s.side = r.side;
            s.next = r.next;
            for (int bd = 0; bd < 9; bd++) {
                s.b[r.side][bd] = r.own[bd];
                s.b[r.side ^ 1][bd] = r.opp[bd];
                for (int q = 0; q < 2; q++) if (WIN[s.b[q][bd]]) { s.macro[q] |= (u16)(1u << bd); s.closed |= (u16)(1u << bd); }
                if ((r.own[bd] | r.opp[bd]) == 0x1FF) s.closed |= (u16)(1u << bd);
            }
            Acc A;
            accRefresh(A, s);
            printf("%.5f\n", nnEval(A, s));
        }
        return 0;
    }
#endif
    if (getenv("UT_GEN")) return runGen(getenv("UT_GEN"));
    if (getenv("UT_BENCH")) {
        auto t0 = Clock::now();
        long long n = 0, plies = 0;
        State s0;
        s0.reset();
        while (msSince(t0) < 2000) {
            for (int i = 0; i < 1000; i++) { State s = s0; rollout(s, rng); n++; }
        }
        (void)plies;
        printf("ROLL=%d rollouts/s %.0f\n", P_ROLL, n / (msSince(t0) / 1000.0));
        return 0;
    }
    if (getenv("UT_POOL")) POOL = atoi(getenv("UT_POOL"));  // small pools exercise compaction
    if (getenv("UT_CHECKACC")) checkAcc = true;
#endif
    pool[0] = (Node*)malloc(sizeof(Node) * POOL);
    pool[1] = (Node*)malloc(sizeof(Node) * POOL);
    root = newRoot();
#ifdef LOCAL
    if (const char* op = getenv("UT_OPENING")) {  // plies played before the game, "r c r c ..."
        int r, c, k;
        const char* p = op;
        while (sscanf(p, "%d %d%n", &r, &c, &k) == 2) { advance(GRID2M[r * 9 + c]); p += k; }
    }
#endif
#ifdef LOCAL
    if (const char* an = getenv("UT_ANALYZE")) {  // "r c r c ...": play the moves, search, print the root
        int r, c, k;
        const char* p = an;
        while (sscanf(p, "%d %d%n", &r, &c, &k) == 2) { advance(GRID2M[r * 9 + c]); p += k; }
        turnStart(rootState.side);
        const long long n = fixedFirstIters > 0 ? fixedFirstIters : 1000000;
        for (long long i = 0; i < n && !pool[cur][root].proven; i++) searchStep();
        const Node& rn = pool[cur][root];
        vector<pair<int, int>> ch;
        for (int i = 0; i < rn.n; i++) ch.push_back({pool[cur][rn.first + i].N, rn.first + i});
        sort(ch.rbegin(), ch.rend());
        printf("ply %d side %d iters %lld root q %.3f proven %d best %d %d\n", rootState.ply, rootState.side, n,
               rn.N ? 1.f - rn.W / rn.N : 0.f, rn.proven, MROW[bestMove()], MCOL[bestMove()]);
        for (size_t i = 0; i < ch.size() && i < 5; i++) {
            const Node& x = pool[cur][ch[i].second];
            printf("  %d %d  N %d  q %.3f  prior %.2f  proven %d\n", MROW[x.move], MCOL[x.move], x.N,
                   x.N ? x.W / x.N : 0.f, x.pad / 255.f, x.proven);
        }
        return 0;
    }
#endif
    bool first = true;
    // Second player: the opponent's first turn can take a second. Search the start position
    // until its move arrives. (Nothing within 50 ms: we are the second player.)
    if (P_PONDER && fixedIters == 0) {
        struct pollfd pf = {0, POLLIN, 0};
        if (poll(&pf, 1, 50) == 0) {
            const auto tp = Clock::now();
            turnStart(rootState.side ^ 1);
            long long pit = 0;
            while (poll(&pf, 1, 0) == 0 && used < POOL / 2 && msSince(tp) < 3000) {
                for (int k = 0; k < 32; k++) searchStep();
                pit += 32;
            }
            fprintf(stderr, "pregame ponder it=%lld t=%.1fms\n", pit, msSince(tp));
        }
    }
    for (int turn = 0;; turn++) {
        int orow, ocol;
        if (scanf("%d %d", &orow, &ocol) != 2) return 0;
        const auto t0 = Clock::now();
        int cnt;
        if (scanf("%d", &cnt) != 1) return 0;
        u64 valid[2] = {0, 0};
        int firstValid = -1;
        for (int i = 0; i < cnt; i++) {
            int r, c;
            if (scanf("%d %d", &r, &c) != 2) return 0;
            const int m = GRID2M[r * 9 + c];
            valid[m >> 6] |= 1ULL << (m & 63);
            if (firstValid < 0) firstValid = m;
        }
        if (orow >= 0) advance(GRID2M[orow * 9 + ocol]);
        // Exactness check: our legal moves must be the referee's.
        {
            u8 mv[81];
            int n = rootState.gen(mv);
            u64 ours[2] = {0, 0};
            for (int i = 0; i < n; i++) ours[mv[i] >> 6] |= 1ULL << (mv[i] & 63);
            if (ours[0] != valid[0] || ours[1] != valid[1])
            {
                fprintf(stderr, "DESYNC turn %d: ours %d theirs %d\n", turn, n, cnt);
#ifdef LOCAL
                exit(3);
#endif
            }
        }
        // Fault in the spare pool during the long first turn: compaction on a later turn would
        // otherwise pay for first-touch page faults inside a 100 ms budget.
        if (first) memset((void*)pool[cur ^ 1], 0, sizeof(Node) * POOL);
        if (used > POOL / 2) compact();
        turnStart(rootState.side);
        const double budget = first ? P_FIRST_MS : P_MS;
        const long long iters = first ? fixedFirstIters : fixedIters;
        long long it = 0;
        if (iters > 0) {
            for (; it < iters && !pool[cur][root].proven; it++) searchStep();
        } else {
            while (!pool[cur][root].proven) {
                for (int k = 0; k < 16; k++) searchStep();
                it += 16;
                if (msSince(t0) >= budget) break;
            }
        }
        if (pool[cur][root].first < 0) expand(root, rootState);
        int m = bestMove();
        if (!(valid[m >> 6] >> (m & 63) & 1)) {
            fprintf(stderr, "ILLEGAL choice %d, falling back\n", m);
            m = firstValid;
        }
        const Node& rn = pool[cur][root];
        printf("%d %d\n", MROW[m], MCOL[m]);
        fflush(stdout);
#ifdef LOCAL
        if (checkAcc) fprintf(stderr, "accumulator checks passed: %lld\n", checkCount);
#endif
        fprintf(stderr, "it=%lld root N=%d q=%.3f pr=%d used=%d t=%.1fms\n", it, rn.N,
                rn.N ? 1.f - rn.W / rn.N : 0.f, rn.proven, used, msSince(t0));
        advance(m);
        first = false;
        // Ponder: keep searching (from the opponent's side of the tree) until their move arrives.
        if (P_PONDER && iters == 0 && !rootState.result && !pool[cur][root].proven) {
            if (used > POOL / 2) compact();
            turnStart(rootState.side ^ 1);
            const auto tp = Clock::now();
            long long pit = 0;
            struct pollfd pf = {0, POLLIN, 0};
            while (!pool[cur][root].proven && used < POOL - (1 << 16) && msSince(tp) < 3000) {
                for (int k = 0; k < 32; k++) searchStep();  // poll() is a syscall: not too often
                pit += 32;
                if (poll(&pf, 1, 0) > 0) break;
            }
            fprintf(stderr, "ponder it=%lld t=%.1fms\n", pit, msSince(tp));
        }
    }
}
