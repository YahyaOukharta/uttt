// Ultimate Tic-Tac-Toe (CodinGame "tic-tac-toe") - Monte Carlo tree search.
//
// Board: 9 small boards of 9 cells. A move is m = board * 9 + cell (cells row-major).
// The side to move plays in the small board named by the previous move's cell, unless that
// board is closed (won or full): then anywhere. Won boards are closed. Three won boards in a
// line win; if every board is closed without a line, more won boards wins (else a draw).
#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")
#if defined(__x86_64__) || defined(__i386__)
#pragma GCC target("avx2,bmi,bmi2,popcnt,lzcnt")
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
using namespace std;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// ------------------------------------------------------------------------------------------
// Parameters (overridable locally through UT_PARAMS="C=0.6,...")
// ------------------------------------------------------------------------------------------
static float P_C = 0.70f;         // UCT exploration constant (values in [0,1])
static float P_MS = 95.0f;        // per-turn budget, ms
static float P_FIRST_MS = 960.0f; // first-turn budget, ms

// ------------------------------------------------------------------------------------------
// Tables
// ------------------------------------------------------------------------------------------
static bool WIN[512];
static u8 POP[512];
static u8 NTH[512][9];  // NTH[mask][k] = index of the k-th set bit
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

// Uniformly random playout to the end. Returns the result.
static inline int rollout(State s, Rng& rng) {
    while (!s.result) {
        int bd = s.next, c;
        if (bd >= 0) {
            const u16 e = s.empty(bd);
            c = NTH[e][rng.below(POP[e])];
        } else {
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

static const int POOL = 1 << 23;
static Node* pool[2] = {nullptr, nullptr};
static int cur = 0;      // active pool
static int used = 0;     // nodes used in the active pool
static int root = 0;
static State rootState;
static Rng rng(0x1234567ULL);
static float LOGT[1 << 16];

static inline float fastlog(int n) { return n < (1 << 16) ? LOGT[n] : logf((float)n); }

static inline int newRoot() {
    Node* P = pool[cur];
    used = 0;
    P[0] = Node{-1, 0, 0.f, 0, 0, PR_UNK, 0};
    used = 1;
    return 0;
}

static inline bool expand(int idx, const State& s) {
    u8 mv[81];
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
    const float c = P_C * sqrtf(fastlog(nd.N));
    int best = -1;
    float bv = -1e30f;
    for (int i = 0; i < nd.n; i++) {
        const int ch = nd.first + i;
        const Node& x = P[ch];
        if (x.N == 0) return ch;
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

// Copy the subtree under `root` into the other pool (breadth first, children stay contiguous).
static void compact() {
    Node* S = pool[cur];
    Node* D = pool[cur ^ 1];
    vector<int> q;
    q.reserve(used);
    D[0] = S[root];
    q.push_back(root);
    int n = 1;
    for (size_t h = 0; h < q.size(); h++) {
        Node& d = D[h];
        if (d.first >= 0) {
            const int f = d.first;
            for (int i = 0; i < d.n; i++) { D[n + i] = S[f + i]; q.push_back(f + i); }
            d.first = n;
            n += d.n;
        }
    }
    cur ^= 1;
    used = n;
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

int main() {
    initTables();
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
    if (getenv("UT_BENCH")) {
        auto t0 = Clock::now();
        long long n = 0, plies = 0;
        State s0;
        s0.reset();
        while (msSince(t0) < 2000) {
            for (int i = 0; i < 1000; i++) { State s = s0; rollout(s, rng); n++; }
        }
        (void)plies;
        printf("rollouts/s %.0f\n", n / (msSince(t0) / 1000.0));
        return 0;
    }
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
    bool first = true;
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
        if (used > POOL / 2) compact();
        const double budget = first ? P_FIRST_MS : P_MS;
        const long long iters = first ? fixedFirstIters : fixedIters;
        long long it = 0;
        if (iters > 0) {
            for (; it < iters && !pool[cur][root].proven; it++) iterate();
        } else {
            while (!pool[cur][root].proven) {
                for (int k = 0; k < 16; k++) iterate();
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
        fprintf(stderr, "it=%lld root N=%d q=%.3f pr=%d used=%d t=%.1fms\n", it, rn.N,
                rn.N ? 1.f - rn.W / rn.N : 0.f, rn.proven, used, msSince(t0));
        advance(m);
        first = false;
    }
}
