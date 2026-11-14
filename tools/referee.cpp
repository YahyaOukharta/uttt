// Native referee for CodinGame Ultimate Tic-Tac-Toe. Written independently of bot.cpp (plain
// arrays, grid coordinates) so the two implementations check each other (perft, games).
//
// Usage: referee BOT_A BOT_B [-g games] [-j threads] [-t ms] [-f first_ms] [-o open_plies]
//                [-s seed] [-n iters] [-nb iters_b] [-pa params] [-pb params] [-x grace_ms] [-v]
//        referee --perft D
// Games come in pairs: each random opening is played twice with seats swapped. The opening is
// handed to both bots through UT_OPENING ("r c r c ..."); the bot to move first after it gets
// "-1 -1" as the opponent's move. -n gives each bot a fixed number of iterations per turn
// (UT_ITERS); otherwise -t ms per turn (UT_MS) and -f for the first turn (UT_FIRST_MS).
// Prints one JSON line per game with -v, then a JSON summary line.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace std;

struct Game {
    int cell[9][9];   // [row][col]: 0 empty, 1 first player, 2 second player
    int small[3][3];  // 0 open, 1/2 won by player, 3 full without winner
    int lastR = -1, lastC = -1;
    int turn = 0;     // plies played; player to move = turn % 2 (0 = first player)
    int winner = -1;  // -1 ongoing, 0/1 player, 2 draw
    Game() { memset(cell, 0, sizeof cell); memset(small, 0, sizeof small); }

    static bool line3(int a, int b, int c, int p) { return a == p && b == p && c == p; }
    static bool wins(int g[3][3], int p) {
        for (int i = 0; i < 3; i++)
            if (line3(g[i][0], g[i][1], g[i][2], p) || line3(g[0][i], g[1][i], g[2][i], p)) return true;
        return line3(g[0][0], g[1][1], g[2][2], p) || line3(g[0][2], g[1][1], g[2][0], p);
    }
    vector<pair<int, int>> moves() const {
        vector<pair<int, int>> out;
        if (winner >= 0) return out;
        int br = -1, bc = -1;
        if (lastR >= 0) { br = lastR % 3; bc = lastC % 3; if (small[br][bc]) br = bc = -1; }
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++) {
                if (cell[r][c] || small[r / 3][c / 3]) continue;
                if (br >= 0 && (r / 3 != br || c / 3 != bc)) continue;
                out.push_back({r, c});
            }
        return out;
    }
    void play(int r, int c) {
        int p = turn % 2 + 1;
        cell[r][c] = p;
        int sr = r / 3, sc = c / 3;
        int g[3][3];
        bool full = true;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) { g[i][j] = cell[sr * 3 + i][sc * 3 + j]; if (!g[i][j]) full = false; }
        if (wins(g, p)) small[sr][sc] = p;
        else if (full) small[sr][sc] = 3;
        lastR = r; lastC = c;
        turn++;
        if (wins(small, p)) { winner = p - 1; return; }
        bool any = false;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (!small[i][j]) any = true;
        if (!any) {
            int n1 = 0, n2 = 0;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { n1 += small[i][j] == 1; n2 += small[i][j] == 2; }
            winner = n1 > n2 ? 0 : (n2 > n1 ? 1 : 2);
        }
    }
};

static unsigned long long perft(const Game& g, int d) {
    auto mv = g.moves();
    if (d == 1) return mv.size();
    unsigned long long t = 0;
    for (auto& m : mv) { Game c = g; c.play(m.first, m.second); t += perft(c, d - 1); }
    return t;
}

static mutex forkMutex, outMutex;

struct Bot {
    pid_t pid = -1;
    int in = -1, out = -1;
    string buf;
    bool start(const string& path, const string& opening, int ms, int firstMs, long long iters, const string& params) {
        lock_guard<mutex> lk(forkMutex);
        int a[2], b[2];
        if (pipe(a) || pipe(b)) return false;
        pid = fork();
        if (pid < 0) return false;
        if (pid == 0) {
            dup2(a[0], 0);
            dup2(b[1], 1);
            int dn = open("/dev/null", O_WRONLY);
            if (dn >= 0) dup2(dn, 2);
            for (int fd = 3; fd < 256; fd++) close(fd);
            setenv("UT_OPENING", opening.c_str(), 1);
            setenv("UT_MS", to_string(ms).c_str(), 1);
            setenv("UT_FIRST_MS", to_string(firstMs).c_str(), 1);
            if (iters > 0) setenv("UT_ITERS", to_string(iters).c_str(), 1); else unsetenv("UT_ITERS");
            if (!params.empty()) setenv("UT_PARAMS", params.c_str(), 1); else unsetenv("UT_PARAMS");
            execl(path.c_str(), path.c_str(), (char*)0);
            _exit(127);
        }
        close(a[0]);
        close(b[1]);
        in = a[1];
        out = b[0];
        fcntl(in, F_SETFD, FD_CLOEXEC);
        fcntl(out, F_SETFD, FD_CLOEXEC);
        return true;
    }
    bool send(const string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t w = write(in, s.data() + off, s.size() - off);
            if (w <= 0) return false;
            off += (size_t)w;
        }
        return true;
    }
    // 1 = line read, 0 = timeout, -1 = closed/crashed
    int readLine(string& line, double timeoutMs) {
        auto end = chrono::steady_clock::now() + chrono::microseconds((long long)(timeoutMs * 1000));
        for (;;) {
            size_t p = buf.find('\n');
            if (p != string::npos) { line = buf.substr(0, p); buf.erase(0, p + 1); return 1; }
            auto now = chrono::steady_clock::now();
            if (now >= end) return 0;
            int left = (int)chrono::duration_cast<chrono::milliseconds>(end - now).count() + 1;
            pollfd pf{out, POLLIN, 0};
            int pr = poll(&pf, 1, left);
            if (pr == 0) return 0;
            if (pr < 0) continue;
            char tmp[512];
            ssize_t r = read(out, tmp, sizeof tmp);
            if (r <= 0) return -1;
            buf.append(tmp, (size_t)r);
        }
    }
    void stop() {
        if (in >= 0) close(in);
        if (out >= 0) close(out);
        if (pid > 0) { kill(pid, SIGKILL); waitpid(pid, 0, 0); }
        in = out = -1;
        pid = -1;
    }
};

// winner: 0 = bot A, 1 = bot B, 2 = draw
struct Result { int winner; int plies; string reason; double maxMs[2]; };

static string botParams[2];
static long long itersB = -1;

// first = index (0 = A, 1 = B) of the bot that plays first (the "X" seat).
static Result playGame(const string paths[2], int first, const string& opening, int ms, int firstMs,
                       long long iters, double grace) {
    Game g;
    {
        int r, c, k;
        const char* p = opening.c_str();
        while (sscanf(p, "%d %d%n", &r, &c, &k) == 2) { g.play(r, c); p += k; }
    }
    int openPlies = g.turn;
    Bot bots[2];
    Result res{-1, 0, "", {0, 0}};
    for (int k = 0; k < 2; k++)
        if (!bots[k].start(paths[k], opening, ms, firstMs, (k == 1 && itersB >= 0) ? itersB : iters, botParams[k])) {
            res.winner = 1 - k; res.reason = "spawn";
            bots[0].stop(); bots[1].stop();
            return res;
        }
    bool firstTurn[2] = {true, true};
    int lr = -1, lc = -1;
    while (g.winner < 0) {
        int seat = g.turn % 2;
        int who = seat == 0 ? first : 1 - first;
        auto legal = g.moves();
        string msg = to_string(g.turn == openPlies ? -1 : lr) + " " + to_string(g.turn == openPlies ? -1 : lc) + "\n" +
                     to_string(legal.size()) + "\n";
        for (auto& m : legal) msg += to_string(m.first) + " " + to_string(m.second) + "\n";
        double limit = (firstTurn[who] ? firstMs : ms) + grace;
        if (iters > 0) limit = 60000;
        firstTurn[who] = false;
        auto t0 = chrono::steady_clock::now();
        string line;
        int rc = bots[who].send(msg) ? bots[who].readLine(line, limit) : -1;
        double el = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
        res.maxMs[who] = max(res.maxMs[who], el);
        if (rc != 1) {
            res.winner = 1 - who;
            res.reason = rc == 0 ? "timeout" : "crash";
            break;
        }
        int r = -9, c = -9;
        sscanf(line.c_str(), "%d %d", &r, &c);
        if (find(legal.begin(), legal.end(), make_pair(r, c)) == legal.end()) {
            res.winner = 1 - who;
            res.reason = "illegal:" + line.substr(0, 20);
            break;
        }
        g.play(r, c);
        lr = r; lc = c;
        if (g.winner >= 0) {
            res.winner = g.winner == 2 ? 2 : (g.winner == 0 ? first : 1 - first);
            res.reason = "end";
        }
    }
    res.plies = g.turn;
    bots[0].stop();
    bots[1].stop();
    return res;
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    if (argc >= 3 && string(argv[1]) == "--perft") {
        int D = atoi(argv[2]);
        Game g;
        for (int d = 1; d <= D; d++) {
            auto t0 = chrono::steady_clock::now();
            unsigned long long r = perft(g, d);
            printf("perft %d %llu %.0fms\n", d, r, chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count());
            fflush(stdout);
        }
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: referee A B [-g games] [-j threads] [-t ms] [-f first_ms] [-o open_plies] [-s seed] [-n iters] [-x grace_ms] [-v]\n");
        return 2;
    }
    string paths[2] = {argv[1], argv[2]};
    int games = 100, threads = 4, ms = 100, firstMs = -1, openPlies = 2;
    long long iters = 0;
    unsigned seed = (unsigned)chrono::system_clock::now().time_since_epoch().count();
    double grace = 15;
    bool verbose = false;
    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : (char*)"0"; };
        if (a == "-g") games = atoi(next());
        else if (a == "-j") threads = atoi(next());
        else if (a == "-t") ms = atoi(next());
        else if (a == "-f") firstMs = atoi(next());
        else if (a == "-o") openPlies = atoi(next());
        else if (a == "-s") seed = (unsigned)strtoul(next(), 0, 10);
        else if (a == "-n") iters = atoll(next());
        else if (a == "-x") grace = atof(next());
        else if (a == "-v") verbose = true;
        else if (a == "-pa") botParams[0] = next();
        else if (a == "-nb") itersB = atoll(next());
        else if (a == "-pb") botParams[1] = next();
    }
    if (firstMs < 0) firstMs = ms * 10;

    mt19937 rng(seed);
    vector<string> openings((games + 1) / 2);
    for (auto& o : openings) {
        Game g;
        for (int p = 0; p < openPlies; p++) {
            auto l = g.moves();
            auto m = l[rng() % l.size()];
            if (!o.empty()) o += " ";
            o += to_string(m.first) + " " + to_string(m.second);
            g.play(m.first, m.second);
        }
    }

    atomic<int> nextGame(0);
    int wins[2] = {0, 0}, draws = 0, firstWins = 0, faults[2] = {0, 0}, done = 0;
    double maxMs[2] = {0, 0};
    long long totalPlies = 0;
    auto worker = [&]() {
        for (;;) {
            int gi = nextGame.fetch_add(1);
            if (gi >= games) return;
            int first = gi % 2;
            const string& op = openings[gi / 2];
            Result r = playGame(paths, first, op, ms, firstMs, iters, grace);
            lock_guard<mutex> lk(outMutex);
            if (r.winner == 2) draws++;
            else if (r.winner >= 0) {
                wins[r.winner]++;
                if (r.winner == first) firstWins++;
                if (r.reason != "end") faults[1 - r.winner]++;
            }
            done++;
            totalPlies += r.plies;
            for (int k = 0; k < 2; k++) maxMs[k] = max(maxMs[k], r.maxMs[k]);
            if (verbose) {
                printf("{\"game\":%d,\"first\":\"%s\",\"winner\":\"%s\",\"plies\":%d,\"reason\":\"%s\",\"opening\":\"%s\"}\n",
                       gi, first == 0 ? "A" : "B", r.winner == 0 ? "A" : (r.winner == 1 ? "B" : "-"), r.plies,
                       r.reason.c_str(), op.c_str());
                fflush(stdout);
            }
        }
    };
    vector<thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    printf("{\"summary\":true,\"games\":%d,\"wins_a\":%d,\"wins_b\":%d,\"draws\":%d,\"first_wins\":%d,\"faults_a\":%d,\"faults_b\":%d,"
           "\"avg_plies\":%.1f,\"max_ms_a\":%.1f,\"max_ms_b\":%.1f,\"ms\":%d,\"first_ms\":%d,\"iters\":%lld,\"seed\":%u}\n",
           done, wins[0], wins[1], draws, firstWins, faults[0], faults[1], done ? (double)totalPlies / done : 0.0,
           maxMs[0], maxMs[1], ms, firstMs, iters, seed);
    return 0;
}
