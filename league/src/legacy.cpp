#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops",              \
    "unsafe-math-optimizations", "no-trapping-math", "tree-vectorize", "fast-math")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target(                                                            \
"sse,sse2,sse3,ssse3,sse4,mmx,avx,avx2,popcnt,rdrnd,abm,bmi2,fma")


#ifndef __COMMON_HPP__
#define __COMMON_HPP__
#include <iostream>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <cmath>
using namespace std;
#define popcnt __builtin_popcount
#define ctz __builtin_ctz
#define clz __builtin_clz
#define sqrtf __builtin_sqrtf
#include <chrono>
#include <cstdlib>
#include <cstdio>

using namespace std;
using namespace std::chrono;
// This is actually floor(log2(x)) / log2(e) which is not exactly ln(x) but
// good enough for our use case.
typedef uint8_t u8;

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#define fastln(x) ((f32)(31 - clz(x)) * 0.6931471805599453f)

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef size_t usize;

typedef float f32;
typedef double f64;

#define UCT_C 1.19999999999
// LUT
#define TABLE_SIZE (1 << 18)

typedef enum {
  PLAYING,
  X_WIN,
  O_WIN,
  TIE,
} result_t;

result_t RESULT_TABLE[TABLE_SIZE];

// State
typedef enum {
  PLAYER_X,
  PLAYER_O,
} player_t;

typedef struct {
  u32 boards[10];

  i8 last_move;

  player_t player;
  result_t result;

} state_t;

// Node

typedef struct {
  u8 grid;
  u8 cell;
} move_t;

typedef struct node_t node_t;

struct node_t {
  move_t move;

  usize children_count;
  vector<node_t> children;
  node_t *parent;

  i32 value;
  u32 samples;

  u8 last_sync;
};

#define NODES_PAGE_SIZE 100000
node_t *nodes = 0;
u64 node_count = 0;
u64 page_count = 0;
// Random

typedef u64 rng_t;

// MCTS
int turn = 0;
#define DECAY_FACTOR 1
float decay_pows[81]={};

void init_decay_pows(){
    for (int i =0;i<81;i++){
        decay_pows[i]=pow(DECAY_FACTOR,i);
    }
}
typedef struct {
  node_t *current_node;

  rng_t rng_state;
} mcts_t;

#endif
void print_board(const state_t *state);

void init_state(state_t *state) {
  for (int i = 0; i < 10; i++)
    state->boards[i] = 0;
  state->player = PLAYER_X;
  state->last_move = -1;
  state->result = PLAYING;
}

void state_replace(state_t *state, u8 grid, u8 cell, u32 new_board) {
  state->boards[grid] = new_board;
  if (RESULT_TABLE[new_board]) {
    state->boards[9] |= RESULT_TABLE[new_board] << (grid << 1);
    state->result = RESULT_TABLE[state->boards[9]];
      if (state->result == TIE)
{      int r = 0;
      u32 global_board = state->boards[9];
      for (int i = 0; i < 9; ++i) {
        u32 cell_value = (global_board & (0x03 << (i * 2))) >> (i * 2);
        if (cell_value == 1) {
          r++;
        } else if (cell_value == 2) {
          r--;
        }
      }
      // // cerr <<" tie " << r <<endl;
      r = r > 0 ? X_WIN : r < 0 ? O_WIN : TIE;
      state->result =(result_t)r;
      }
  }
  state->player = (player_t)(1 - state->player);
  state->last_move = (state->boards[9] & (3 << (cell << 1))) ? -1 : cell;
}
int a = 2;

void state_move(state_t *state, u8 grid, u8 cell) {
  // if (state->last_move != -1 && state->last_move != grid) {
  //   cerr << "wrong board" << endl;
  //   print_board(state);
  //   cerr << (int)grid << " " << (int)cell << endl;
  //   throw "exception()";
  // }

  u32 board = state->boards[grid];
  u32 new_board = board | (1 << (cell << 1 | state->player));
  // if (new_board == state->boards[grid]) {
  //   cerr << "same board" << endl;
  //   cerr << (int)grid << " " << (int)cell << endl;
  //   throw "exception()";
  // }

  state_replace(state, grid, cell, new_board);
}

#include <iostream>

void print_board(const state_t *state) {
  cerr << "Current player: " << (state->player == PLAYER_X ? 'X' : 'O') << endl;

  cerr << "    0   1   2   3   4   5   6   7   8" << endl;

  for (int i = 0; i < 9; i++) {
    cerr << "  ---------------------------------" << endl;
    cerr << i << " ";
    for (int j = 0; j < 9; j++) {
      int grid = i / 3;
      int cell = j / 3;
      int board_idx = grid * 3 + cell;
      int board_value =
          (state->boards[board_idx] >> (2 * (3 * (i % 3) + j % 3))) & 3;

      char cell_char;
      if (board_value == 0)
        cell_char = ' ';
      else if (board_value == 1)
        cell_char = 'X';
      else
        cell_char = 'O';

      cerr << "| " << cell_char << " ";
    }
    cerr << "|" << endl;
  }
  cerr << "  ---------------------------------" << endl;
  cerr << "last move " << (int)state->last_move << endl;
  cerr << "Result: ";
  switch (state->result) {
  case PLAYING:
    cerr << "Game still in progress." << endl;
    break;
  case X_WIN:
    cerr << "Player X wins!" << endl;
    break;
  case O_WIN:
    cerr << "Player O wins!" << endl;
    break;
  case TIE:
    cerr << "It's a tie!" << endl;
    break;
  }
}

void create_table() {
  const u8 WIN_TRIPLETS[][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6},
  };

  u32 win_mask_x[8] = {0};
  u32 win_mask_o[8] = {0};

  for (usize i = 0; i < 8; ++i) {
    for (const u8 *it = WIN_TRIPLETS[i]; it < WIN_TRIPLETS[i + 1]; ++it) {
      win_mask_x[i] |= 1 << (*it << 1);
      win_mask_o[i] |= 2 << (*it << 1);
    }
  }

  for (u32 board = 0; board < TABLE_SIZE; ++board) {
    // NOTE: Maybe LUT isn't necessary since this part is going to be vectorized
    // by the compiler and might be faster than array lookup
    for (usize i = 0; i < 8; ++i) {

      if ((board | win_mask_x[i]) == board && (board & win_mask_o[i]) == 0) {
        RESULT_TABLE[board] = X_WIN;
        break;
      }

      if ((board | win_mask_o[i]) == board && (board & win_mask_x[i]) == 0) {
        RESULT_TABLE[board] = O_WIN;
        break;
      }
    }

    if (RESULT_TABLE[board])
      continue;

    // Check if a move is avaialble
    RESULT_TABLE[board] = (~(board | (board >> 1))) & 0x15555u ? PLAYING : TIE;
  }
}

node_t new_node(u8 grid, u8 cell, node_t *parent) {
  node_t node;
  node.children = {};
  node.children_count = 0;
  node.parent = parent;
  node.move.grid = grid;
  node.move.cell = cell;

  node.samples = 0;
  node.value = 0;

  node .last_sync = turn;

  if (parent)
    parent->children.push_back(node);
  if (!node_count)
    nodes[node_count++] = node;
  return node;
}

node_t *nodes_head() { return nodes + node_count; }
void print_node(node_t *node) {
  cout << "Node: " << node << endl;
  cout << "Grid: " << (int)node->move.grid << endl;
  cout << "Cell: " << (int)node->move.cell << endl;
  cout << "Children count: " << node->children_count << endl;
  cout << "Parent: " << node->parent << endl;
  cout << "Value: " << node->value << endl;
  cout << "Samples: " << node->samples << endl;
}

#define MUL 6364136223846793005ULL
#define INC 1442695040888963407ULL

// 32-bit PCG with 64-bit state
// See: <https://www.pcg-random.org>
u32 rand_u32(rng_t *state) {
  rng_t oldstate = *state;
  *state = oldstate * MUL + INC;

  u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  u32 rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

rng_t rng_new(u64 seed) {
  rng_t state = 0;
  rand_u32(&state);

  state += seed;
  rand_u32(&state);

  return state;
}

void mcts_init(mcts_t *mcts) {
  node_count = 0;
  page_count = 0;
  if (nodes)
    free(nodes);
  nodes = (node_t *)malloc(sizeof(node_t) + 1);

  new_node(-1, -1, 0);
  mcts->current_node = nodes;

  mcts->rng_state = rng_new(time(0));
}

// Upper Confidence bounds applied to Trees (UCT)
f32 uct(node_t *node, f32 t) {

  return ((f32)node->value / node->samples) + UCT_C * sqrtf(t / node->samples);
}

node_t *best_child(node_t *node, int select = 1) {
  node_t *best_child = node->children.data();
  // if(best_child->last_sync != turn)
  // {
  //   best_child->samples *= DECAY_FACTOR;
  //   best_child->value *= DECAY_FACTOR;
  //   best_child->last_sync = turn;
  // }
  if (!best_child->samples || best_child->value == 1e8)
    return best_child; // no visits

  f32 t = fastln(node->samples);
  f32 best_uct = select ? uct(best_child, t) : (f32)best_child->value / best_child->samples;
  // if(!select)
  //   cerr <<" v "<<best_child->value << endl;

  for (usize i = 1; i < node->children.size(); ++i) {
    node_t *child = node->children.data() + i;
    // if(child->last_sync != turn)
    // {
    //   child->samples *= DECAY_FACTOR;
    //   child->value *= DECAY_FACTOR;
    //   child->last_sync = turn;
    // }
    // if(!select)
    //   cerr <<" v "<<child->value << endl;
    if (!child->samples || child->value == 1e8)
      return child;

    f32 child_uct = select ? uct(child, t) : (f32)child->value / child->samples;
    if (child_uct > best_uct) {
      best_uct = child_uct;
      best_child = child;
    }
  }

  return best_child;
}

void expand_board(node_t *node, u32 grid_idx, u32 board) {
  u32 mask = (~(board | (board >> 1))) & 0x15555u;

  while (mask) {
    u32 cell = mask & -mask;
    mask ^= cell;

    u32 cell_idx = ctz(cell) >> 1;

    node->children_count++;
    new_node(grid_idx, cell_idx, node);
  }
}

void expand_node(node_t *node, state_t *state) {
  node->children.reserve(state->last_move == -1 ? 81 : 9);
  node->children_count = 0;
  u32 grid_idx = state->last_move;

  if (grid_idx == -1) {
    u32 global_board = state->boards[9];
    u32 global_mask = (~(global_board | (global_board >> 1))) & 0x15555u;

    while (global_mask) {
      u32 grid = global_mask & -global_mask;
      global_mask ^= grid;

      grid_idx = ctz(grid) >> 1;
      expand_board(node, grid_idx, state->boards[grid_idx]);
    }
  } else
    expand_board(node, grid_idx, state->boards[grid_idx]);
}

// Picking a random empty cell
static u32 random_move_mask(u32 board, rng_t *rng) {
  // Create a bitmask representing all empty cells
  u32 mask = (~(board | (board >> 1))) & 0x15555u;

  // Remove a random amount of bits from the left
  u32 idx = rand_u32(rng) % popcnt(mask);
  for (usize i = 0; i < idx; ++i)
    mask &= (mask - 1);

  // Select the leftmost bit after removal
  return mask & -mask;
}

result_t playout(state_t *state, rng_t *rng, i32 *steps) {
  while (!state->result) {
    u32 grid = state->last_move;

    // Select a random local board when the player can move anywhere
    if (grid == -1) {
      grid = ctz(random_move_mask(state->boards[9], rng)) >> 1;
    }

    u32 board = state->boards[grid];
    u32 mask = random_move_mask(board, rng);

    board |= (mask << state->player);
    state_replace(state, grid, ctz(mask) >> 1, board);

    // *steps = *steps - 1;
  }

  return state->result;
}
void random_move(state_t *state, rng_t *rng) {
  u32 grid = state->last_move;

  // Select a random local board when the player can move anywhere
  if (grid == -1) {

    grid = ctz(random_move_mask(state->boards[9], rng)) >> 1;
  }

  u32 board = state->boards[grid];
  u32 mask = random_move_mask(board, rng);

  board |= (mask << state->player);
  state_replace(state, grid, ctz(mask) >> 1, board);
}
void mcts_search(mcts_t *mcts, state_t state, i32 *steps) {
  // selection

  while (mcts->current_node->children_count && !state.result && mcts->current_node->value != 1e8) {
    // cerr << "selection" << endl;
    mcts->current_node = best_child(mcts->current_node);
    // if (state.last_move != -1 && state.last_move != child->move.grid) {
    //   cerr << "children _count " << mcts->current_node->children_count << endl;
    //   cerr << "wtf ??" << endl;
    //   // cerr << child->move.grid << << child->move.cell
    // }
    state_move(&state, mcts->current_node->move.grid, mcts->current_node->move.cell);
    // *steps = *steps - 1;
  }

  // Expansion
  int solved = 0;
  if (mcts->current_node->samples && !state.result && mcts->current_node->value != 1e8) {
    // cerr << "expanding" << endl;
    if (!mcts->current_node->children_count)
      expand_node(mcts->current_node, &state);
    if (mcts->current_node->children_count) {
      mcts->current_node = mcts->current_node->children.data();
      // if (state.last_move != -1 && state.last_move != child->move.grid) {
      //   cerr << "children _count " << mcts->current_node->children_count
      //        << endl;
      //   cerr << "wtf ??" << endl;
      //   // cerr << child->move.grid << << child->move.cell
      // }
      state_move(&state, mcts->current_node->move.grid, mcts->current_node->move.cell);
      if(state.result){
        solved=1;
      }
    }
    // else if(state.result){
    //   solved=1;
    // }
    // *steps = *steps - 1;
  }
  else if(state.result || mcts->current_node->value == 1e8){
    solved=1;
  }

  // Simulation
  player_t current_player = (player_t)(1 - state.player);
  result_t result = state.result;
  if (!solved)
    result = playout(&state, &mcts->rng_state, steps);
  // else{
  //   //cerr <<"solved " <<endl; 
  // }

  int abort = 0;
  for (;;) {
    mcts->current_node->samples++;

    if (solved){
      if (result != TIE) {
        int factor = current_player == result - 1 ? 1 : -1;
        int allChildrenLose = 1;
          for (usize i = 0; i < mcts->current_node->children.size(); ++i) {
            node_t *c = mcts->current_node->children.data() + i;
            if (c->value != factor * 1e8 || !c->samples){
              allChildrenLose = 0;
              break;
            }
          }
          if (allChildrenLose){
            mcts->current_node->value= factor * 1e8;
          }
          else{
            if(abs(mcts->current_node->value) != 1e8)
              mcts->current_node->value += factor;
            solved = 0;
          }        
      }
    }
    else{
      if (result != TIE) {
        if(abs(mcts->current_node->value) < 1e8-500)
          mcts->current_node->value += current_player == result - 1 ? 1 : -1;
      }
    }

    current_player = (player_t)(1 - current_player);

    if (!mcts->current_node->parent)
      break;
    mcts->current_node = mcts->current_node->parent;
  }
}
int iters;
long long harness_iters = getenv("UT_ITERS") ? atoll(getenv("UT_ITERS")) : 0; // harness
void search(mcts_t *mcts, state_t *state, i32 max_us) {

  // Dynamic time allocation based on tree-traversal and simulation steps
  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  iters = 0;
  // cerr << "starting loop" << endl;
  mcts->current_node->parent = 0;
  for (;;) {
    // if (!(i % 10000)) {
    //   // cerr << "iter " << i << endl;
    //   // cerr << ((u64)tv2.tv_usec - (u64)tv1.tv_usec) << " " << max_us << endl;
    // }
    mcts_search(mcts, *state, &max_us);
    iters++;
    if (harness_iters > 0) { if (iters >= (turn == 0 ? harness_iters * 10 : harness_iters)) break; else continue; } // harness
    if(iters%251 == 0)
    {
      gettimeofday(&tv2, NULL);
      u64 elapsed_us = (tv2.tv_sec *1e6 + tv2.tv_usec) - (tv1.tv_sec *1e6 + tv1.tv_usec);
      if (elapsed_us > max_us)
          break;
    }
  }
  cerr << iters <<" iters"<<endl;
}

node_t* best_move(mcts_t *mcts, state_t *state, int max_us) {

  search(mcts, state, max_us);

  return best_child(mcts->current_node, false);
}

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

#define xy_to_board(x, y) y / 3 * 3 + x / 3
#define xy_to_cell(x, y) y % 3 * 3 + x % 3

#define board_to_y(grid, cell) grid / 3 * 3 + cell / 3
#define board_to_x(grid, cell) grid % 3 * 3 + cell % 3

#define idx_to_board(idx) idx / 9
#define idx_to_cell(idx) idx % 9

#define idx_to_x(idx) idx % 9
#define idx_to_y(idx) idx / 9

int main() {

  // game loop
  mcts_t mcts;
  mcts_init(&mcts);
  state_t state;
  while (1) {
    int opponent_row;
    int opponent_col;
    cin >> opponent_row >> opponent_col;
    cin.ignore();

    // start counting time
    auto start  = high_resolution_clock::now();

    // init
    if (!turn){
      // init_decay_pows();
      init_state(&state);
      create_table();
      if (const char* op = getenv("UT_OPENING")) { // harness: plies played before the game
        int r, c, k; const char* p = op;
        while (sscanf(p, "%d %d%n", &r, &c, &k) == 2) { state_move(&state, xy_to_board(c, r), xy_to_cell(c, r)); p += k; }
      }
    }

    if (opponent_row != -1) {
      move_t move = {(u8)(xy_to_board(opponent_col, opponent_row)), (u8)(xy_to_cell(opponent_col, opponent_row))};
      state_move(&state, move.grid, move.cell);
      if (turn)
      {
        int f = 0;
        // cerr << "current "<< (int)mcts.current_node->move.grid << " " <<  (int)mcts.current_node->move.cell << endl;
        for (usize i = 0; i < mcts.current_node->children.size(); ++i){
          node_t *c = mcts.current_node->children.data() + i;
          // cerr << (int)c->move.grid << " " <<  (int)c->move.cell << " " << (int)move.grid << " " << (int)move.cell << endl;
          if (c->move.grid == move.grid && c->move.cell == move.cell){
              // f = 1;
              mcts.current_node = c;
              break;
          }
        }
        // if(!f) cout << "not found wtfd" <<endl;
      }
      // print_board(&state);
    }

    int valid_action_count;
    cin >> valid_action_count;
    cin.ignore();
    for (int i = 0; i < valid_action_count; i++) {
      int row;
      int col;
      cin >> row >> col;
      cin.ignore();
    }
    // Get starting timepoint
    // if (turn || state.last_move != -1)
    if(1)
    {
      /* stuff to do! */
      // cerr << "in" << endl;
      auto init_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
      cerr <<"init time " << init_time << endl;
      int time_limit = (turn == 0 ? 1e6 - 700 : 1e5 - 250) - init_time;
      node_t* best = best_move(&mcts, &state, time_limit);
      mcts.current_node = best;

      cerr << (int)best->move.grid << " " << (int)best->move.cell << endl;
      state_move(&state, best->move.grid, best->move.cell);
      // print_board(&state);
      int row, col;
      col = board_to_x(best->move.grid, best->move.cell);
      row = board_to_y(best->move.grid, best->move.cell);

      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      cerr << "Time taken by function: " << duration.count() << " microseconds"
          << endl;
      cout << row << " " << col <<" "<< iters << endl;
    }
    else
    {
      state_move(&state, 4,4);

      cout << "4 4"<<endl;
    }
    turn++;
  }
}