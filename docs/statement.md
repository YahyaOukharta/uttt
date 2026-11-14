![](//cdn.codingame.com/smash-the-code/statement/league_wood_04.png)

This is a **league based** challenge.

For this challenge, multiple versions of the same game are available. Once you have proven your skills on this first version, you will access a higher league and extra rules will be unlocked.

## The Goal

We're adding a new level of complexity: the game is now played on a tic-tac-toe board where each square is a normal tic-tac-toe. The objective is to align 3 marks on the main tic-tac-toe board by winning on the smaller boards.

## Rules

When a player plays on a small board, he also decides where the next player will be allowed to play: for example, if a player has played in the bottom left square of one of the small boards, the next player will play on the small board located at the bottom left square of the main board.

If a player is sent to a board that is either already won, or full, then that player is allowed to play in any empty square.

You can download the source code of this game on [GitHub](https://github.com/CodinGame/game-ultimate-tictactoe). You can create your own game using the [CodinGame SDK](https://github.com/CodinGame/codingame-sdk-doc)!

### Victory Conditions

You've won on 3 aligned smaller tic-tac-toe boards. If nobody managed to get 3 marks aligned, the player that won the most smaller tic-tac-toe boards wins.

## Game Input

### Input for one game turn

Line 1: 2 space separated integers `opponentRow` and `opponentCol`, the opponent's last action (`-1` `-1` for the first turn).
Line 2: the number of valid actions for this turn, `validActionCount`.
Next `validActionCount` lines: 2 space separated integers `row` and `col`, the coordinates you're allowed to play at.

### Output for one game turn

Line 1: 2 space separated integers `row` and `col`.

### Constraints

Response time for first turn ≤ 1000ms
Response time for one turn ≤ 100ms
