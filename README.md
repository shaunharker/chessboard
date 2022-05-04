# chessboard

This is a Python chessboard that can handle [standard algebraic notation](https://en.wikipedia.org/wiki/Algebraic_notation_(chess)), [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and legal move [generation](https://www.chessprogramming.org/Move_Generation). It is implemented with C++ and pybind11.

There are many interesting techniques in the literature I have not used that appear to be required to obtain the best speeds. However, I couldn't find anything with a fast enough python integration for my needs involving a data-hungry AI project. However I had some good experience with pybind11 so I took a shot at making a simple bitboard implementation. A few days later, this was giving me data much faster. Perhaps someday I'll come back and work on implementing look-ups for all the various pieces in order to shave off some cycles.

IMPORTANT NOTE: I have not yet run the benchmark tree-search test usually used to establish the correctness of the implementation!

## docs

The module provides the `chessboard.Chessboard` class. The three main methods of the `Chessboard` class are:

* `move(game: str)`, which accepts a move or sequence of moves as a string of standard algebraic chess notation (except no move numbers) `"e4 e5"`,
* `fen() -> str`, which provides standard FEN output
* `legal() -> List[str]` provides a list of legal moves given the current board state.

## example usage

The usage is pretty straightforward:

```python
from chessboard import Chessboard as Board

board = Board()
print(board.fen())
print(board.legal())
board.move("e4") # evaluates to true
board.move("a8") # evaluates to false
board.move("42") # evaluates to false
print(board.fen())
print(board.legal())
board.move("Nc6 d3")
print(board.fen())
print(board.legal())
board = Board("e4 e5")
print(board.fen())
print(board.legal())
```

## acknowledgements

I thank the community of chess programmers for their inventiveness and enthusiasm in making so many resources available online.
