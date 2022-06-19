# chessboard

This is a Python chessboard that can handle [standard algebraic notation](https://en.wikipedia.org/wiki/Algebraic_notation_(chess)), [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) and legal move [generation](https://www.chessprogramming.org/Move_Generation). It is implemented with C++ and pybind11. I used it for a deep learning project where I needed direct access to chess engine internals at compiled speeds.

## docs

The module provides the `chessboard.Chessboard` class. The three main methods of the `Chessboard` class are:

* `move(game: str)`, which accepts a move or sequence of moves as a string of standard algebraic chess notation (except no move numbers) `"e4 e5"`,
* `fen() -> str`, which provides standard FEN output
* `legal() -> List[str]` provides a list of legal moves given the current board state.

## example usage

The usage is pretty straightforward:

```python
>>> from chessboard import Chessboard as Board
>>> board = Board()
>>> board.fen()
'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
>>> board.legal()
['Na6', 'Nc6', 'Nf6', 'Nh6', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6', 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5']
>>> board.play("e4")
>>> board.fen()
'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 1 1'
>>> board.play("a8")
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# RuntimeError: illegal move
>>> board.play("e5")
>>> board.fen()
'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 2 2'
>>> board.legal()
['Ke2', 'Qh5', 'Qg4', 'Qf3', 'Qe2', 'Ba6', 'Bb5', 'Bc4', 'Bd3', 'Be2', 'Na3', 'Nc3', 'Nf3', 'Nh3', 'Ne2', 'a3', 'b3', 'c3', 'd3', 'f3', 'g3', 'h3', 'a4', 'b4', 'c4', 'd4', 'f4', 'g4', 'h4']
>>> board.undo()
>>> board.fen()
'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 1 1'
```

## acknowledgements

Thanks to <https://www.chessprogramming.org/Main_Page> for a few hints and tips about corner cases to expect and perft debugging examples.
