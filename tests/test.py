from chessboard import Chessboard as Board

board = Board()
print(board.fen())
print(board.legal())
board.move("e4")
print(board.fen())
print(board.legal())
board.move("Nc6 d3")
print(board.fen())
print(board.legal())
board = Board("e4 e5")
print(board.fen())
print(board.legal())
