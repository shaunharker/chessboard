import random
import string
import chess
from stockfish import Stockfish
from IPython.display import display, Image, HTML
import asyncio
import bisect

# load a book corpus for highlighting
with open("/home/sharker/data/standard-chess.utf8") as infile:
    book = list(infile.readlines())
book = sorted(book)


class TransChess:
    def __init__(self, game=""):
        self.dispid = ''.join(random.choice(string.ascii_uppercase +
            string.ascii_lowercase + string.digits) for _ in range(16))
        self.game = game.strip()
        self.moves = game.split()
        self.board = chess.Board()
        for move in self.moves:
            self.board.push_san(move)
        self.board_display = display(self.board,
            display_id=f"board{self.dispid}")
        self.html = HTML("")
        self.html_display = display(self.html, display_id=f"html{self.dispid}")
        self.html_games = []
        self.engine = Stockfish()
        self.in_book = lambda s: book[bisect.bisect_left(book,
            s)].startswith(s)

    def fen(self):
        return self.board.fen()

    def play(self, move):
        """
        play the given move on the board
        """
        if move is None:
            return None
        move = self.board.san(self.board.parse_san(move))
        self.board.push_san(move)
        self.moves.append(move)
        self.game = self.game + " " + move if self.game != "" else move
        self.display_update()
        return move

    def back(self):
        self.game = ''.join(self.game.split()[:-1])
        self.board.pop()
        self.display_update()

    def restart(self):
        # self.html_games.append(self.game)
        self.game = ''
        for _ in range(len(self.moves)):
            self.board.pop()
        self.moves = []
        self.display_update()

    def legal(self):
        return [self.board.san(move) for move in self.board.legal_moves]

    def generate(self, model=None, time=None):
        """
        Use the provided neural net to come up with next move
        equivalent to `model.move(game)`
        If no model is provides, just pick a random legal move
        """
        if model is None:
            return random.choice(self.legal())
        if model == "stockfish":
            time = time or 1.0
            return self.stockfish(playmove=False, time=time)
        return model.move(self.game)

    # private

    def stockfish(self, playmove=False, time=1.0):
        self.engine.set_fen_position(fen_position = self.board.fen())
        move = self.engine.get_best_move_time(time=1.0)
        if move is None:
            return None
        move_uci = self.board.parse_uci(move)
        move_san = self.board.san(move_uci)
        if playmove:
            self.board.push_san(move_san)
            self.moves.append(move_san)
            self.game = self.game + " " + move_san if self.game != "" else move_san
            self.display_update()
        return move_san

    def highlight_game(self, game):
        if game == "":
            return ""
        moves = game.split()
        hl_game = ""
        for n, move in enumerate(moves):
            if n > 0:
                hl_game += " "
            if self.in_book(' '.join(moves[:n])):
                hl_game += f'<span style="color:blue">{move}</span>'
            else:
                hl_game += f'{move}'
        return hl_game

    def display_update(self):
        self.html_games = self.html_games[-5:]
        self.html = HTML("<pre>" + self.highlight_game(self.game) + '\n' + '\n'.join(self.html_games[::-1]) + "</pre")
        self.html_display.update(self.html)
        self.board_display.update(self.board)
