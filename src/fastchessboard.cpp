// fastchessboard.cpp
// Shaun Harker 2022-05-09
// MIT LICENSE

#include <iostream>
#include <array>
#include <functional>
#include <utility>

// non-lazy constexpr versions of range, map, and enumerate
template <uint64_t N>
constexpr auto range() {
    std::array<uint64_t, N> result {};
    for(uint64_t i = 0; i < N; ++i) result[i] = i;
    return result;
}

template <typename Func, typename Seq>
constexpr auto map(Func func, Seq seq) {
    typedef typename Seq::value_type value_type;
    using return_type = decltype(func(std::declval<value_type>()));
    std::array<return_type, std::tuple_size<Seq>::value> result {};
    uint64_t i = 0;
    for (auto x : seq) result[i++] = func(x);
    return result;
}

template <typename Seq>
constexpr auto enumerate(Seq seq) {
    typedef typename Seq::value_type value_type;
    std::array<std::pair<uint64_t, value_type>, std::tuple_size<Seq>::value> result {};
    uint64_t i = 0;
    for (auto x : seq) {
      // `result[i] = {i, x};` creates a gcc 9.4.0 C++17
      // compiler error complaining `pair::operator=` isn't
      // marked `constexpr`. But this works:
      result[i].first = i;
      result[i].second = x;
      ++i;
    }
    return result;
}


// Bitboards

// Chessboard squares are represented as integers in range(64)
// as follows:
//   0 ~ a8, 1 ~ b8, ..., 62 ~ g1, 63 ~ h1
constexpr auto squares = range<64>();

// A Bitboard represents a subset of chessboard squares
//  using the bits of a 64 bit unsigned integer such that
// the least significant bit corresponds to the integer 0,
// and so on.
typedef uint64_t Bitboard;

// The singleton Bitboards, representing the subset
// consisting of one Square, can be constructed from
// a Square via the power of two operation.
constexpr Bitboard square_to_bitboard(Square i) {
  return Bitboard(1) << i;
}

// We often want to iterate over the sequence
//  (i, square_to_bitboard(i)) from i = 0, ..., 63,
// so we produce the following array (i.e. type
//  std::array<std::pair<Square, Bitboard>, 64>)
constexpr auto SquareBitboardRelation =
  enumerate(map(square_to_bitboard, squares));

// Special Bitboards
constexpr Bitboard rank_8       = 0x00000000000000FFUL;
constexpr Bitboard rank_6       = 0x0000000000FF0000UL;
constexpr Bitboard rank_3       = 0x0000FF0000000000UL;
constexpr Bitboard rank_1       = 0xFF00000000000000UL;
constexpr Bitboard file_a       = 0x0101010101010101UL;
constexpr Bitboard file_h       = 0x8080808080808080UL;
constexpr Bitboard diagonal     = 0x8040201008040201UL;
constexpr Bitboard antidiagonal = 0x0102040810204080UL;

// The following functions translate bitboards "west" "east"
// "south" "north" and so on. Bits do not "roll around" but instead
// are lost if they go over the edge.
constexpr auto w(Bitboard x) -> Bitboard {return (x >> 1) & ~(file_h);}
constexpr auto e(Bitboard x) -> Bitboard {return (x << 1) & ~(file_a);}
constexpr auto s(Bitboard x) -> Bitboard {return (x << 8) & ~(rank_8);}
constexpr auto n(Bitboard x) -> Bitboard {return (x >> 8) & ~(rank_1);}
constexpr auto nw(Bitboard x) -> Bitboard {return n(w(x));}
constexpr auto ne(Bitboard x) -> Bitboard {return n(e(x));}
constexpr auto sw(Bitboard x) -> Bitboard {return s(w(x));}
constexpr auto se(Bitboard x) -> Bitboard {return s(e(x));}
constexpr auto nwn(Bitboard x) -> Bitboard {return nw(n(x)));}
constexpr auto nen(Bitboard x) -> Bitboard {return ne(n(x)));}
constexpr auto sws(Bitboard x) -> Bitboard {return sw(s(x)));}
constexpr auto ses(Bitboard x) -> Bitboard {return se(s(x)));}
constexpr auto wnw(Bitboard x) -> Bitboard {return w(nw(x)));}
constexpr auto ene(Bitboard x) -> Bitboard {return e(ne(x)));}
constexpr auto wsw(Bitboard x) -> Bitboard {return w(sw(x)));}
constexpr auto ese(Bitboard x) -> Bitboard {return e(se(x)));}

// Given a singleton bitboard x and the set of empty squares
// on it's "+"-mask, this function determines those squares
// a rook or queen is "attacking".
constexpr Bitboard rooklook(Bitboard const& x, Bitboard const& E) {  //
    // E is empty squares intersected with rook "+"-mask
    auto s = ntz(x);  // 5
    auto constexpr A = antidiagonal;
    auto constexpr T = rank_8;
    auto constexpr L = file_a;
    auto X = T & (E >> (s & 0b111000));  // 3
    auto Y = modmul(A, L & (E >> (s & 0b000111))) >> 56;  // 5
    return (Y << 14) | (X << 6) | s; // 4
}

// Given a singleton bitboard x and the set of empty squares
// on it's "x"-mask, this function determines those squares
// a bishop or queen is "attacking".
constexpr Bitboard bishoplook(Bitboard const& x, Bitboard const& E) {  //
  // E is empty squares intersected with bishop "X"-mask
  auto s = ntz(x);  // 5
  auto row = s >> 3;  // 1
  auto col = s & 7;  // 1
  auto t = row - col;  // 1
  auto t2 = row + col - 7 // 2
  auto OD = diagonal >> t ; // 1
  auto OA = antidiagonal << t2;  // 1
  auto constexpr L = file_a;
  auto X = modmul(L, OA&E) >> 56;  // 3
  auto Y = modmul(L, OD&E) >> 56;  // 3
  return (Y << 14) | (X << 6) | s;  // 4
}

constexpr std::array<Bitboard, (1 << 22)> computerookthreats(){
  auto result = std::array<Bitboard, (1 << 22)>();
  for (auto const& [i, x] : SquareBitboardRelation) { // can loop variable be constexpr?
    auto const row = i >> 3;
    auto const col = i & 7;
    for (int k = 0x0000; k <= 0xFFFF; k += 0x0001) {
      auto E = uint64_t(0);
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << d)) ? (1UL << (8*row + d)) : 0;
      }
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << (8+d))) ? (1UL << (8*d + col)) : 0;
      }
      // E is empty squares intersected with rook "+"-mask possibility
      idx = rooklook(x, E);
      for (auto f: {n, w, s, e}) {
        auto tmp = f(x);
        for (int d = 0; d < 8; ++d) {
          result[idx] |= tmp;
          if ((tmp & E) == 0) break;
          tmp = f(tmp);
        }
      }
    }
  }
  return result;
}

constexpr std::array<Bitboard, (1 << 22)> ROOKTHREATS =
  computerookthreats();

constexpr std::array<Bitboard, (1 << 22)> computebishopthreats(){
  auto result = std::array<Bitboard, (1 << 22)>();
  for (auto const& [i, x] : SquareBitboardRelation) {
    auto const row = i >> 3;
    auto const col = i & 7;
    auto const s = row+col;
    auto const t = row-col;
    for (int k = 0; k < 65536; ++k) {
      auto E = uint64_t(0);
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << d)) ? (1UL << (8*(s-d) + d)) : 0;
      }
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << (8+d))) ? (1UL << (8*(t+d) + d)) : 0;
      }
      // E is empty squares intersected with bishop "x"-mask possibility
      idx = bishoplook(x, E);
      for (auto f: {nw, ne,
                    sw, se}) {
        auto tmp = f(x);
        for (int d = 0; d < 8; ++d) {
          result[idx] |= tmp;
          if ((tmp & E) == 0) break;
          tmp = f(tmp);
        }
      }
    }
  }
  return result;
}

constexpr std::array<Bitboard, (1 << 22)> BISHOPTHREATS =
  computebishopthreats();

constexpr std::array<Bitboard, 64> computeknightthreats(){
  auto result = std::array<Bitboard, 64>();
  for (auto const& [i, x] : SquareBitboardRelation) {
    result[i] =          nwn(x) | nen(x) |
                wnw(x) |                   ene(x) |

                wsw(x) |                   ese(x) |
                         sws(x) | ses(x);
  }
  return result;
}

constexpr std::array<Bitboard, 64> KNIGHTTHREATS = computeknightthreats();

constexpr std::array<Bitboard, 64> computekingthreats(){
  auto result = std::array<Bitboard, 64>();
  for (auto const& [i, x] : SquareBitboardRelation) {
    result[i] = nw(x) |  n(x) | ne(x) |
                 w(x) |          e(x) |
                sw(x) |  s(x) | se(x);
  }
  return result;
}

constexpr std::array<Bitboard, 64> KINGTHREATS = computekingthreats();


Bitboard const& rookthreats(Bitboard const& x, Bitboard const& empty) {
  return ROOKTHREATS[rooklook(x, empty)];
}

Bitboard const& bishopthreats(Bitboard const& x, Bitboard const& empty) {
  return BISHOPTHREATS[bishoplook(x, empty)];
}

Bitboard const& queenthreats(Bitboard const& x, Bitboard const& empty) {
  return ROOKTHREATS[rooklook(x, empty)] | BISHOPTHREATS[bishoplook(x, empty)];
}

Bitboard const& knightthreats(Bitboard const& x) {
  return KNIGHTTHREATS[ntz(x)];
}

Bitboard const& kingthreats(Bitboard const& x) {  // 40
  return KINGTHREATS[ntz(x)];
}

Bitboard pawnthreats(Bitboard const& X, bool color) { // 4
  return color ? ((X << 7) | (X << 9)) : ((X >> 7) | (X >> 9));
}


// Move flags
typedef uint32_t MoveFlag;

constexpr MoveFlag WHITE_MOVED = 1 << 0;

constexpr MoveFlag PAWN_CAPTURED = 1 << 1;
constexpr MoveFlag EN_PASSANT = 1 << 2;
constexpr MoveFlag KNIGHT_CAPTURED = 1 << 3;
constexpr MoveFlag BISHOP_OR_QUEEN_CAPTURED = 1 << 4;
constexpr MoveFlag ROOK_OR_QUEEN_CAPTURED = 1 << 5;

constexpr auto CAPTURED = PAWN_CAPTURED |
                          EN_PASSANT |
                          KNIGHT_CAPTURED |
                          BISHOP_OR_QUEEN_CAPTURED |
                          ROOK_OR_QUEEN_CAPTURED;

constexpr MoveFlag STANDARD_KING_MOVE = 1 << 6;
constexpr MoveFlag KINGSIDE_CASTLE = 1 << 7;
constexpr MoveFlag QUEENSIDE_CASTLE = 1 << 8;
constexpr MoveFlag STANDARD_PAWN_MOVE = 1 << 9;
constexpr MoveFlag DOUBLEPUSH = 1 << 10;
constexpr MoveFlag PROMOTE_Q = 1 << 11;
constexpr MoveFlag PROMOTE_R = 1 << 12;
constexpr MoveFlag PROMOTE_B = 1 << 13;
constexpr MoveFlag PROMOTE_N = 1 << 14;
constexpr MoveFlag KNIGHT_MOVED = 1 << 15;
constexpr MoveFlag BISHOP_OR_QUEEN_MOVED = 1 << 16;
constexpr MoveFlag ROOK_OR_QUEEN_MOVED = 1 << 17;

constexpr auto MOVED = STANDARD_KING_MOVE |
                       KINGSIDE_CASTLE |
                       QUEENSIDE_CASTLE |
                       PAWN_MOVED |
                       DOUBLEPUSH |
                       PROMOTE_Q |
                       PROMOTE_R |
                       PROMOTE_B |
                       PROMOTE_N |
                       KNIGHT_MOVED |
                       BISHOP_OR_QUEEN_MOVED |
                       ROOK_OR_QUEEN_MOVED;


struct Move {  // 20 bytes
  Bitboard source;
  Bitboard target;
  MoveFlag flags;
};

class Chess {
private:
  Bitboard white;
  Bitboard black;
  Bitboard king;
  Bitboard rook_or_queen;
  Bitboard bishop_or_queen;
  Bitboard knight;
  Bitboard pawn;
  Bitboard enpassant_square; // behind double-pushed pawn
  Bitboard castling_rights; // K/Q/k/q ~ h1/a1/h8/a8
  uint64_t ply;
  uint64_t halfmove_clock;

public:

  Chess() {
    white = 0xFFFF000000000000;  // rank_1 | rank_2;
    black = 0x000000000000FFFF;  // rank_7 | rank_8;
    king = 0x1000000000000010;  //e1 | e8;
    bishop_or_queen = 0x2C0000000000002C;  // c1 | c8 | d1 | d8 | f1 | f8;
    rook_or_queen = 0x8900000000000089;  // a1 | a8 | d1 | d8 | h1 | h8;
    knight = 0x4200000000000042;  // b1 | b8 | g1 | g8;
    pawn = 0x00FF00000000FF00;  // rank_2 | rank_7
    enpassant_square = 0; // behind double-pushed pawn
    castling_rights = 0x8100000000000081;  // a1 | a8 | h1 | h8
    ply = 0;
    halfmove_clock = 0;
  }

  void
  playmove(Move const& m) {
    auto const& [s, t, flags] = m;

    bool color = ply & 1;
    Bitboard const& us = color ? black : white;
    Bitboard const& them = color ? white : black;

    // This happens in all cases:
    auto st = s | t;
    us ^= st;
    ply += 1;


    if (flags & CAPTURED) {
      halfmove_clock = 0;
    } else {
      halfmove_clock += 1;
    }

    switch (flags & CAPTURED) {
      case PAWN_CAPTURED:
        pawn ^= t;
        them ^= t;
        break;
      case EN_PASSANT:
        auto captured = en_passant << (color ? -8 : 8);
        pawn ^= st | captured;
        them ^= captured;
        break;
      case KNIGHT_CAPTURED:
        knight ^= t;
        them ^= t;
        break;
      case BISHOP_OR_QUEEN_CAPTURED:
        bishop_or_queen ^= t;
        them ^= t;
        break;
      case ROOK_OR_QUEEN_CAPTURED:
        rook_or_queen ^= t;
        them ^= t;
        castling_rights &= ~st;
        break;
      case BISHOP_OR_QUEEN_CAPTURED | ROOK_OR_QUEEN_CAPTURED:
        bishop_or_queen ^= t;
        rook_or_queen ^= t;
        them ^= t;
        break;
    }

    switch (flags & MOVED) {
        case STANDARD_KING_MOVE:
          king ^= st;
          castling_rights &= color ? rank_1 : rank_8;
          break;
        case KINGSIDE_CASTLE:
          rook_or_queen ^= ksc;
          us ^= white_move ? 0xA000000000000000UL
                           : 0x00000000000000A0UL;
          castling_rights &= color ? rank_1 : rank_8;
          break;
        case QUEENSIDE_CASTLE:
          rook_or_queen ^= qsc;
          us ^= white_move ? 0x0900000000000009UL
                           : 0x0000000000000009UL;
          castling_rights &= color ? rank_1 : rank_8;
          break;
        case STANDARD_PAWN_MOVE:
          pawn ^= st;
          break;
        case DOUBLEPUSH:
          pawn ^= st;
          enpassant_square = t << (color ? -8 : 8);
          break;
        case PROMOTE_Q:
          pawn ^= s;
          rook_or_queen ^= t;
          bishop_or_queen ^= t;
          break;
        case PROMOTE_N:
          pawn ^= s;
          knight ^= t;
          break;
        case PROMOTE_R:
          pawn ^= s;
          rook_or_queen ^= t;
          break;
        case PROMOTE_B:
          pawn ^= s;
          bishop_or_queen ^= t;
          break;
        case KNIGHT_MOVED:
          knight ^= st;
          break;
        case BISHOP_OR_QUEEN_MOVED:
          bishop_or_queen ^= st;
          break;
        case ROOK_OR_QUEEN_MOVED:
          rook_or_queen ^= st;
          castling_rights &= ~st;
          break;
        case ROOK_OR_QUEEN_MOVED | BISHOP_OR_QUEEN_MOVED:
          bishop_or_queen ^= st;
          rook_or_queen ^= st;
          break;
    }
  }
};



Bitboard attackers(Chess const& g, Bitboard const& x) {
  bool color = g.ply & 1;
  Bitboard const& us = color ? g.black : g.white;
  Bitboard const& them = color ? g.white : g.black;
  auto empty = ~(us | them);
  return them & (
    (kingthreats(x) & g.king)                     |
    (knightthreats(x) & g.knight)                 |
    (bishopthreats(x, empty) & g.bishop_or_queen) |
    (rookthreats(x, empty) & g.rook_or_queen)     |
    (pawnthreats(x, color) & g.pawn));
}


std::vector<Move> legal_moves(BoardState const& g) {
  bool color = g.ply & 1;
  Bitboard const& us = color ? g.black : g.white;
  Bitboard const& them = color ? g.white : g.black;
  Bitboard const& endrank = color ? rank_1 : rank_8;
  auto COLOR_MOVED = color ? BLACK_MOVED : WHITE_MOVED;
  auto our_king = g.king & g.us;
  auto empty = ~(g.us | g.them);
  auto not_us = ~g.us;

  std::vector<Move> result;

  auto add_if_legal = [&](Move const& m) {
    auto const& [s, t, flags] = m;
    auto h = playmove(g, m);
    auto maybe_moved_king = (flags & KING_MOVED) ? t : our_king;
    if (attackers(h, maybe_moved_king) == 0) {
      // legal move
      result.push_back(m);
    }
  };

  auto add = [&](Bitboard s, Bitboard T, MoveFlag flags){
    while (T) {
      t = ((T ^ (T - 1)) >> 1) + 1;  // 4
      T ^= t;
      Move m {s, t, flags};
      add_if_legal(m);
    }
  };

  // Pseudolegal King moves
  add(our_king, kingthreats(our_king) & not_us, KING_MOVED | COLOR_MOVED);

  auto S = g.bishop_or_queen & us;
  while (S) {
    auto x = ((S ^ (S - 1)) >> 1) + 1;  // 4
    S ^= x;
    add(x, bishopthreats(x) & not_us, BISHOP_OR_QUEEN_MOVED | COLOR_MOVED);
  }

  S = g.rook_or_queen & us;
  while (S) {
    auto x = ((S ^ (S - 1)) >> 1) + 1;  // 4
    S ^= x;
    add(x, rookthreats(x) & not_us, ROOK_OR_QUEEN_MOVED | COLOR_MOVED);
  }

  S = g.knight & us;
  while (S) {
    auto x = ((S ^ (S - 1)) >> 1) + 1;  // 4
    S ^= x;
    add(x, knightthreats(x) & not_us, KNIGHT_MOVED | COLOR_MOVED);
  }

  // Pseudolegal Pawn pushes
  S = g.pawn & us
  auto T = (S << (color ? 8 : -8)) & empty;
  while (T) {
    auto y = ((T ^ (T - 1)) >> 1) + 1;  // 4
    T ^= y;
    auto x = y >> (color ? 8 : -8);
    if (y & endrank) {
      add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_Q});
      add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_R});
      add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_B});
      add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_N});
    } else {
      add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED});
    }
  }

  // Pseudolegal Pawn captures
  T = pawnthreats(us & g.pawn, color) & them;
  while (T) {
    auto y = ((T ^ (T - 1)) >> 1) + 1;  // 4
    T ^= y;
    S = pawnthreats(y, ~color) & us & g.pawn;
    while (S) {
      x = ((S ^ (S - 1)) >> 1) + 1;  // 4
      S ^= x;
      if (y & endrank) {
        add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_Q});
        add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_R});
        add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_B});
        add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED | PROMOTE_N});
      } else {
        add_if_legal({x, y, PAWN_MOVED | COLOR_MOVED});
      }
    }
  }

  // Pseudolegal Double Pawn pushes
  S = g.pawn & us & (color ? (255UL << 8) : (255UL << 48));
  T = (S << (color ? 16 : -16)) & (empty << (color ? 8 : -8)) & empty;
  while (T) {
    auto y = ((T ^ (T - 1)) >> 1) + 1;  // 4
    T ^= y;
    auto x = y >> (color ? 16 : -16);
    add_if_legal({x, y, PAWN_PUSHED | COLOR_MOVED});
  }

  // Pseudolegal En Passant
  S = pawnthreats(g.enpassant_square, ~color) & us & g.pawn;
  while (S) {
    x = ((S ^ (S - 1)) >> 1) + 1;  // 4
    S ^= x;
    add_if_legal({x, g.enpassant_square,
      PAWN_MOVED | EN_PASSANT | COLOR_MOVED});
  }

  // Legal Kingside Castle
  if (g.castling_rights & (color ? (1UL << 7) : (1UL << 63))) {
    auto conf = (color ? 240UL : (240UL << 56));
    if ((us & conf) == (color ? 144UL : (144UL << 56))) {
      if ((empty & conf) == (color ? 96UL : (96UL << 56))) {
        if ((attackers(g, our_king) == 0) &&
          (attackers(g, our_king << 1) == 0) &&
          (attackers(g, our_king << 2) == 0)) {
            result.push_back({our_king, our_king << 2, KINGSIDE_CASTLE});
        }
      }
    }
  }

  // Legal Queenside Castle
  if (g.castling_rights & (color ? (1UL << 0) : (1UL << 56))) {
    auto conf = (color ? 31UL : (31UL << 56));
    if ((us & conf) == (color ? 17UL : (17UL << 56))) {
      if ((empty & conf) == (color ? 14UL : (14UL << 56))) {
        if ((attackers(g, our_king) == 0) &&
          (attackers(g, our_king >> 1) == 0) &&
          (attackers(g, our_king >> 2) == 0)) {
            result.push_back({our_king, our_king >> 2, QUEENSIDE_CASTLE});
        }
      }
    }
  }

  return result;
}

int main(int argc, char * argv []) {
  constexpr auto g = newgame();
  auto moves = legal_moves(g);
  for (auto const& move : moves) {
    std::cout << ntz(move.source) << " " << ntz(move.target) << " " << move.flags << "\n";
  }
  return 0;
}
