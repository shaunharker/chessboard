// fastchessboard.cpp
// Shaun Harker 2022-05-09
// MIT LICENSE

#include <iostream>
#include <array>
#include <functional>
#include <utility>
#include <algorithm>
#include <bitset>

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
typedef int Square;

// debugging tool: type wrapper for cout
struct Vizboard {
  Bitboard x;
};

// debugging tool
std::ostream & operator << (std::ostream & stream, Vizboard x) {
  stream << "as bitset: " << std::bitset<64>(x.x) << "\n";
  for (int row = 0; row < 8; ++ row) {
    for (int col = 0; col < 8; ++ col) {
      stream << ((x.x & (1UL << (8*row+col))) ? "1" : "0");
    }
    stream << "\n";
  }
  return stream;
}

// The singleton Bitboards, representing the subset
// consisting of one Square, can be constructed from
// a Square via the power of two operation.
constexpr auto twopow = [](Square x){
  return Bitboard(1) << x;
};

constexpr Square ntz(Bitboard x) {
  constexpr std::array<Square,64> debruijn
      {0, 47,  1, 56, 48, 27,  2, 60,
      57, 49, 41, 37, 28, 16,  3, 61,
      54, 58, 35, 52, 50, 42, 21, 44,
      38, 32, 29, 23, 17, 11,  4, 62,
      46, 55, 26, 59, 40, 36, 15, 53,
      34, 51, 20, 43, 31, 22, 10, 45,
      25, 39, 14, 33, 19, 30,  9, 24,
      13, 18,  8, 12,  7,  6,  5, 63};
  return debruijn[(0x03f79d71b4cb0a89*(x^(x-1)))>>58];
}


// We often want to iterate over the sequence
//  (i, square_to_bitboard(i)) from i = 0, ..., 63,
// so we produce the following array (i.e. type
//  std::array<std::pair<Square, Bitboard>, 64>)
constexpr auto SquareBitboardRelation =
  enumerate(map(twopow, squares));

// Rooks, Bishops, and Queens are "slider" pieces,
// and we need to generate masks showing how they
// can move, starting from any square.
// Thus, we use the following code to take a bitboard
// and repeatedly apply a slide translation to it
// until it becomes zero (indicating the piece(s) have
// all slid off the board) and OR the results.
template<typename BoardOp>
constexpr Bitboard slide(Bitboard x, BoardOp f) {
  return f(x) | ((f(x) == 0) ? 0 : slide(f(x), f));
}

template<typename... BoardOps>
constexpr std::array<uint64_t, 64>
SliderMask(BoardOps... args) {
  auto result = std::array<uint64_t, 64>();
  for (auto [i, x] : SquareBitboardRelation) {
    for (auto f : {args...}) {
      result[i] |= slide(x, f);
    }
  }
  return result;
}

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
// are lost if they go over the edge. North is towards Black,
// West is towards queenside.
constexpr auto w(Bitboard x) -> Bitboard {return (x >> 1) & ~(file_h);}
constexpr auto e(Bitboard x) -> Bitboard {return (x << 1) & ~(file_a);}
constexpr auto s(Bitboard x) -> Bitboard {return (x << 8) & ~(rank_8);}
constexpr auto n(Bitboard x) -> Bitboard {return (x >> 8) & ~(rank_1);}
constexpr auto nw(Bitboard x) -> Bitboard {return n(w(x));}
constexpr auto ne(Bitboard x) -> Bitboard {return n(e(x));}
constexpr auto sw(Bitboard x) -> Bitboard {return s(w(x));}
constexpr auto se(Bitboard x) -> Bitboard {return s(e(x));}
constexpr auto nwn(Bitboard x) -> Bitboard {return nw(n(x));}
constexpr auto nen(Bitboard x) -> Bitboard {return ne(n(x));}
constexpr auto sws(Bitboard x) -> Bitboard {return sw(s(x));}
constexpr auto ses(Bitboard x) -> Bitboard {return se(s(x));}
constexpr auto wnw(Bitboard x) -> Bitboard {return w(nw(x));}
constexpr auto ene(Bitboard x) -> Bitboard {return e(ne(x));}
constexpr auto wsw(Bitboard x) -> Bitboard {return w(sw(x));}
constexpr auto ese(Bitboard x) -> Bitboard {return e(se(x));}

constexpr auto ROOKMASK = SliderMask(n, s, w, e);
constexpr auto BISHOPMASK = SliderMask(nw, sw, ne, se);

// Given a chessboard square i and the Bitboard of empty squares
// on it's "+"-mask, this function determines those squares
// a rook or queen is "attacking".
constexpr Bitboard rookcollisionfreehash(Square i, Bitboard const& E) {
    // E is empty squares intersected with rook "+"-mask
    auto constexpr A = antidiagonal;
    auto constexpr T = rank_8;
    auto constexpr L = file_a;
    auto X = T & (E >> (i & 0b111000));  // 3
    auto Y = (A * (L & (E >> (i & 0b000111)))) >> 56;  // 5
    return (Y << 14) | (X << 6) | i; // 4
}

// Given a singleton bitboard x and the set of empty squares
// on it's "x"-mask, this function packages that information
// into a unique 22-bit key for lookup table access.
constexpr Bitboard bishopcollisionfreehash(Square i, Bitboard const& E) {
  // E is empty squares intersected with bishop "X"-mask
  auto row = i >> 3;
  auto col = i & 7;
  auto t = row - col;
  auto t2 = row + col - 7;
  auto OD = (t > 0) ? (diagonal >> t) : (diagonal << -t);
  auto OA = (t2 > 0) ? (antidiagonal << t2) : (antidiagonal >> -t2);
  auto constexpr L = file_a;
  auto X = (L*(OA&E)) >> 56;
  auto Y = (L*(OD&E)) >> 56;
  return (Y << 14) | (X << 6) | i;
}

std::vector<Bitboard>
computerookthreats(){
  std::vector<Bitboard> result (1UL << 22);
  for (Square i = 0; i < 64; ++ i) {
    Bitboard x = 1UL << i;
    auto const row = i >> 3;
    auto const col = i & 7;
    for (int k = 0x0000; k <= 0xFFFF; k += 0x0001) {
      Bitboard E = Bitboard(0);
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << d)) ? (1UL << (8*row + d)) : 0;
      }
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << (8+d))) ? (1UL << (8*d + col)) : 0;
      }
      // E is empty squares intersected with rook "+"-mask possibility
      auto idx = rookcollisionfreehash(i, E);
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

std::vector<Bitboard> ROOKTHREATS;

std::vector<Bitboard>
computebishopthreats(){
  auto result = std::vector<Bitboard>(1 << 22);
  for (auto const& [i, x] : SquareBitboardRelation) {
    Square const row = i >> 3;
    Square const col = i & 7;
    for (int k = 0x0000; k <= 0xFFFF; k += 0x0001) {
      Bitboard E = Bitboard(0);
      for (int d = 0; d < 8; ++d) {
        Square r = row + col - d;
        if (r < 0 || r >= 8) continue;
        E |= (k & (1 << d)) ? (1UL << (8*r + d)) : 0;
      }
      for (int d = 0; d < 8; ++d) {
        Square r = row - col + d;
        if (r < 0 || r >= 8) continue;
        E |= (k & (1 << (8+d))) ? (1UL << (8*r + d)) : 0;
      }
      // E is empty squares intersected with bishop "x"-mask possibility
      auto idx = bishopcollisionfreehash(i, E);
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

std::vector<Bitboard> BISHOPTHREATS;

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

Bitboard const& rookthreats(Square i, Bitboard const& empty) {
  return ROOKTHREATS[rookcollisionfreehash(i, empty & ROOKMASK[i])];
}

Bitboard const& bishopthreats(Square i, Bitboard const& empty) {
  return BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])];
}

Bitboard queenthreats(Square i, Bitboard const& empty) {
  return ROOKTHREATS[rookcollisionfreehash(i, empty & ROOKMASK[i])] |
    BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])];
}

Bitboard const& knightthreats(Square i) {
  return KNIGHTTHREATS[i];
}

Bitboard const& kingthreats(Square i) {
  return KINGTHREATS[i];
}

Bitboard pawnthreats(Bitboard const& X, bool color) {
  constexpr Bitboard not_file_a = ~file_a;
  constexpr Bitboard not_file_h = ~file_h;
  return color ? (((not_file_a & X) << 7) | ((not_file_h & X) << 9)) :
    (((not_file_h & X) >> 7) | ((not_file_a & X) >> 9));
}

// Move flags
typedef uint32_t MoveFlag;

constexpr MoveFlag CAPTURE_P_STANDARD = 1 << 0;
constexpr MoveFlag CAPTURE_P_EN_PASSANT = 1 << 1;
constexpr MoveFlag CAPTURE_Q = 1 << 2;
constexpr MoveFlag CAPTURE_R = 1 << 3;
constexpr MoveFlag CAPTURE_B = 1 << 4;
constexpr MoveFlag CAPTURE_N = 1 << 5;

constexpr auto CAPTURED = CAPTURE_P_STANDARD |
                          CAPTURE_P_EN_PASSANT |
                          CAPTURE_Q |
                          CAPTURE_R |
                          CAPTURE_B |
                          CAPTURE_N;

constexpr MoveFlag MOVE_P_STANDARD = 1 << 6;
constexpr MoveFlag MOVE_P_DOUBLEPUSH = 1 << 7;
constexpr MoveFlag MOVE_P_EN_PASSANT = 1 << 8;
constexpr MoveFlag MOVE_P_PROMOTE_Q = 1 << 9;
constexpr MoveFlag MOVE_P_PROMOTE_R = 1 << 10;
constexpr MoveFlag MOVE_P_PROMOTE_B = 1 << 11;
constexpr MoveFlag MOVE_P_PROMOTE_N = 1 << 12;
constexpr MoveFlag MOVE_Q = 1 << 13;
constexpr MoveFlag MOVE_R = 1 << 14;
constexpr MoveFlag MOVE_B = 1 << 15;
constexpr MoveFlag MOVE_N = 1 << 16;
constexpr MoveFlag MOVE_K_STANDARD = 1 << 17;
constexpr MoveFlag MOVE_K_KINGSIDE_CASTLE = 1 << 18;
constexpr MoveFlag MOVE_K_QUEENSIDE_CASTLE = 1 << 19;

constexpr auto MOVED = MOVE_P_STANDARD |
                       MOVE_P_DOUBLEPUSH |
                       MOVE_P_EN_PASSANT |
                       MOVE_P_PROMOTE_Q |
                       MOVE_P_PROMOTE_R |
                       MOVE_P_PROMOTE_B |
                       MOVE_P_PROMOTE_N |
                       MOVE_Q |
                       MOVE_R |
                       MOVE_B |
                       MOVE_N |
                       MOVE_K_STANDARD |
                       MOVE_K_KINGSIDE_CASTLE |
                       MOVE_K_QUEENSIDE_CASTLE;

constexpr MoveFlag EN_PASSANT = MOVE_P_EN_PASSANT | CAPTURE_P_EN_PASSANT;

struct Move {  // 20 bytes
  Square s;
  Square t;
  MoveFlag flags;
  Bitboard ep; // for reversibility
  Bitboard rights;
};

void init_tables() {
  ROOKTHREATS = computerookthreats();
  BISHOPTHREATS = computebishopthreats();
}

class Chess {
public:
  Bitboard white;
  Bitboard black;
  Bitboard king;
  Bitboard pawn;
  Bitboard queen;
  Bitboard rook;
  Bitboard bishop;
  Bitboard knight;
  Bitboard enpassant_square; // behind double-pushed pawn
  Bitboard castling_rights; // a1, a8, h1, h8  (a six bit solution would be better)
  char board[65];
  uint64_t ply;

  Chess() : board("rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR") {
    white =            0xFFFF000000000000;  // rank_1 | rank_2;
    black =            0x000000000000FFFF;  // rank_7 | rank_8;
    king =             0x1000000000000010;  // e1 | e8;
    pawn =             0x00FF00000000FF00;  // rank_2 | rank_7
    queen =            0x0800000000000008;  // d1 | d8
    rook =             0x8100000000000081;  // a1 | a8 | h1 | h8;
    bishop =           0x2400000000000024;  // c1 | c8 | f1 | f8;
    knight =           0x4200000000000042;  // b1 | b8 | g1 | g8;
    enpassant_square = 0x0000000000000000;  // behind double-pushed pawn
    castling_rights =  0x8100000000000081;  // a1 | a8 | h1 | h8
    ply = 0;
  }

  void play(Move const& m) {
    auto const& [si, ti, flags, m_ep, m_cr] = m;
    bool color = ply & 1;
    Bitboard & us = color ? black : white;
    Bitboard & them = color ? white : black;
    auto s = 1UL << si;
    auto t = 1UL << ti;
    auto st = s | t;
    ply += 1;
    enpassant_square &= color ? rank_6 : rank_3;
    switch (flags & CAPTURED) {
      case CAPTURE_P_STANDARD:
        pawn ^= t;
        them ^= t;
        break;
      case CAPTURE_P_EN_PASSANT:
      {
        auto ci = ti + (color ? -8 : 8);
        board[ci] = '.';
        auto captured = 1UL << ci;
        pawn ^= captured;
        them ^= captured;
        break;
      }
      case CAPTURE_Q:
        queen ^= t;
        them ^= t;
        break;
      case CAPTURE_R:
        rook ^= t;
        them ^= t;
        castling_rights &= ~st;
        break;
      case CAPTURE_B:
        bishop ^= t;
        them ^= t;
        break;
      case CAPTURE_N:
        knight ^= t;
        them ^= t;
        break;
    }
    switch (flags & MOVED) {
      case MOVE_K_STANDARD:
        us ^= st;
        king ^= st;
        castling_rights &= color ? rank_1 : rank_8;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_K_KINGSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x00000000000000F0UL
                    : 0xF000000000000000UL;
        castling_rights &= color ? rank_1 : rank_8;
        board[ti] = board[si];
        board[si] = '.';
        board[si-1] = color ? 'r' : 'R';
        board[si-4] = '.';
        break;
      case MOVE_K_QUEENSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x0900000000000009UL
                    : 0x0000000000000009UL;
        castling_rights &= color ? rank_1 : rank_8;
        board[ti] = board[si];
        board[si] = '.';
        board[si+1] = color ? 'r' : 'R';
        board[si+3] = '.';
        break;
      case MOVE_P_STANDARD:
        pawn ^= st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_P_DOUBLEPUSH:
        pawn ^= st;
        us ^= st;
        enpassant_square = color ? (t >> 8) : (t << 8);
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_P_EN_PASSANT:
        pawn ^= st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_P_PROMOTE_Q:
        pawn ^= s;
        queen ^= t;
        us ^= st;
        board[ti] = ('q'-'p') + board[si];
        board[si] = '.';
        break;
      case MOVE_P_PROMOTE_N:
        pawn ^= s;
        knight ^= t;
        us ^= st;
        board[ti] = ('n'-'p') + board[si];
        board[si] = '.';
        break;
      case MOVE_P_PROMOTE_R:
        pawn ^= s;
        rook ^= t;
        us ^= st;
        board[ti] = ('r'-'p') + board[si];
        board[si] = '.';
        break;
      case MOVE_P_PROMOTE_B:
        pawn ^= s;
        bishop ^= t;
        us ^= st;
        board[ti] = ('b'-'p') + board[si];
        board[si] = '.';
        break;
      case MOVE_Q:
        queen ^= st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_R:
        rook ^= st;
        castling_rights &= ~st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_B:
        bishop ^= st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
      case MOVE_N:
        knight ^= st;
        us ^= st;
        board[ti] = board[si];
        board[si] = '.';
        break;
    }
  }

  void undo(Move const& m) {
    auto const& [si, ti, flags, m_ep, m_cr] = m;
    ply -= 1;
    bool color = ply & 1;
    Bitboard & us = color ? black : white;
    Bitboard & them = color ? white : black;
    auto s = 1UL << si;
    auto t = 1UL << ti;
    auto st = s | t;
    switch (flags & MOVED) {
      case MOVE_K_STANDARD:
        us ^= st;
        king ^= st;
        board[si] = board[ti];
        break;
      case MOVE_K_KINGSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x00000000000000F0UL
                    : 0xF000000000000000UL;
        board[si] = board[ti];
        board[si+3] = board[si+1];
        board[ti] = '.';
        board[si+1] = '.';
        break;
      case MOVE_K_QUEENSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x0900000000000009UL
                    : 0x0000000000000009UL;
        board[si] = board[ti];
        board[si-4] = board[si-1];
        board[ti] = '.';
        board[si-1] = '.';
        break;
      case MOVE_P_STANDARD:
        pawn ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_P_DOUBLEPUSH:
        pawn ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_P_EN_PASSANT:
        pawn ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_P_PROMOTE_Q:
        pawn ^= s;
        queen ^= t;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_P_PROMOTE_N:
        pawn ^= s;
        knight ^= t;
        us ^= st;
        board[si] = color ? 'p' : 'P';
        break;
      case MOVE_P_PROMOTE_R:
        pawn ^= s;
        rook ^= t;
        us ^= st;
        board[si] = color ? 'p' : 'P';
        break;
      case MOVE_P_PROMOTE_B:
        pawn ^= s;
        bishop ^= t;
        us ^= st;
        board[si] = color ? 'p' : 'P';
        break;
      case MOVE_Q:
        queen ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_R:
        rook ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_B:
        bishop ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
      case MOVE_N:
        knight ^= st;
        us ^= st;
        board[si] = board[ti];
        break;
    }
    switch (flags & CAPTURED) {
      case CAPTURE_P_STANDARD:
        pawn ^= t;
        them ^= t;
        board[ti] = color ? 'P' : 'p';
        break;
      case CAPTURE_P_EN_PASSANT:
      {
        auto ci = ti + (color ? -8 : 8);
        board[ti] = '.';
        board[ci] = color ? 'P' : 'p';
        auto captured = 1UL << ci;
        pawn ^= captured;
        them ^= captured;
        break;
      }
      case CAPTURE_Q:
        queen ^= t;
        them ^= t;
        board[ti] = color ? 'Q' : 'q';
        break;
      case CAPTURE_R:
        rook ^= t;
        them ^= t;
        board[ti] = color ? 'R' : 'r';
        break;
      case CAPTURE_B:
        bishop ^= t;
        them ^= t;
        board[ti] = color ? 'B' : 'b';
        break;
      case CAPTURE_N:
        knight ^= t;
        them ^= t;
        board[ti] = color ? 'N' : 'n';
        break;
      case 0:
        board[ti] = '.';
    }
    enpassant_square = m_ep;
    castling_rights = m_cr;
  }

  bool check() const {
    bool color = ply & 1;
    auto us = color ? black : white;
    auto them = color ? white : black;
    auto oki = ntz(king & us);
    auto checkers = them & attackers(oki);
    return (checkers != 0);
  }

  bool doublecheck() const {
    bool color = ply & 1;
    auto us = color ? black : white;
    auto them = color ? white : black;
    auto oki = ntz(king & us);
    auto attack = them & attackers(oki);
    attack &= ~(attack ^ (attack-1)); // remove lsb
    return (attack != 0);
  }

  bool mate() {
    bool color = ply & 1;
    auto us = color ? black : white;
    auto them = color ? white : black;
    auto oki = ntz(king & us);
    return ((attackers(oki) & them) != 0) && (legal_moves().size()==0);
  }

  auto attackers (Square i) const -> Bitboard {
    auto empty = ~(white | black);
    return (kingthreats(i) & king) |
      (knightthreats(i) & knight) |
      (bishopthreats(i, empty) & bishop) |
      (rookthreats(i, empty) & rook) |
      (queenthreats(i, empty) & queen) |
      (pawnthreats(1UL << i, true) & white & pawn) |
      (pawnthreats(1UL << i, false) & black & pawn);
  }

  std::vector<Move> legal_moves() {
    bool color = ply & 1;
    Bitboard const& us = color ? black : white;
    Bitboard const& them = color ? white : black;
    Bitboard const& endrank = color ? rank_1 : rank_8;
    Bitboard our_king = king & us;
    Square oki = ntz(our_king);
    Bitboard empty = ~(us | them);
    Bitboard not_us = ~us;
    std::vector<Move> result;

    auto add_if_legal = [&](Move m) {
      play(m);
      auto oki = ntz(us & king);
      auto checkers = them & attackers(oki);
      if (checkers == 0) result.push_back(m);
      undo(m);
    };

    auto add = [&](Square si, Bitboard T, MoveFlag flags){
      while (T) {
        auto ti = ntz(T);
        T &= T-1;
        switch(board[ti]) {
          case '.':
            add_if_legal({si, ti, flags, enpassant_square, castling_rights});
            break;
          case 'Q':
            add_if_legal({si, ti, CAPTURE_Q | flags, enpassant_square, castling_rights});
            break;
          case 'q':
            add_if_legal({si, ti, CAPTURE_Q | flags, enpassant_square, castling_rights});
            break;
          case 'R':
            add_if_legal({si, ti, CAPTURE_R | flags, enpassant_square, castling_rights});
            break;
          case 'r':
            add_if_legal({si, ti, CAPTURE_R | flags, enpassant_square, castling_rights});
            break;
          case 'B':
            add_if_legal({si, ti, CAPTURE_B | flags, enpassant_square, castling_rights});
            break;
          case 'b':
            add_if_legal({si, ti, CAPTURE_B | flags, enpassant_square, castling_rights});
            break;
          case 'N':
            add_if_legal({si, ti, CAPTURE_N | flags, enpassant_square, castling_rights});
            break;
          case 'n':
            add_if_legal({si, ti, CAPTURE_N | flags, enpassant_square, castling_rights});
            break;
          case 'P':
            add_if_legal({si, ti, CAPTURE_P_STANDARD | flags, enpassant_square, castling_rights});
            break;
          case 'p':
            add_if_legal({si, ti, CAPTURE_P_STANDARD | flags, enpassant_square, castling_rights});
            break;
        }
      }
    };

    // Standard King Moves (castling comes later)
    add(oki, kingthreats(oki) & not_us, MOVE_K_STANDARD);

    // Queen Moves
    auto S = queen & us;
    while (S) {
      auto si = ntz(S);
      S &= S-1;
      add(si, rookthreats(si, empty) & not_us, MOVE_Q);
      add(si, bishopthreats(si, empty) & not_us, MOVE_Q);
    }

    // Rook moves
    S = rook & us;
    while (S) {
      auto si = ntz(S);
      S &= S-1;
      add(si, rookthreats(si, empty) & not_us, MOVE_R);
    }

    // Bishop moves
    S = bishop & us;
    while (S) {
      auto si = ntz(S);
      S &= S-1;
      add(si, bishopthreats(si, empty) & not_us, MOVE_B);
    }

    // Knight moves
    S = knight & us;
    while (S) {
      auto si = ntz(S);
      S &= S-1;
      add(si, knightthreats(si) & not_us, MOVE_N);
    }

    // Pawn pushes
    Bitboard our_pawns = pawn & us;
    Bitboard T = empty & (color ? (our_pawns << 8) : (our_pawns >> 8));
    while (T) {
      auto ti = ntz(T);
      T &= T-1;
      Square si = ti - (color ? 8 : -8);
      if ((1UL << ti) & endrank) {
        add_if_legal({si, ti, MOVE_P_PROMOTE_Q, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_R, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_B, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_N, enpassant_square, castling_rights});
      } else {
        add_if_legal({si, ti, MOVE_P_STANDARD, enpassant_square, castling_rights});
      }
    }

    // Pawn captures (except en passant)
    T = pawnthreats(our_pawns, color) & them;
    while (T) {
      auto ti = ntz(T);
      T &= T-1;
      Bitboard t = 1UL << ti;
      S = pawnthreats(t, !color) & our_pawns;
      while (S) {
        auto si = ntz(S);
        S &= S-1;
        if (t & endrank) {
          add(si, t, MOVE_P_PROMOTE_Q);
          add(si, t, MOVE_P_PROMOTE_R);
          add(si, t, MOVE_P_PROMOTE_B);
          add(si, t, MOVE_P_PROMOTE_N);
        } else {
          add(si, t, MOVE_P_STANDARD);
        }
      }
    }

    // Double Pawn pushes
    S = our_pawns & (color ? 0x000000000000FF00UL :
                             0x00FF000000000000UL);
    T = empty & (color ? ((S << 16) & (empty << 8))
                       : ((S >> 16) & (empty >> 8)));
    while (T) {
      auto ti = ntz(T);
      T &= T-1;
      Square si = ti - (color ? 16 : -16);
      add_if_legal({si, ti, MOVE_P_DOUBLEPUSH, enpassant_square, castling_rights});
    }

    // En Passant
    S = pawnthreats(enpassant_square, !color) & our_pawns;
    while (S) {
      auto si = ntz(S);
      S &= S-1;
      add_if_legal({si, ntz(enpassant_square), EN_PASSANT, enpassant_square, castling_rights});
    }

    // Kingside Castle
    if (castling_rights & (color ? (1UL << 7) : (1UL << 63))) {
      Bitboard conf = (color ? 240UL : (240UL << 56));
      if ((us & conf) == (color ? 144UL : (144UL << 56))) {
        if ((empty & conf) == (color ? 96UL : (96UL << 56))) {
          if (((attackers(oki + Square(0)) & them) == 0) &&
              ((attackers(oki + Square(1)) & them) == 0) &&
              ((attackers(oki + Square(2)) & them) == 0)) {
            result.push_back({oki, oki + Square(2),
              MOVE_K_KINGSIDE_CASTLE, enpassant_square, castling_rights});
          }
        }
      }
    }

    // Queenside Castle
    if (castling_rights & (color ? (1UL << 0) : (1UL << 56))) {
      auto conf = (color ? 31UL : (31UL << 56));
      if ((us & conf) == (color ? 17UL : (17UL << 56))) {
        if ((empty & conf) == (color ? 14UL : (14UL << 56))) {
          if (((attackers(oki - Square(0)) & them) == 0) &&
              ((attackers(oki - Square(1)) & them) == 0) &&
              ((attackers(oki - Square(2)) & them) == 0)) {
            result.push_back({oki, oki - Square(2),
              MOVE_K_QUEENSIDE_CASTLE, enpassant_square, castling_rights});
          }
        }
      }
    }
    return result;
  }
};



// tests

uint64_t perft(Chess & board, int depth) {
  if (depth == 0) return 1;
  auto moves = board.legal_moves();
  if (depth == 1) return moves.size();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    result += perft(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t capturetest(Chess & board, int depth) {
  if (depth == 0) return 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    if ((depth == 1) && (move.flags & CAPTURED)) result += 1;
    result += capturetest(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t enpassanttest(Chess & board, int depth) {
  if (depth == 0) return 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    if ((depth == 1) && (move.flags & EN_PASSANT)) result += 1;
    result += enpassanttest(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t checktest(Chess & board, int depth) {
  if (depth == 0) return board.check() ? 1 : 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    result += checktest(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t doublechecktest(Chess & board, int depth) {
  if (depth == 0) return board.doublecheck() ? 1 : 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    result += doublechecktest(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t matetest(Chess & board, int depth) {
  if (depth == 0) return board.mate() ? 1 : 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    result += matetest(board, depth-1);
    board.undo(move);
  }
  return result;
}



// pybind11
/// Python Bindings

// #include <fstream>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;
//
// PYBIND11_MODULE(chessboard, m) {
//   py::class_<Chess>(m, "Chess")
//     .def(py::init<>())
//     .def("legal", &Chess::legal_moves)
//     .def("play", &Chess::play)
//     .def("undo", &Chess::undo);
//   m.def("init_tables", &init_tables);
// }

int main(int argc, char * argv []) {
  init_tables();
  auto board = Chess();
  for (int depth = 0; depth < 8; ++ depth) {
    std::cout << depth << ": " << perft(board, depth) << "\n";
  }
  std::cout << "complete.\n";
  return 0;
}
