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
  //std::cout << "crt. enter \n";
  //std::cout.flush();
  std::vector<Bitboard> result (1UL << 22);
  //std::cout << "array ready to go.\n";
  for (Square i = 0; i < 64; ++ i) {
    Bitboard x = 1UL << i;
    //std::cout << "crt loop. " << i << " " << x << "\n";
    //std::cout.flush();
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
      //if (k % 0x1000 == 0) {
      //  std::cout << "k = " << k << "\n";
      //  std::cout.flush();
      //}
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

std::vector<Bitboard> ROOKTHREATS; // = computerookthreats();

//std::array<Bitboard, (1 << 22)>
std::vector<Bitboard>
computebishopthreats(){
  auto result = std::vector<Bitboard>(1 << 22);
  for (auto const& [i, x] : SquareBitboardRelation) {
    Square const row = i >> 3;
    Square const col = i & 7;
    Square const s = row+col;
    Square const t = row-col;
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

  void
  play(Move const& m) {
    auto const& [si, ti, flags, m_ep, m_cr] = m;

    bool color = ply & 1;
    Bitboard & us = color ? black : white;
    Bitboard & them = color ? white : black;
    auto s = 1UL << si;
    auto t = 1UL << ti;
    auto st = s | t;
    ply += 1;
    enpassant_square &= color ? rank_6 : rank_3;
    //
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

    // if (us & them) {
    //   std::cout << "us and them error\n";
    //   std::cout << "us:\n" << Vizboard({us}) << "them:\n" << Vizboard({them}) << "\n";
    //
    //   return result;
    // }

    auto add_if_legal = [&](Move m) {

      auto const& [si, ti, flags, ep, cr] = m;
      // std::cout << "\n\n-----> add_if_legal. si = " << si << " ti = " << ti << "\n";
      // std::cout << "s = " << (char)('a'+(si&7)) << (8-(si >> 3)) << "\n";
      // std::cout << "t = " << (char)('a'+(ti&7)) << (8-(ti >> 3)) << "\n";
      // std::cout << "board[si] = '" << board[si] << "' board[ti] = '" << board[ti] << "'\n";
      // char piece = (board[si] > 'Z') ? board[si] - 32 : board[si];
      // if (piece == 'P') {
      //   if (flags&CAPTURED) {
      //     piece = (char)('a'+(si&7));
      //   } else {
      //     piece = ' ';
      //   }
      // }
      // std::cout << piece << ((flags&CAPTURED)?"x":"") << (char)('a'+(ti&7)) << (8-(ti >> 3)) << " ";

      // std::cout << "\n\n------------------\nIntegrity check A\n\n";
      // for (int r = 0; r < 8; ++ r) {
      //   for (int c = 0; c < 8; ++ c) {
      //     std::cout << board[8*r+c] << " ";
      //   }
      //   std::cout << "\n";
      // }
      // std::cout << "white:\n" << Vizboard({white}) << "\nblack:\n" << Vizboard({black}) << "\n";
      // for (Square i = 0; i < 64; ++ i) {
      //   uint64_t x = 1UL << i;
      //   if (board[i] == '.') {
      //     if (white & x) exit(1);
      //     if (black & x) exit(1);
      //   }
      //   if (board[i] >= 'a' && board[i] <= 'z') {
      //     if (white & x) exit(1);
      //     if (!(black & x)) exit(1);
      //   }
      //   if (board[i] >= 'A' && board[i] <= 'Z') {
      //     if (!(white & x)) exit(1);
      //     if ((black & x)) exit(1);
      //   }
      // }


      play(m);
      auto oki = ntz(us & king);
      auto checkers = them & attackers(oki);
      //std::cout << "\n  oki: " << oki << "\n";
      //std::cout << "\n  color: " << (color ? "black" : "white") << "\n";
      //std::cout << "\n  king   : " << std::bitset<64>(king) << "\n";
      //std::cout << "\n  bish   : " << std::bitset<64>(bishop) << "\n";
      //std::cout << "\n  us     : " << std::bitset<64>(us) << "\n";
      //std::cout << "\n  them   : " << std::bitset<64>(them) << "\n";

      //std::cout << "\n  bishatt: " << std::bitset<64>(bishopthreats(oki, empty)) << "\n";
      auto empty = ~(us | them);
      //std::cout << "\n  empty  : " << std::bitset<64>(empty) << "\n";
      //std::cout << "\n  bmask  : " << std::bitset<64>(BISHOPMASK[oki]) << "\n";
      //std::cout << "\n  ints   : " << std::bitset<64>(empty & BISHOPMASK[oki]) << "\n";
      //std::cout << "\n  bcfh   : " << std::bitset<22>(bishopcollisionfreehash(oki, empty & BISHOPMASK[oki])) << "\n";
      //std::cout << "\n  answ   : " << std::bitset<64>(BISHOPTHREATS[bishopcollisionfreehash(oki, empty & BISHOPMASK[oki])]) << "\n";
      //BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])]
      //
      // 0100 0100
      // 0010 1000
      // 0000 0000
      // 0010 1000
      // 0100 0100
      // 1000 0010
      // 0000 0001

      if (checkers == 0) {
         result.push_back(m);
         // std::cout << " ...accepted.\n";
       } else {
         // std::cout << " ...rejected.\n";
       }


       // std::cout << "\n\n------------------\nIntegrity check B\n\n";
       // for (int r = 0; r < 8; ++ r) {
       //   for (int c = 0; c < 8; ++ c) {
       //     std::cout << board[8*r+c] << " ";
       //   }
       //   std::cout << "\n";
       // }
       // std::cout << "white:\n" << Vizboard({white}) << "\nblack:\n" << Vizboard({black}) << "\n";
       // for (Square i = 0; i < 64; ++ i) {
       //   uint64_t x = 1UL << i;
       //   if (board[i] == '.') {
       //     if (white & x) exit(1);
       //     if (black & x) exit(1);
       //   }
       //   if (board[i] >= 'a' && board[i] <= 'z') {
       //     if (white & x) exit(1);
       //     if (!(black & x)) exit(1);
       //   }
       //   if (board[i] >= 'A' && board[i] <= 'Z') {
       //     if (!(white & x)) exit(1);
       //     if ((black & x)) exit(1);
       //   }
       // }



      undo(m);

      // std::cout << "\n\n------------------\nIntegrity check B\n\n";
      // for (int r = 0; r < 8; ++ r) {
      //   for (int c = 0; c < 8; ++ c) {
      //     std::cout << board[8*r+c] << " ";
      //   }
      //   std::cout << "\n";
      // }
      // std::cout << "white:\n" << Vizboard({white}) << "\nblack:\n" << Vizboard({black}) << "\n";
      // for (Square i = 0; i < 64; ++ i) {
      //   uint64_t x = 1UL << i;
      //   if (board[i] == '.') {
      //     if (white & x) exit(1);
      //     if (black & x) exit(1);
      //   }
      //   if (board[i] >= 'a' && board[i] <= 'z') {
      //     if (white & x) exit(1);
      //     if (!(black & x)) exit(1);
      //   }
      //   if (board[i] >= 'A' && board[i] <= 'Z') {
      //     if (!(white & x)) exit(1);
      //     if ((black & x)) exit(1);
      //   }
      // }

    };

    auto add = [&](Square si, Bitboard T, MoveFlag flags){
      //std::cout << "ADD si = " << si << " T = " << std::bitset<64>(T) << "\n";
      //std::cout << "    flags = " << std::bitset<20>(flags) << "\n";
      while (T) {
        auto t = ((T ^ (T - 1)) >> 1) + 1;
        T ^= t;
        auto ti = ntz(t);
        //std::cout << "\n--->Add si = " << si << " " << "ti = " << ti << "\n";
        //std::cout << "board[si] = '" << board[si] << "'\n";
        //std::cout << "board[ti] = '" << board[ti] << "'\n";


        //for ( int k = 0; k<64;++k) std::cout << board[k];
        //std::cout << "'\n";

        //std::cout << "preflags  = " << std::bitset<20>(flags) << "\n";
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
    //std::cout << "\n------------------\nKing moves.\n";
    add(oki, kingthreats(oki) & not_us, MOVE_K_STANDARD);

    // Queen Moves
    //std::cout << "\n------------------\nQueen moves.\n";
    auto S = queen & us;
    while (S) {
      auto s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      auto si = ntz(s);
      //std::cout << "Q si = " << si << " board[si] = '" << board[si] << "\n";
      // std::cout << "qb empty  = " << std::bitset<64>(empty) << "\n";
      // std::cout << "qbthreats = " << std::bitset<64>(bishopthreats(si, empty)) << "\n";
      // std::cout << "bmask     = \n" << std::bitset<64>(BISHOPMASK[si]) << "\n";
      // std::cout << "bmask     = \n" << Vizboard({BISHOPMASK[si]}) << "\n";
      // std::cout << "bmask&empt= " << std::bitset<64>(empty&BISHOPMASK[si]) << "\n";
      // std::cout << "bcfh      = " << std::bitset<22>(bishopcollisionfreehash(si, empty & BISHOPMASK[si])) << "\n";
      //BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])]
      //std::cout <<"\n----Queen/Rook:\n";
      add(si, rookthreats(si, empty) & not_us, MOVE_Q);
      //std::cout <<"\n----Queen/Bishop:\n";
      add(si, bishopthreats(si, empty) & not_us, MOVE_Q);
    }

    // Rook moves
    //std::cout << "\n------------------\nRook moves.\n";
    S = rook & us;
    while (S) {
      auto s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      auto si = ntz(s);
      //std::cout << "R si = " << si << " board[si] = '" << board[si] << "\n";
      add(si, rookthreats(si, empty) & not_us, MOVE_R);
    }

    // Bishop moves
    //std::cout << "\n------------------\nBishop moves.\n";
    S = bishop & us;
    while (S) {
      auto s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      auto si = ntz(s);
      add(si, bishopthreats(si, empty) & not_us, MOVE_B);
    }

    // Knight moves
    //std::cout << "\n------------------\nKnight moves.\n";
    S = knight & us;
    while (S) {
      auto s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      auto si = ntz(s);
      add(si, knightthreats(si) & not_us, MOVE_N);
    }

    // Pawn pushes
    //std::cout << "\n------------------\nPawn pushes.\n";
    Bitboard our_pawns = pawn & us;
    //std::cout << "Pawn pushes. our_pawns = " << Vizboard({our_pawns}) << "\n";
    //std::cout << "Pawn pushes. empty     = " << std::bitset<64>(empty) << "\n";
    Bitboard T = empty & (color ? (our_pawns << 8) : (our_pawns >> 8));
    //std::cout << "Pawn pushes. T = " << Vizboard({T}) << "\n";
    while (T) {
      Bitboard t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      Square ti = ntz(t);
      Square si = ti - (color ? 8 : -8);
      //std::cout << "pawn push si = " << si << " ti =" << ti << "\n";
      if (t & endrank) {
        add_if_legal({si, ti, MOVE_P_PROMOTE_Q, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_R, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_B, enpassant_square, castling_rights});
        add_if_legal({si, ti, MOVE_P_PROMOTE_N, enpassant_square, castling_rights});
      } else {
        add_if_legal({si, ti, MOVE_P_STANDARD, enpassant_square, castling_rights});
      }
    }

    // Pawn captures (except en passant)
    //std::cout << "\n------------------\nCaptures by pawns.\n";
    T = pawnthreats(our_pawns, color) & them;
    // std::cout << "our_pawns   = \n" << Vizboard({our_pawns}) << "\n";
    // std::cout << "pawnthreats = \n" << Vizboard({pawnthreats(our_pawns,color)}) << "\n";
    // std::cout << "T           = \n" << Vizboard({T}) << "\n";
    while (T) {
      Bitboard t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      Square ti = ntz(t);
      // std::cout << "ti = " << ti << "\n";
      // std::cout << "S recip threat = \n" << Vizboard({pawnthreats(t, !color)}) << "\n";
      S = pawnthreats(t, !color) & our_pawns;
      while (S) {
        Bitboard s = ((S ^ (S - 1)) >> 1) + 1;
        S ^= s;
        Square si = ntz(s);
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
    // std::cout << "\n------------------\nPawn double pushes.\n";
    S = our_pawns & (color ? 0x000000000000FF00UL :
                             0x00FF000000000000UL);
    T = empty & (color ? ((S << 16) & (empty << 8))
                       : ((S >> 16) & (empty >> 8)));
    while (T) {
      Bitboard t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      Square ti = ntz(t);
      Square si = ti - (color ? 16 : -16);
      add_if_legal({si, ti, MOVE_P_DOUBLEPUSH, enpassant_square, castling_rights});
    }

    // En Passant
    //std::cout << "\n------------------\nEn passant.\n";
    S = pawnthreats(enpassant_square, !color) & our_pawns;
    while (S) {
      Bitboard s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      Square si = ntz(s);
      //std::cout << "si = " << si << "\n";
      //std::cout << "ti = " << ntz(enpassant_square) << "\n";
      add_if_legal({si, ntz(enpassant_square), EN_PASSANT, enpassant_square, castling_rights});
    }

    // Kingside Castle
    //std::cout << "\n------------------\nKingside Castle.\n";
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
    //std::cout << "\n------------------\nQueenside Castle.\n";
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
    //std::cout << "\n------------------\n  legal moves done.\n--------\n";

    return result;
  }
};


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
  //if (depth == 1) return moves.size();
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
  //if (depth == 1) return moves.size();
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


int main(int argc, char * argv []) {
  ROOKTHREATS = computerookthreats();
  BISHOPTHREATS = computebishopthreats();
  auto board = Chess();
  for (int depth = 0; depth < 10; ++ depth) {
    std::cout << depth << ": " << perft(board, depth) << " " << capturetest(board, depth) << " " << enpassanttest(board, depth) << " " << checktest(board, depth) << " " << doublechecktest(board, depth) << " " << matetest(board, depth) << "\n";
  }
  // auto moves = board.legal_moves();
  // for (auto const& move : moves) {
  //  std::cout << "{" << move.s << ", " << move.t << ", " << move.flags << ", " << move.ep << ", " << move.rights << "}\n";
  // }
  // board.play({57, 40, 65536, 0, 9295429630892703873UL});
  // std::cout << "main: " << board.board << "\n";
  // for (auto const& move : board.legal_moves()) {
  //  std::cout << "{" << move.s << ", " << move.t << ", " << move.flags << ", " << move.ep << ", " << move.rights << "}\n";
  // }
  // int cnt = 0;
  // while (cnt < 10000000) {
    // std::cout << "\n\n----------------\n PLY " << board.ply << "\n";
    // std::cout << "\n\n------------------\nIntegrity check 1\n\n";
    // for (int r = 0; r < 8; ++ r) {
    //   for (int c = 0; c < 8; ++ c) {
    //     std::cout << board.board[8*r+c] << " ";
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "white:\n" << Vizboard({board.white}) << "\nblack:\n" << Vizboard({board.black}) << "\n";
    // for (Square i = 0; i < 64; ++ i) {
    //   uint64_t x = 1UL << i;
    //   if (board.board[i] == '.') {
    //     if (board.white & x) return 1;
    //     if (board.black & x) return 1;
    //   }
    //   if (board.board[i] >= 'a' && board.board[i] <= 'z') {
    //     if (board.white & x) return 1;
    //     if (!(board.black & x)) return 1;
    //   }
    //   if (board.board[i] >= 'A' && board.board[i] <= 'Z') {
    //     if (!(board.white & x)) return 1;
    //     if ((board.black & x)) return 1;
    //   }
    // }


    // std::cout << "Legal moves...\n";
    // auto moves = board.legal_moves();
    // if(moves.size() == 0) break;
    // for (auto move : moves) {
    //   if (board.board[move.s] == '.') {
    //     std::cout << "move error\n";
    //     return 1;
    //   }
    // }


    // std::cout << "\n\n------------------\nIntegrity check 2\n\n";
    // for (int r = 0; r < 8; ++ r) {
    //   for (int c = 0; c < 8; ++ c) {
    //     std::cout << board.board[8*r+c] << " ";
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "white:\n" << Vizboard({board.white}) << "\nblack:\n" << Vizboard({board.black}) << "\n";
    // for (Square i = 0; i < 64; ++ i) {
    //   uint64_t x = 1UL << i;
    //   if (board.board[i] == '.') {
    //     if (board.white & x) return 1;
    //     if (board.black & x) return 1;
    //   }
    //   if (board.board[i] >= 'a' && board.board[i] <= 'z') {
    //     if (board.white & x) return 1;
    //     if (!(board.black & x)) return 1;
    //   }
    //   if (board.board[i] >= 'A' && board.board[i] <= 'Z') {
    //     if (!(board.white & x)) return 1;
    //     if ((board.black & x)) return 1;
    //   }
    // }

    //std::cout << "Playmove.\n";
    // auto move = moves[std::rand()%moves.size()];

    // std::cout << board.board[move.s] << (char)('a' + (move.t&7)) << (8-(move.t >> 3)) << "\n";
    // board.play(move);
    // std::cout << "Played Move.\n";
    // std::cout << "  si = " << move.s << "\n";
    // std::cout << "  ti = " << move.t << "\n";
    // std::cout << "  flags = " << std::bitset<20>(move.flags) << "\n";
    // if (move.flags & CAPTURE_P_STANDARD) {
    //   std::cout << "  Pawn captured.\n";
    // }
    // if (move.flags & MOVE_Q) {
    //   std::cout << "  Queen moved.\n";
    // }


    // std::cout << "\n\n";
    // for (int r = 0; r < 8; ++ r) {
    //   for (int c = 0; c < 8; ++ c) {
    //     std::cout << board.board[8*r+c] << " ";
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "\n";


    // std::cout << cnt << "\t" << board.board << "\n\n";
  //   ++ cnt;
  // }
  // for (auto const& move : board.legal_moves()) {
  //  std::cout << "{" << move.s << ", " << move.t << ", " << move.flags << ", " << move.ep << ", " << move.rights << "}\n";
  // }
  return 0;
}
