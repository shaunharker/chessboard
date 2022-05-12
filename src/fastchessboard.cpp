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

constexpr auto ROOKMASK = SliderMask(n, s, w, e);
constexpr auto BISHOPMASK = SliderMask(nw, sw, ne, se);


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

// Given a chessboard square i and the Bitboard of empty squares
// on it's "+"-mask, this function determines those squares
// a rook or queen is "attacking".
constexpr Bitboard rookcollisionfreehash(int i, Bitboard const& E) {  //
    // E is empty squares intersected with rook "+"-mask
    auto constexpr A = antidiagonal;
    auto constexpr T = rank_8;
    auto constexpr L = file_a;
    auto X = T & (E >> (i & 0b111000));  // 3
    auto Y = modmul(A, L & (E >> (i & 0b000111))) >> 56;  // 5
    return (Y << 14) | (X << 6) | i; // 4
}

// Given a singleton bitboard x and the set of empty squares
// on it's "x"-mask, this function packages that information
// into a unique 22-bit key for lookup table access.
constexpr Bitboard bishopcollisionfreehash(int i, Bitboard const& E) {  //
  // E is empty squares intersected with bishop "X"-mask
  auto row = i >> 3;  // 1
  auto col = i & 7;  // 1
  auto t = row - col;  // 1
  auto t2 = row + col - 7 // 2
  auto OD = diagonal >> t ; // 1
  auto OA = antidiagonal << t2;  // 1
  auto constexpr L = file_a;
  auto X = modmul(L, OA&E) >> 56;  // 3
  auto Y = modmul(L, OD&E) >> 56;  // 3
  return (Y << 14) | (X << 6) | i;  // 4
}

constexpr std::array<Bitboard, (1 << 22)> computerookthreats(){
  auto result = std::array<Bitboard, (1 << 22)>();
  for (auto const& [i, x] : SquareBitboardRelation) {
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
      idx = rookcollisionfreehash(i, E);
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
    for (int k = 0x0000; k <= 0xFFFF; k += 0x0001) {
      auto E = uint64_t(0);
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << d)) ? (1UL << (8*(s-d) + d)) : 0;
      }
      for (int d = 0; d < 8; ++d) {
        E |= (k & (1 << (8+d))) ? (1UL << (8*(t+d) + d)) : 0;
      }
      // E is empty squares intersected with bishop "x"-mask possibility
      idx = bishopcollisionfreehash(i, E);
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


Bitboard const& rookthreats(int i, Bitboard const& empty) {
  return ROOKTHREATS[rookcollisionfreehash(i, empty & ROOKMASK[i])];
}

Bitboard const& bishopthreats(int i, Bitboard const& empty) {
  return BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])];
}

Bitboard const& queenthreats(int i, Bitboard const& empty) {
  return ROOKTHREATS[rookcollisionfreehash(i, empty & ROOKMASK[i])] |
    BISHOPTHREATS[bishopcollisionfreehash(i, empty & BISHOPMASK[i])];
}

Bitboard const& knightthreats(int i) {
  return KNIGHTTHREATS[i];
}

Bitboard const& kingthreats(int i) {
  return KINGTHREATS[i];
}

Bitboard pawnthreats(Bitboard const& X, bool color) { // 4
  return color ? ((X << 7) | (X << 9)) : ((X >> 7) | (X >> 9));
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
constexpr MoveFlag MOVE_P_PROMOTE_Q = 1 << 8;
constexpr MoveFlag MOVE_P_PROMOTE_R = 1 << 9;
constexpr MoveFlag MOVE_P_PROMOTE_B = 1 << 10;
constexpr MoveFlag MOVE_P_PROMOTE_N = 1 << 11;
constexpr MoveFlag MOVE_Q = 1 << 12;
constexpr MoveFlag MOVE_R = 1 << 13;
constexpr MoveFlag MOVE_B = 1 << 14;
constexpr MoveFlag MOVE_N = 1 << 15;
constexpr MoveFlag MOVE_K_STANDARD = 1 << 16;
constexpr MoveFlag MOVE_K_KINGSIDE_CASTLE = 1 << 17;
constexpr MoveFlag MOVE_K_QUEENSIDE_CASTLE = 1 << 18;

constexpr auto MOVED = MOVE_P_STANDARD |
                       MOVE_P_DOUBLEPUSH |
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
  Bitboard pawn;
  Bitboard queen;
  Bitboard rook;
  Bitboard bishop;
  Bitboard knight;
  Bitboard enpassant_square; // behind double-pushed pawn
  Bitboard castling_rights; // K/Q/k/q ~ h1/a1/h8/a8
  char board[64];
  uint64_t ply;
  uint64_t halfmove_clock;

public:

  Chess() : board("rnbqkbnrpppppppp................PPPPPPPPRNBQKBNR") {
    white =            0xFFFF000000000000;  // rank_1 | rank_2;
    black =            0x000000000000FFFF;  // rank_7 | rank_8;
    king =             0x1000000000000010;  // e1 | e8;
    bishop =           0x2400000000000024;  // c1 | c8 | f1 | f8;
    rook =             0x8100000000000081;  // a1 | a8 | h1 | h8;
    queen =            0x0800000000000008;  // d1 | d8
    knight =           0x4200000000000042;  // b1 | b8 | g1 | g8;
    pawn =             0x00FF00000000FF00;  // rank_2 | rank_7
    enpassant_square = 0x0000000000000000;  // behind double-pushed pawn
    castling_rights =  0x8100000000000081;  // a1 | a8 | h1 | h8
    ply = 0;
    halfmove_clock = 0;
  }

  void
  playmove(Move const& m) {
    auto const& [s, t, flags] = m;

    bool color = ply & 1;
    Bitboard const& us = color ? black : white;
    Bitboard const& them = color ? white : black;

    auto st = s | t;
    ply += 1;

    auto si = ntz(s);
    auto ti = ntz(t);
    board[ti] = board[si];
    board[si] = '.';

    if (flags & CAPTURED) {
      halfmove_clock = 0;
    } else {
      halfmove_clock += 1;
    }

    enpassant_square &= color ? rank_3 : rank_6;

    switch (flags & CAPTURED) {
      case CAPTURE_P_STANDARD:
        pawn ^= t;
        them ^= t;
        break;
      case CAPTURE_P_EN_PASSANT:
        auto ci = ti + (color ? -8 : 8);
        board[ci] = '.';
        auto captured = 1UL << ci;
        pawn ^= st | captured;
        them ^= captured;
        break;
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
        break;
      case MOVE_K_KINGSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x00000000000000F0UL
                    : 0xF000000000000000UL;
        castling_rights &= color ? rank_1 : rank_8;
        break;
      case MOVE_K_QUEENSIDE_CASTLE:
        king ^= st;
        rook ^= color ? 0x00000000000000A0UL
                      : 0xA000000000000000UL;
        us ^= color ? 0x0900000000000009UL
                    : 0x0000000000000009UL;
        castling_rights &= color ? rank_1 : rank_8;
        break;
      case MOVE_P_STANDARD:
        pawn ^= st;
        us ^= st;
        break;
      case MOVE_P_DOUBLEPUSH:
        pawn ^= st;
        us ^= st;
        enpassant_square = t << (color ? -8 : 8);
        break;
      case MOVE_P_PROMOTE_Q:
        pawn ^= s;
        queen ^= t;
        us ^= st;
        board[ti] += ('q'-'p')
        break;
      case MOVE_P_PROMOTE_N:
        pawn ^= s;
        knight ^= t;
        us ^= st;
        board[ti] += ('n'-'p')
        break;
      case MOVE_P_PROMOTE_R:
        pawn ^= s;
        rook ^= t;
        us ^= st;
        board[ti] += ('r'-'p')
        break;
      case MOVE_P_PROMOTE_B:
        pawn ^= s;
        bishop ^= t;
        us ^= st;
        board[ti] += ('b'-'p')
        break;
      case MOVE_Q:
        queen ^= st;
        us ^= st;
        break;
      case MOVE_R:
        rook ^= st;
        castling_rights &= ~st;
        us ^= st;
        break;
      case MOVE_B:
        bishop ^= st;
        us ^= st;
        break;
      case MOVE_N:
        knight ^= st;
        us ^= st;
        break;
    }
  }

  Bitboard attackers(int i) {
    bool color = ply & 1;
    Bitboard const& us = color ? black : white;
    Bitboard const& them = color ? white : black;
    auto empty = ~(us | them);
    return them & (
      (kingthreats(i) & king)                     |
      (knightthreats(i) & knight)                 |
      (bishopthreats(i, empty) & bishop_or_queen) |
      (rookthreats(i, empty) & rook_or_queen)     |
      (pawnthreats(i, color) & pawn));
  }

  std::vector<Move> legal_moves() {
    bool color = ply & 1;
    Bitboard const& us = color ? black : white;
    Bitboard const& them = color ? white : black;
    Bitboard const& endrank = color ? rank_1 : rank_8;
    auto our_king = king & us;
    auto empty = ~(us | them);
    auto not_us = ~us;

    std::vector<Move> result;

    auto add_if_legal = [&](Move const& m) {
      auto const& [s, t, flags] = m;
      // TODO: OPTIMIZE THIS, INTRODUCE "UNDO MOVE", reversible stack of moves
      // tree search without copying
      auto clone = *this;  // code smell: shouldn't copy new object
      auto clone = playmove(m);
      auto our_maybe_moved_king = (s & king) ? t : our_king;
      if (clone.attackers(ntz(our_maybe_moved_king)) == 0) {
        result.push_back(m);
      }
    };

    auto add = [&](int si, Bitboard T, MoveFlag flags){
      while (T) {
        t = ((T ^ (T - 1)) >> 1) + 1;
        T ^= t;
        auto ti = ntz(t);
        switch(board[ti]) {
          case '.':
            add_if_legal({si, ti, flags});
            break;
          case 'Q':
            add_if_legal({si, ti, CAPTURE_Q | flags});
            break;
          case 'q':
            add_if_legal({si, ti, CAPTURE_Q | flags});
            break;
          case 'R':
            add_if_legal({si, ti, CAPTURE_R | flags});
            break;
          case 'r':
            add_if_legal({si, ti, CAPTURE_R | flags});
            break;
          case 'B':
            add_if_legal({si, ti, CAPTURE_B | flags});
            break;
          case 'b':
            add_if_legal({si, ti, CAPTURE_B | flags});
            break;
          case 'N':
            add_if_legal({si, ti, CAPTURE_N | flags});
            break;
          case 'n':
            add_if_legal({si, ti, CAPTURE_N | flags});
            break;
          case 'P':
            add_if_legal({si, ti, CAPTURE_P | flags});
            break;
          case 'p':
            add_if_legal({si, ti, CAPTURE_P | flags});
            break;
        }
      }
    };

    // Pseudolegal King moves
    auto oki = ntz(our_king);
    add(oki, kingthreats(oki) & not_us, MOVE_K_STANDARD);


    auto S = queen & us;
    while (S) {
      auto x = ((S ^ (S - 1)) >> 1) + 1;
      S ^= x;
      auto i = ntz(x);
      add(i, rookthreats(i) & not_us, MOVE_Q);
      add(i, bishopthreats(i) & not_us, MOVE_Q);
    }

    S = rook & us;
    while (S) {
      auto x = ((S ^ (S - 1)) >> 1) + 1;
      S ^= x;
      auto i = ntz(x);
      add(x, rookthreats(x) & not_us, MOVE_R);
    }

    S = bishop & us;
    while (S) {
      auto x = ((S ^ (S - 1)) >> 1) + 1;
      S ^= x;
      auto i = ntz(x);
      add(x, bishopthreats(x) & not_us, MOVE_B);
    }


    S = knight & us;
    while (S) {
      auto x = ((S ^ (S - 1)) >> 1) + 1;
      S ^= x;
      auto i = ntz(x);
      add(x, knightthreats(x) & not_us, MOVE_N);
    }

    // Pseudolegal Pawn pushes
    auto our_pawns = pawn & us;
    int pp = color ? 8 : -8; // single pawn push offset
    auto T = (our_pawns << pp) & empty;
    while (T) {
      auto t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      auto s = t >> pp;
      auto si = ntz(s);
      auto ti = si + pp;

      if (y & endrank) {
        add_if_legal({si, ti, MOVE_P_PROMOTE_Q});
        add_if_legal({si, ti, MOVE_P_PROMOTE_R});
        add_if_legal({si, ti, MOVE_P_PROMOTE_B});
        add_if_legal({si, ti, MOVE_P_PROMOTE_N});
      } else {
        add_if_legal({si, ti, MOVE_P_STANDARD});
      }
    }

    // Pseudolegal Pawn captures
    T = pawnthreats(our_pawns, color) & them;
    while (T) {
      auto t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      auto ti = ntz(t);
      S = pawnthreats(t, ~color) & our_pawns;
      while (S) {
        s = ((S ^ (S - 1)) >> 1) + 1;
        S ^= s;
        auto si = ntz(s);
        if (y & endrank) {
          add_if_legal({si, ti, MOVE_P_PROMOTE_Q});
          add_if_legal({si, ti, MOVE_P_PROMOTE_R});
          add_if_legal({si, ti, MOVE_P_PROMOTE_B});
          add_if_legal({si, ti, MOVE_P_PROMOTE_N});
        } else {
          add_if_legal({si, ti, MOVE_P_STANDARD});
        }
      }
    }

    // Pseudolegal Double Pawn pushes
    auto dpp = pp << 1; // double pawn push offset
    S = our_pawns & (color ? 0x000000000000FF00UL :
                             0x00FF000000000000UL);
    T = (S << dpp) & (empty << pp) & empty;
    while (T) {
      auto t = ((T ^ (T - 1)) >> 1) + 1;
      T ^= t;
      auto s = t >> dpp;
      auto si = ntz(s);
      auto ti = si + dpp;
      add_if_legal({si, ti, DOUBLEPUSH});
    }

    // Pseudolegal En Passant
    S = pawnthreats(enpassant_square, ~color) & our_pawns;
    while (S) {
      auto s = ((S ^ (S - 1)) >> 1) + 1;
      S ^= s;
      auto si = ntz(s);
      add_if_legal({ntz(s), ntz(enpassant_square), EN_PASSANT});
    }

    // Legal Kingside Castle
    if (castling_rights & (color ? (1UL << 7) : (1UL << 63))) {
      auto conf = (color ? 240UL : (240UL << 56));
      if ((us & conf) == (color ? 144UL : (144UL << 56))) {
        if ((empty & conf) == (color ? 96UL : (96UL << 56))) {
          if ((attackers(oki + 0) == 0) &&
              (attackers(oki + 1) == 0) &&
              (attackers(oki + 2) == 0)) {
              result.push_back({oki, oki + 2, KINGSIDE_CASTLE});
          }
        }
      }
    }

    // Legal Queenside Castle
    if (castling_rights & (color ? (1UL << 0) : (1UL << 56))) {
      auto conf = (color ? 31UL : (31UL << 56));
      if ((us & conf) == (color ? 17UL : (17UL << 56))) {
        if ((empty & conf) == (color ? 14UL : (14UL << 56))) {
          if ((attackers(oki - 0) == 0) &&
              (attackers(oki - 1) == 0) &&
              (attackers(oki - 2) == 0)) {
              result.push_back({oki, oki - 2, QUEENSIDE_CASTLE});
          }
        }
      }
    }

    return result;
  }
};

int main(int argc, char * argv []) {
  constexpr auto g = newgame();
  auto moves = legal_moves(g);
  for (auto const& move : moves) {
    std::cout << ntz(move.source) << " " << ntz(move.target) << " " << move.flags << "\n";
  }
  return 0;
}
