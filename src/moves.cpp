// moves.cpp
// Shaun Harker
// BSD ZERO CLAUSE LICENSE

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

// Part I. Precomputing Some Tables
// non-lazy constexpr versions of range, map, and enumerate
template <uint64_t N> constexpr auto range() {
    std::array<uint64_t, N> result {};
    for(uint64_t i = 0; i < N; ++i) result[i] = i;
    return result;
}
template <typename F, typename S> constexpr auto map(F func, S seq) {
    typedef typename S::value_type value_type;
    using return_type = decltype(func(std::declval<value_type>()));
    std::array<return_type, std::tuple_size<S>::value> result {};
    uint64_t i = 0;
    for (auto x : seq) result[i++] = func(x);
    return result;
}
template <typename S> constexpr auto enumerate(S seq) {
    typedef typename S::value_type value_type;
    std::array<std::pair<uint64_t, value_type>, std::tuple_size<S>::value> result {};
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
struct Vizboard {Bitboard x;};
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
constexpr auto twopow = [](Square x){return Bitboard(1) << x;};
constexpr uint8_t popcount(Bitboard x) {
  // number of one bits
  return __builtin_popcountll(x);
}
constexpr uint8_t nlz(Bitboard x) {
  // number of leading zeros
  return __builtin_clzll(x);
}
constexpr uint8_t ntz(Bitboard x) {
  // number of trailing zeros
  return __builtin_ctzll(x);
}
constexpr auto SquareBitboardRelation = enumerate(map(twopow, squares));

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

// a fistful of hash functions
constexpr Bitboard rookcollisionfreehash(Square i, Bitboard const& E) {
    // Given a chessboard square i and the Bitboard of empty squares
    // on it's "+"-mask, this function determines those squares
    // a rook or queen is "attacking".
    // E is empty squares intersected with rook "+"-mask
    auto constexpr A = antidiagonal;
    auto constexpr T = rank_8;
    auto constexpr L = file_a;
    auto X = T & (E >> (i & 0b111000));  // 3
    auto Y = (A * (L & (E >> (i & 0b000111)))) >> 56;  // 5
    return (Y << 14) | (X << 6) | i; // 4
}
constexpr Bitboard bishopcollisionfreehash(Square i, Bitboard const& E) {
  // Given a singleton bitboard x and the set of empty squares
  // on it's "x"-mask, this function packages that information
  // into a unique 22-bit key for lookup table access.
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
constexpr uint8_t bitreverse8(uint8_t x) {
  return __builtin_bitreverse8(x);
}
template <uint8_t row, uint8_t col> constexpr uint8_t e_hash (Bitboard x) {
    return (x >> (8*row+col+1)) & rank_8;
};
template <uint8_t row, uint8_t col> constexpr uint8_t n_hash (Bitboard x) {
    return (antidiagonal * (file_a & (x >> (8*row+col+8)))) >> 56;
};
template <uint8_t row, uint8_t col> constexpr uint8_t w_hash (Bitboard x) {
    return bitreverse8(((x >> (8*row)) << (8-col)) & rank_8);
};
template <uint8_t row, uint8_t col> constexpr uint8_t s_hash (Bitboard x) {
    // there is probably slightly faster magic
    return bitreverse8((antidiagonal * (file_a & (x << (8*(8-row)-col)))) >> 56);
};
template <uint8_t row, uint8_t col> constexpr Bitboard nwse_diagonal() {
  if constexpr (row > col) {
    return diagonal >> (8*(row-col));
  } else {
    return diagonal << (8*(col-row));
  }
}
template <uint8_t row, uint8_t col> constexpr Bitboard swne_diagonal() {
  if constexpr (row + col < 7) {
    return antidiagonal << (8*(7 - row+col));
  } else {
    return antidiagonal >> (8*(row+col - 7));
  }
}
template <uint8_t row, uint8_t col> constexpr uint8_t nw_hash (Bitboard x) {
    constexpr Bitboard this_diagonal = nwse_diagonal<row,col>();
    return bitreverse8((((file_a * (x & this_diagonal)) >> 56) << (8-col)) & rank_8);
};
template <uint8_t row, uint8_t col> constexpr uint8_t ne_hash (Bitboard x) {
    constexpr Bitboard this_diagonal = swne_diagonal<row,col>();
    return (((file_a * (x & this_diagonal)) >> 56) >> (col+1)) & rank_8;
};
template <uint8_t row, uint8_t col> constexpr uint8_t sw_hash (Bitboard x) {
    constexpr Bitboard this_diagonal = swne_diagonal<row,col>();
    return bitreverse8((((file_a * (x & this_diagonal)) >> 56) << (8-col)) & rank_8);
};
template <uint8_t row, uint8_t col> constexpr uint8_t se_hash (Bitboard x) {
    constexpr Bitboard this_diagonal = nwse_diagonal<row,col>();
    return (((file_a * (x & this_diagonal)) >> 56) << (col+1)) & rank_8;
};

// table computations
std::array<std::pair<uint8_t,uint8_t>, (1 << 21)> compute_cap() {
    // compute checks and pins to the right,
    // that is, least sig bits are closer to king
    // 0x00-0x07 0x08-0x0E 0x0F-0x014
    // slider    us        them
    std::array<std::pair<uint8_t,uint8_t>, (1 << 21)> result {};
    for (uint32_t x = 0; x < (1 << 21); ++ x) {
        uint8_t slider = x & 0x7F;
        uint8_t us = (x >> 7) & 0x7F;
        uint8_t them = (x >> 14) & 0x7F;
        uint8_t checker;
        if (slider & them) {
          checker = ntz(slider & them);
        } else {
          result[x] = {0,0};
          continue;
        }
        uint8_t front = (1 << checker) - 1;
        if (them & front) {
          result[x] = {0,0};
          continue;
        }
        uint8_t pcnt = popcount(us & front);
        switch (pcnt) {
            case 0:
                result[x] = {checker, 0};
                break;
            case 1:
                result[x] = {checker, ntz(us)};
                break;
            default:
                result[x] = {0,0};
                break;
        }
    }
    return result;
}
std::array<std::pair<uint8_t,uint8_t>, (1 << 21)> CAP = compute_cap();
template<typename T> constexpr Bitboard slide(Bitboard x, T f) {
  // T stands for "translation" and we expect it to be a
  // uint64_t(uint64_t) function type that translates bitboards.
  //
  // Rooks, Bishops, and Queens are "slider" pieces,
  // and we need to generate masks showing how they
  // can move, starting from any square.
  // Thus, we use the following code to take a bitboard
  // and repeatedly apply a slide translation to it
  // until it becomes zero (indicating the piece(s) have
  // all slid off the board) and OR the results.
  return f(x) | ((f(x) == 0) ? 0 : slide(f(x), f));
}
template<typename... Ts> constexpr std::array<uint64_t, 64>
SliderMask(Ts... args) {
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
std::vector<Bitboard> computerookthreats(){
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
std::vector<Bitboard> ROOKTHREATS = computerookthreats();
std::vector<Bitboard> computebishopthreats(){
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
std::vector<Bitboard> BISHOPTHREATS = computebishopthreats();
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

// threat queries
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

// Part II. The Move Representations
enum Piece {
  SPACE = 0,
  PAWN = 1,
  KNIGHT = 2,
  BISHOP = 3,
  ROOK = 4,
  QUEEN = 5,
  KING = 6
};

constexpr std::string_view GLYPHS(".PNBRQK");

struct ThreeByteMove {
    uint32_t X;
    //  bit range  | mode0 |
    // 0x17        | c     | ply % 2
    // 0x14 - 0x16 | sp    | source piece enum index
    // 0x11 - 0x13 | sr    | source row 87654321
    // 0x0E - 0x10 | sc    | source col abcdefgh
    // 0x0B - 0x0D | tr    | target row 87654321
    // 0x08 - 0x0A | tc    | target col abcdefgh
    // 0x05 - 0x07 | tp    | target piece enum index
    // 0x02 - 0x04 | cp    | capture piece enum index
    // 0x01        | qcr   | change queen castling rights
    // 0x00        | kcr   | change king castling rights
    // note: for Ra1xRa8, Ra8xRa1, Rh1xRh8, Rh8xRh1
    //       qcr and kcr instead are bcr and wcr, which
    //       indicate which colors lose castling rights
    //       the side of which can be inferred from the
    //       RxR move.
    constexpr ThreeByteMove ():X(0){};
    constexpr ThreeByteMove (uint32_t X):X(X){};
    constexpr bool c() const {return X & 0x800000;}
    constexpr uint8_t sp() const {return (X >> 0x14) & 0x07;}
    constexpr uint8_t si() const {return (X >> 0x0E) & 0x3F;}
    constexpr uint8_t ti() const {return (X >> 0x08) & 0x3F;}
    constexpr uint8_t tp() const {return (X >> 0x05) & 0x07;}
    constexpr uint8_t cp() const {return (X >> 0x02) & 0x07;}
    constexpr bool wkrr() const {
      return (sp() == ROOK) && (cp() == ROOK) && (
       (si() == 63) && (ti() == 7)) && !c();
    }
    constexpr bool bkrr() const {
      return (sp() == ROOK) && (cp() == ROOK) && (
       (si() == 7) && (ti() == 63)) && c();
    }
    constexpr bool wqrr() const {
      return (sp() == ROOK) && (cp() == ROOK) && (
       (si() == 56) && (ti() == 0)) && !c();
    }
    constexpr bool bqrr() const {
      return (sp() == ROOK) && (cp() == ROOK) && (
       (si() == 0) && (ti() == 56)) && c();
    }
    constexpr bool rr() const {
      return wkrr() || bkrr() || wqrr() || bqrr();
    }
    constexpr bool qcr() const {
      // in the rr() case we may compute it:
      if (rr()) return ((X & 0x03) != 0) && (wqrr() || bqrr());
      // Otherwise it is stored in the 0x02 bit.
      return X & 0x02;
    }
    constexpr bool kcr() const {
      // in the rr() case we may compute it:
      if (rr()) return ((X & 0x03) !=0) && (wkrr() || bkrr());
      // Otherwise it is stored in the 0x01 bit.
      return X & 0x01;
    }
    constexpr bool wcr() const {
      bool whitemove = !c();
      // in the rr() case this is stored directly:
      if (rr()) return X & 0x01;
      // if no rights are lost then false:
      if (!kcr() && !qcr()) return false;
      // if both rights are lost then it must
      // be a castling or a king move; just
      // return whose turn it is:
      if (kcr() && qcr()) return whitemove;
      // a move from the king home square
      // never removes the enemies castling right
      // so, the removed right must be white's in
      // that case:
      if (si() == 60) return true;
      // we know the move isn't RxR, so:
      if ((sp() == ROOK) && ((si() == 56) || (si() == 63)))
        return true;
      // we know the move isn't RxR, so:
      if ((cp() == ROOK) && ((ti() == 56) || (ti() == 63)))
        return true;
      // still here?
      return false;
    }
    constexpr bool bcr() const {
      bool whitemove = !c();
      // in the rr() case this is stored directly:
      if (rr()) return X & 0x02;
      // if no rights are lost then false:
      if (!kcr() && !qcr()) return false;
      // if both rights are lost then it must
      // be a castling or a king move; just
      // return whose turn it is:
      if (kcr() && qcr()) return !whitemove;
      // a move from the king home square
      // never removes the enemies castling right
      // so, the removed right must be black's in
      // that case:
      if (si() == 4) return true;
      // we know the move isn't RxR, so:
      if ((sp() == ROOK) && ((si() == 0) || (si() == 7)))
        return true;
      // we know the move isn't RxR, so:
      if ((cp() == ROOK) && ((ti() == 0) || (ti() == 7)))
        return true;
      // still here?
      return false;
    }
    constexpr uint64_t s() const {return 1UL << si();}
    constexpr uint64_t t() const {return 1UL << ti();}
    constexpr uint64_t st() const {return s() | t();}
    constexpr uint64_t ui() const {return (tc() << 3) | sr();}
    constexpr uint64_t u() const {return 1UL << ui();}
    constexpr uint8_t sr() const {return si() >> 3;}
    constexpr uint8_t sc() const {return si() & 0x07;}
    constexpr uint8_t tr() const {return ti() >> 3;}
    constexpr uint8_t tc() const {return ti() & 0x07;}
    constexpr bool ep() const {
        return (sp() == PAWN) && (sc() != tc()) && (cp() == SPACE);
    }
    constexpr bool x() const {return (cp() != SPACE) || ep();}
    constexpr bool wkcr() const {return wcr() && kcr();}
    constexpr bool bkcr() const {return bcr() && kcr();}
    constexpr bool wqcr() const {return wcr() && qcr();}
    constexpr bool bqcr() const {return bcr() && qcr();}
    constexpr bool feasible() const {
        // determine if move could happen in play
        bool whitemove = !c();

        // sp must name a glyph
        if (sp() > 6) return false;

        // can't move from an empty square
        if (sp() == SPACE) return false;

        // tp must name a glyph
        if (tp() > 6) return false;

        // tp can't name SPACE
        if (tp() == SPACE) return false;

        // cp must name a glyph
        if (cp() > 6) return false;

        // cp may not name KING
        if (cp() == KING) return false;

        // source != target
        if ((sc() == tc()) && (sr() == tr())) return false;

        // only pawns promote, and it must be properly positioned
        if ((sp() != tp()) && ((sr() != (whitemove ? 1 : 6)) || (tr() != (whitemove ? 0 : 7)) || (sp() != PAWN))) return false;

        // pawns can't promote to space, pawns, or kings
        if ((sp() != tp()) && ((tp() == SPACE) ||
            (tp() == PAWN) || (tp() == KING))) return false;

        // pawns are never on rank 8 or rank 1 (row 0 or row 7)
        if ((sp() == PAWN) && ((sr() == 0) ||
            (sr() == 7))) return false;
        if ((tp() == PAWN) && ((tr() == 0) ||
            (tr() == 7))) return false;
        if ((cp() == PAWN) && ((tr() == 0) ||
            (tr() == 7))) return false;

        if (sp() == PAWN) {
            // pawns can only move forward one rank at a time,
            // except for their first move
            if (sr() != tr() + (whitemove ? 1 : -1)) {
                if ((sr() != (whitemove ? 6 : 1)) ||
                    (tr() != (whitemove ? 4 : 3))) return false;
                // can't capture on double push
                if (cp() != SPACE) return false;
            }
            // pawns stay on file when not capturing,
            // and move over one file when capturing.
            // i) can't move over more than one file
            if (sc()*sc() + tc()*tc() > 1 + 2*sc()*tc()) return false;
            // ii) can't capture forward
            if ((sc() == tc()) && (cp() != SPACE)) return false;
            // iii) can't move diagonal without capture
            if ((sc() != tc()) && (cp() == SPACE)) {
                // invalid unless possible en passant
                if (tr() != (whitemove ? 2 : 5)) return false;
            }
        }

        if (sp() != tp()) {
            // can only promote on the endrank
            if (tr() != (whitemove ? 0 : 7)) return false;
            // can only promote to N, B, R, Q
            if ((tp() == SPACE) || (tp() == PAWN) ||
                (tp() == KING)) return false;
        }

        if (sp() == KNIGHT) {
            // i know how horsies move
            if ((sc()*sc() + tc()*tc() + sr()*sr() + tr()*tr())
                != 5 + 2*(sc()*tc() + sr()*tr())) return false;
        }
        if (sp() == BISHOP) {
            // bishops move on diagonals and antidiagonals
            if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
                    (sc() + tr() != tc() + sr())) // not on same diagonal
                return false;
        }
        if (sp() == ROOK) {
            // rooks move on ranks and files (rows and columns)
            if ((sc() != tc()) && (sr() != tr())) return false;
            // conditions where kingside castle right may change
            if (kcr() && !((sc() == 7) && (sr() == (whitemove ? 7 : 0))) && !((tc() == 7) && (tr() == (whitemove ? 0 : 7)))) return false;
            // if losing kingside rights, cannot move to a rook to files a-e
            if (kcr() && (tc() < 5)) return false;
            // conditions where queenside castle right may change
            if (qcr() && !((sc() == 0) && (sr() == (whitemove ? 7 : 0))) && !((tc() == 0) && (tr() == (whitemove ? 0 : 7)))) return false;
            // if losing queenside rights, cannot move a rook to files e-h
            if (qcr() && ((tc() > 3))) return false;
        }
        if (sp() == QUEEN) {
            // queens move on ranks, files, diagonals, and
            // antidiagonals.
            if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
                    (sc() + tr() != tc() + sr()) && // not on same diagonal
                    (sc() != tc()) && // not on same file
                    (sr() != tr())) // not on same rank
                return false;
            if ((sc() == tc()) && (sr() == tr())) return false;
        }
        if (sp() == KING) {
            // if kingside castle, must be losing kingside rights
            if ((sc() == 4) && (sr() == (whitemove ? 7 : 0)) && (tc() == 6) && (tr() == (whitemove ? 7 : 0)) && !kcr()) return false;
            // if queenside castle, must be losing queenside rights
            if ((sc() == 4) && (sr() == (whitemove ? 7 : 0)) && (tc() == 2) && (tr() == (whitemove ? 7 : 0)) && !qcr()) return false;
            // king takes rook losing castling rights:
            //   only diagonal/antidiagonal captures could
            //   possibly occur during play:
            if ((cp() == ROOK) && kcr() && (tr() == (whitemove ? 0 : 7)) && (tc() == 7) && !((sr() == (whitemove ? 1 : 6)) && (sc() == 6))) return false;
            if ((cp() == ROOK) && qcr() && (tr() == (whitemove ? 0 : 7)) && (tc() == 0) && !((sr() == (whitemove ? 1 : 6)) && (sc() == 1))) return false;
            // castling cannot capture, must be properly positioned
            if (sc()*sc() + tc()*tc() > 1 + 2*sc()*tc()) {
                if (!((tc() == 6) && kcr()) && !((tc() == 2) && qcr())) return false;
                if (cp() != SPACE) return false;
                if (sc() != 4) return false;
                if (sr() != (whitemove ? 7 : 0)) return false;
                if (tr() != (whitemove ? 7 : 0)) return false;
            }
            // kings move to neighboring squares
            if (((sc()*sc() + tc()*tc() + sr()*sr()) + tr()*tr() >
                2*(1 + sc()*tc() + sr()*tr())) && !((sc() == 4) &&
                (sr() == (whitemove ? 7 : 0)) && (tr()==sr()) &&
                (((tc()==2) && qcr()) || ((tc()==6) && kcr()))))
                return false;
        }
        // to change castling rights there are nine cases:
        // 1. king move from its home square
        // 2. a1 rook captured by black
        // 3. h1 rook captured by black
        // 4. a8 rook captured by white
        // 5. h8 rook captured by white
        // 6. a1 rook moved by white
        // 7. h1 rook moved by white
        // 8. a8 rook moved by black
        // 9. h8 rook moved black
        // White could capture an unmoved a8 rook with its unmoved a1 rook,
        // and similar scenarios, so the cases aren't mutually exclusive.
        // it isn't possible to remove castling rights via a Rf8 x Rh8 because the enemy king would be in check. Similarly for other exceptions
        bool kingmove = (sp() == KING) && (sr() == (whitemove ? 7 : 0)) && (sc() == 4);
        bool a1rookcapture = (cp() == ROOK) && (ti() == 56) && !whitemove;
        bool a8rookcapture = (cp() == ROOK) && (ti() == 0) && whitemove;
        bool h1rookcapture = (cp() == ROOK) && (ti() == 63) && !whitemove;
        bool h8rookcapture = (cp() == ROOK) && (ti() == 7) && whitemove;

        bool a1rookmove = (sp() == ROOK) && (si() == 56) && whitemove && (tc() < 4);
        bool a8rookmove = (sp() == ROOK) && (si() == 0) && !whitemove && (tc() < 4);
        bool h1rookmove = (sp() == ROOK) && (si() == 63) && whitemove && (tc() > 4);
        bool h8rookmove = (sp() == ROOK) && (si() == 7) && !whitemove && (tc() > 4);
        if (kcr() && !(kingmove || h1rookmove || h8rookmove)) {
            if (h1rookcapture || h8rookcapture) {
                // exclude moves implying a king is en prise
                if ((sp() == ROOK) && (sc() < 6)) return false;
                if ((sp() == QUEEN) && (sr() == tr()) && (sc() < 6)) return false;
                if ((sp() == KING) && ((sr() == tr()) || (sc() == tc()))) return false;
            } else {
                return false;
            }
        }
        if (qcr() && !(kingmove || a1rookmove || a8rookmove)) {
            if (a1rookcapture || a8rookcapture) {
                // exclude moves implying a king is en prise
                if ((sp() == ROOK) && (sc() > 2)) return false;
                if ((sp() == QUEEN) && (sr() == tr()) && (sc() > 2)) return false;
                if ((sp() == KNIGHT) && (sr() == (whitemove ? 1 : 6)) && (sc() == 2)) return false;
                if ((sp() == KING) && ((sr() == tr()) || (sc() == tc()))) return false;
            } else {
                    return false;
            }
        }
        return true;
    }
};

// PART III. The Position
struct Position {
    Bitboard pawn;
    Bitboard knight;
    Bitboard bishop;
    Bitboard rook;
    Bitboard queen;
    Bitboard king;
    Bitboard white;
    Bitboard black;
    uint8_t rights;
    //char board[65];
    Position() {
        white = 0xFFFF000000000000; // rank_1 | rank_2;
        black = 0x000000000000FFFF; // rank_7 | rank_8;
        king = 0x1000000000000010; // e1 | e8;
        pawn = 0x00FF00000000FF00; // rank_2 | rank_7
        queen = 0x0800000000000008; // d1 | d8
        rook = 0x8100000000000081; // a1 | a8 | h1 | h8;
        bishop = 0x2400000000000024; // c1 | c8 | f1 | f8;
        knight = 0x4200000000000042; // b1 | b8 | g1 | g8;
        rights = 0x00; // move, castling rights
    }
    constexpr bool c() const {return rights & 1;}
    constexpr bool wkcr() const {return rights & 2;}
    constexpr bool wqcr() const {return rights & 4;}
    constexpr bool bkcr() const {return rights & 8;}
    constexpr bool bqcr() const {return rights & 16;}
    constexpr bool epi() const {
      return (c() ? (2 << 5) : (5 << 5)) | (rights >> 5);
    }
    void play(Position rhs) {
        pawn ^= rhs.pawn;
        knight ^= rhs.knight;
        bishop ^= rhs.bishop;
        rook ^= rhs.rook;
        queen ^= rhs.queen;
        king ^= rhs.king;
        white ^= rhs.white;
        black ^= rhs.black;
        rights ^= rhs.rights;
    }
    void play(ThreeByteMove const& tbm) {
        auto c = tbm.c();
        auto si = tbm.si();
        auto ti = tbm.ti();
        auto sc = tbm.sc();
        auto tc = tbm.tc();
        auto ui = tbm.ui();
        auto sp = tbm.sp();
        auto tp = tbm.tp();
        auto cp = tbm.cp();
        auto wkcr = tbm.wkcr();
        auto bkcr = tbm.bkcr();
        auto wqcr = tbm.wqcr();
        auto bqcr = tbm.bqcr();
        uint64_t s = tbm.s();
        uint64_t t = tbm.t();
        //uint64_t u = tbm.u();

        bool whitemove = !c;
        uint64_t & us = whitemove ? white : black;
        uint64_t & them = whitemove ? black : white;

        rights ^= (wkcr ? 0x02 : 0x00) | (wqcr ? 0x04 : 0x00) |
                  (bkcr ? 0x08 : 0x00) | (bqcr ? 0x10 : 0x00) |
                  (0x01);
        us ^= s;
        //auto bsp = whitemove ? sp : (sp + 32);
        //board[si] ^= (SPACE ^ bsp);

        switch (sp) {
            case PAWN: pawn ^= s; break;
            case KNIGHT: knight ^= s; break;
            case BISHOP: bishop ^= s; break;
            case ROOK: rook ^= s; break;
            case QUEEN: queen ^= s; break;
            case KING: king ^= s; break;
        }

        switch (cp) {
            case PAWN: pawn ^= t; break;
            case KNIGHT: knight ^= t; break;
            case BISHOP: bishop ^= t; break;
            case ROOK: rook ^= t; break;
            case QUEEN: queen ^= t; break;
        }

        us ^= t;
        if (cp != SPACE) {
          //auto bcp = whitemove ? (cp + 32) : cp;
          //board[ti] ^= bcp;
          them ^= t;
        }

        switch (tp) {
            case PAWN: pawn ^= t; break;
            case KNIGHT: knight ^= t; break;
            case BISHOP: bishop ^= t; break;
            case ROOK: rook ^= t; break;
            case QUEEN: queen ^= t; break;
        }

        //auto btp = whitemove ? tp : (tp + 32);
        //board[ti] ^= btp;

        if ((sp == PAWN) && (tp == PAWN) &&
                (cp == SPACE) && (sc != tc)) {
            // en passant capture
            pawn ^= tbm.u();
            them ^= tbm.u();
            //board[ui] ^= (SPACE ^ (whitemove ? SPACE : PAWN));
        }

        if ((sp == KING) && (tc == sc + 2)) {
            rook ^= whitemove ? 0xA000000000000000 :
                            0x00000000000000A0;
            us ^= whitemove ? 0xA000000000000000 :
                          0x00000000000000A0;
        }

        if ((sp == KING) && (tc + 2 == sc)) {
            rook ^= whitemove ? 0x0900000000000000 :
                            0x0000000000000009;
            us ^= whitemove ? 0x0900000000000000 :
                          0x0000000000000009;
        }
    }
    void legal_moves(uint16_t *out, uint8_t *moves_written) {

        bool color = c();

        std::vector<ThreeByteMove> moves {};

        void add_move_s_t(bool c, bool pr, Piece sp,
            uint8_t si, uint8_t ti, uint_t flag) {
            // todo { 1 }
        }

        void add_move_s_T(bool c, bool pr, Piece sp,
            uint8_t si, Bitboard T, uint_t flag) {
            // todo { 2 }
        }

        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;
        bool check(uint8_t si) {
            // without removing king from board, determine if si square is chked
            constexpr auto translations = {-8, 8, -1, 1, -9, -7, 7, 9};
            constexpr auto const& hashes = {n_hash, s_hash, w_hash, e_hash,
                nw_hash, ne_hash, sw_hash, se_hash};
            for (auto const& [i, step] : enumerate(translations)) {
                constexpr auto const& f = hashes[i];
                Bitboard tx = (i < 4) ? qr : qb;
                auto [checker, pinned] = CAP[(f(them)<<14)|(f(us)<<7)|f(tx)];
                if (checker != 0 && pinned == 0) return false;
            }
            // knight threats
            Bitboard nt = knightthreats(oki) & knight & them;
            if (nt) return false;

            // pawn threats
            Bitboard pt = pawnthreats(ok, c()) & pawn & them;
            if (pt) return false;

            // safe!
            return true;
        }
        // Step 1. Which player is active? (i.e. Whose turn?)
        bool whitemove = !c();
        // Take the perspective of the moving player, so
        // it becomes 'us' vs 'them'.
        Bitboard & us = whitemove ? white : black;
        Bitboard & them = whitemove ? black : white;
        Bitboard empty = ~(us | them);

        // Let's do king moves first.
        Bitboard ok = us & king;
        uint8_t oki = ntz(ok);
        uint8_t row = oki >> 3;
        uint8_t col = oki & 0x07;
        auto S = kingthreats(oki) & ~us;
        // take our king off the board
        us ^= ok;
        king ^= ok;
        // loop through the possibilities
        while (S) {
          uint8_t si = ntz(S);
          if (!check(ni)) add_move(c(), false, KING, oki, ni);
          S &= (S-1);
        }
        // put the king back on the board
        us ^= ok;
        king ^= ok;

        // Step 2. Are we in check?
        Bitboard targets = ~us;
        Bitboard pinned = uint64_t(0); // idea: init pinned as 'them'
        uint8_t num_checks = 0;

        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;
        auto translations = {-8, 8, -1, 1, -9, -7, 7, 9};
        auto const& hashes = {n_hash, s_hash, w_hash, e_hash,
            nw_hash, ne_hash, sw_hash, se_hash};
        for (auto const& [i, step] : enumerate(translations)) {
            auto const& f = hashes[i];
            Bitboard const& tx = (i < 4) ? qr : qb;
            auto [checker, pinned] = CAP[(f(them)<<14)|(f(us)<<7)|f(tx)];
            if (checker != 0) {
               if(pinned == 0) {
                 num_checks += 1;
                 uint8_t ci = oki + step * (checker + 1);
                 targets &= INTERPOSITIONS[oki | (ci << 6)];  // { 3 }
               } else {
                 pinned |= 1UL << (oki + step * (pinned + 1));
               }
               // maybe +1's can be removed along with other shifts elsewhere
            }
        }
        // knight checks
        S = knightthreats(oki) & knight & them;
        num_checks += popcount(S);
        while (S) {
          targets &= (1UL << ntz(S));
          S &= S - 1;
        }

        // pawn checks
        S = pawnthreats(ok, c()) & pawn & them;
        num_checks += popcount(pt);
        while (S) {
          targets &= (1UL << ntz(S));
          S &= S - 1;
        }

        if (targets == 0) return moves;

        // Move generation

        // Queen Moves
        S = queen & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                // TODO: deal with pinned queen case { 4 }
            } else {
              add_s_T(c(), false, QUEEN, si, rookthreats(si, empty) & not_us);
              add_s_T(c(), false, QUEEN, si, bishopthreats(si, empty) & not_us);
            }
        }

        // Rook moves
        S = rook & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                // TODO: deal with pinned rook case { 5 }
            } else {
                add(si, false, rookthreats(si, empty) & not_us, MOVE_R);
            }
        }

        // Bishop moves
        S = bishop & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                // TODO: deal with pinned bishop case { 6 }
            } else {
                add(si, false, bishopthreats(si, empty) & not_us, MOVE_B);
            }
        }

        // Find Knight moves
        S = knight & us & ~pinned;
        while (S) {
          uint8_t si = ntz(S);
          S &= S-1;
          add_s_T(c(), false, KNIGHT, si, knightthreats(si) & targets);
        }

        // Pawn pushes
        Bitboard our_pawns = pawn & us;
        Bitboard T = empty & (color ? (our_pawns << 8) : (our_pawns >> 8));
        while (T) {
          // TODO: deal with pinned pawn for this case { 7 }
          auto ti = ntz(T);
          T &= T-1;
          Square si = ti - (color ? 8 : -8);
          if ((1UL << ti) & endrank) {
            add_s_t(c(), true, QUEEN, si, ti);
            add_s_t(c(), true, ROOK, si, ti);
            add_s_t(c(), true, BISHOP, si, ti);
            add_s_t(c(), true, KNIGHT, si, ti);
          } else {
            add_s_t(c(), false, PAWN, si, ti);
          }
        }

        // Pawn captures (except en passant)
        T = pawnthreats(our_pawns, color) & them;
        while (T) {
          // TODO: deal with pinned pawn for this case { 8 }
          auto ti = ntz(T);
          T &= T-1;
          Bitboard t = 1UL << ti;
          S = pawnthreats(t, !color) & our_pawns;
          while (S) {
            auto si = ntz(S);
            S &= S-1;
            if (t & endrank) {
              add_s_t(c(), true, QUEEN, si, ti);
              add_s_t(c(), true, ROOK, si, ti);
              add_s_t(c(), true, BISHOP, si, ti);
              add_s_t(c(), true, KNIGHT, si, ti);
            } else {
              add_s_t(c(), false, PAWN, si, ti);
            }
          }
        }

        // Double Pawn pushes
        S = our_pawns & (color ? 0x000000000000FF00UL :
                                 0x00FF000000000000UL);
        T = empty & (color ? ((S << 16) & (empty << 8))
                           : ((S >> 16) & (empty >> 8)));
        while (T) {
          // TODO: deal with pinned pawn for this case { 9 }
          auto ti = ntz(T);
          T &= T-1;
          Square si = ti - (color ? 16 : -16);
          add_s_t(c(), false, PAWN, si, ti, si & 0x07);
        }

        // En Passant
        if (epi() < 8) {
            Bitboard ep = 1UL << epi();
            S = pawnthreats(ep, !color) & our_pawns;
            while (S) {
                auto si = ntz(S);
                S &= S-1;

                // !! ATTENTION !! { 10 }

                // there is a crazy corner case not dealt with here
                //   pp.p
                //   ....
                //   rPpK

                add_s_t(c(), false, PAWN, si, epi(), 8);
            }
        }


        // Kingside Castle
        if (color ? bkcr() : wkcr()) {
          Bitboard conf = (color ? 240UL : (240UL << 56));
          if ((us & conf) == (color ? 144UL : (144UL << 56))) {
            if ((empty & conf) == (color ? 96UL : (96UL << 56))) {
              if (!check(oki) && !check(oki+1) && !check(oki+2)) {
                add_s_t(c(), false, KING, oki, oki + 2, 9);
              }
            }
          }
        }

        // Queenside Castle
        if (color ? bqcr() : wqcr()) {
          auto conf = (color ? 31UL : (31UL << 56));
          if ((us & conf) == (color ? 17UL : (17UL << 56))) {
            if ((empty & conf) == (color ? 14UL : (14UL << 56))) {
              if (!check(oki) && !check(oki-1) && !check(oki-2)) {
                add_s_t(c(), false, KING, oki, oki + 2, 10);
              }
            }
          }
        }
        return moves;
    }
};

std::array<ThreeByteMove,44304> compute_move_table() {
    std::array<ThreeByteMove,44304> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto tbm = ThreeByteMove(i);
        if (tbm.feasible()) result[j++] = tbm;
    }
    return result;
}
std::array<Position,44304> compute_posmove_table() {
    std::array<Position,44304> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto tbm = ThreeByteMove(i);
        if (tbm.feasible()){
          Position p;
          p.play(p);
          p.play(tbm);
          result[j++] = p;
        }
    }
    return result;
}
std::array<uint16_t,16777216> compute_lookup_table() {
    // to make this independent from compute_move_table()
    // we simply recompute it here.
    std::array<uint32_t,44304> movetable {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto tbm = ThreeByteMove(i);
        if (tbm.feasible()) movetable[j++] = i;
    }
    std::array<uint16_t,16777216> result {};
    j = 0;
    for (uint32_t i : movetable) {
      if (i < 16777216) {
        result[i] = j++;
      }
    }
    return result;
}
std::array<ThreeByteMove,44304> MOVETABLE = compute_move_table();
std::array<Position,44304> POSMOVETABLE = compute_posmove_table();
std::array<uint16_t,16777216> LOOKUP = compute_lookup_table();
void moves_csv_to_stdout() {
    uint32_t cnt = 0;
    uint32_t pcnt = 0;
    uint32_t ncnt = 0;
    uint32_t rcnt = 0;
    uint32_t bcnt = 0;
    uint32_t qcnt = 0;
    uint32_t kcnt = 0;
    std::cout << "turn, sp, sc, sr, tp, tc, tr, cp, wkcr, wqcr, bkcr, bqcr\n";
    for ( uint32_t i = 0; i < 256*256*256; ++i) {
        ThreeByteMove tbm(i);
        if (tbm.feasible()) {
            std::cout <<
                (tbm.c() ? "b" : "w") <<
                GLYPHS[tbm.sp()] <<
                char('a'+tbm.sc()) <<
                (8-tbm.sr()) <<
                GLYPHS[tbm.tp()] <<
                char('a'+tbm.tc()) <<
                (8-tbm.tr()) <<
                GLYPHS[tbm.cp()] <<
                (tbm.wkcr() ? "K" : "-") <<
                (tbm.wqcr() ? "Q" : "-") <<
                (tbm.bkcr() ? "k" : "-") <<
                (tbm.bqcr() ? "q" : "-") <<
                 "\n";
            ++ cnt;
            switch(tbm.sp()){
                case PAWN: ++pcnt; break;
                case KNIGHT: ++ncnt; break;
                case ROOK: ++rcnt; break;
                case BISHOP: ++bcnt; break;
                case QUEEN: ++qcnt; break;
                case KING: ++kcnt; break;
            }
        }
    }
    std::cerr << cnt << "\n";
    std::cerr << "P " << pcnt << "\n";
    std::cerr << "N " << ncnt << "\n";
    std::cerr << "R " << rcnt << "\n";
    std::cerr << "B " << bcnt << "\n";
    std::cerr << "Q " << qcnt << "\n";
    std::cerr << "K " << kcnt << "\n";
    // 44304 total
    // P 1352
    // N 3934
    // R 10556
    // B 6524
    // Q 16862
    // K 5076

    // here's the breakdown (hand check):
    // P: (8+(8+14*5)*5+(8*4+14*4*4))*2+28+16
    //    == 1352
    // N: ((2*4+3*8+4*16)+(4*4+6*16)+8*16)*2*6-2*4*2-3*4*2
    //    -4*8*2+8-2
    //    == 3934
    // R: (8*5+6*6)*32+48*(2*5+12*6)*2+(3*5+6*6)*2+(4*5+6*6)*2
    //    +(7+8)*2+8==10548+8 (8 extras due to R-R cr)
    //    == 10556
    // B: (7*28+9*20+11*12+13*4)*2*6-7*32+28 == 6524
    // Q: (21*28+23*20+25*12+27*4)*6*2-21*16*2+21*4-11*2
    //    == 16862
    // K: (3*4+5*24+8*36)*6*2-(3*4+5*12)*2+8+4+(3*6+2*5)*2*3
    //    == 5076
}
int main(int argc, char * argv []) {
    // ... test goes here ...
    std::cout << "MOVETABLE = [";
    for (uint16_t code = 0; code < 44304; ++ code) {
        if (code > 0) std::cout << ", ";
        std::cout << MOVETABLE[code].X;
    }
    std::cout << "]\n";

    // Position P;
    // for (int x = 0; x < 100000; ++ x) {
    //   for (uint16_t code = 0; code < 44304; ++ code) {
    //       P.play2(POSMOVETABLE[code]);
    //   }
    // }
    return 0;
}
