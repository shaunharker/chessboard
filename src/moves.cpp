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

// Chessboard squares are represented as integers in range(64)
// as follows: 0 ~ a8, 1 ~ b8, ..., 62 ~ g1, 63 ~ h1
constexpr auto squares = range<64>();

// A Bitboard represents a subset of chessboard squares
// using the bits of a 64 bit unsigned integer such that
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

Bitboard rookcollisionfreehash(Square i, Bitboard const& E) {
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

Bitboard bishopcollisionfreehash(Square i, Bitboard const& E) {
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

uint8_t bitreverse8(uint8_t x) {
  return __builtin_bitreverse8(x);
}

uint8_t e_hash (Bitboard x, uint8_t row, uint8_t col) {
    return (x >> (8*row+col+1)) & rank_8;
}

uint8_t n_hash (Bitboard x, uint8_t row, uint8_t col) {
    return (antidiagonal * (file_a & (x >> (8*row+col+8)))) >> 56;
}

uint8_t w_hash (Bitboard x, uint8_t row, uint8_t col) {
    return bitreverse8(((x >> (8*row)) << (8-col)) & rank_8);
}

uint8_t s_hash (Bitboard x, uint8_t row, uint8_t col) {
    // there is probably slightly faster magic
    return bitreverse8((antidiagonal * (file_a & (x << (8*(8-row)-col)))) >> 56);
}

Bitboard nwse_diagonal(uint8_t row, uint8_t col) {
  if (row > col) {
    return diagonal >> (8*(row-col));
  } else {
    return diagonal << (8*(col-row));
  }
}

Bitboard swne_diagonal(uint8_t row, uint8_t col) {
  if (row + col < 7) {
    return antidiagonal << (8*(7 - row+col));
  } else {
    return antidiagonal >> (8*(row+col - 7));
  }
}

uint8_t nw_hash (Bitboard x, uint8_t row, uint8_t col) {
    Bitboard this_diagonal = nwse_diagonal(row, col);
    return bitreverse8((((file_a * (x & this_diagonal)) >> 56) << (8-col)) & rank_8);
};

uint8_t ne_hash (Bitboard x, uint8_t row, uint8_t col) {
    Bitboard this_diagonal = swne_diagonal(row, col);
    return (((file_a * (x & this_diagonal)) >> 56) >> (col+1)) & rank_8;
};

uint8_t sw_hash (Bitboard x, uint8_t row, uint8_t col) {
    Bitboard this_diagonal = swne_diagonal(row, col);
    return bitreverse8((((file_a * (x & this_diagonal)) >> 56) << (8-col)) & rank_8);
};

uint8_t se_hash (Bitboard x, uint8_t row, uint8_t col) {
    Bitboard this_diagonal = nwse_diagonal(row, col);
    return (((file_a * (x & this_diagonal)) >> 56) << (col+1)) & rank_8;
};

std::array<std::pair<uint8_t,uint8_t>, (1 << 24)> compute_cap() {
    // compute checks and pins on every possible ray
    //
    // table indexing scheme:
    // +------------------------------------+
    // | cap address bits                   |
    // +-----------+-----------+------------+
    // | 0x00-0x07 | 0x08-0x0E | 0x0F-0x014 |
    // | slider    | us        | them       |
    // +-----------+-----------+------------+
    //
    // `slider` gives the bits where a ray-sliding enemy attackers are
    //
    // `them` gives the bits where enemy pieces are.
    //   notice we have `(them & slider) == slider`
    //
    // `us` gives the bits where our pieces are.
    //
    // each byte orders bits so >> is kingward.
    // the king position itself can be thought of as -1, rolled
    //   one off the edge.
    //
    // Output:
    //
    // auto const& [checker, pin] = cap[slider | (us << 8) | (them << 16)];

    std::array<std::pair<uint8_t,uint8_t>, (1 << 24)> result {};
    for (uint32_t x = 0; x < (1 << 21); ++ x) {
        uint8_t slider = x & 0x7F;
        uint8_t us = (x >> 8) & 0x7F;
        uint8_t them = (x >> 16) & 0x7F;
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
                result[x] = {checker + 1, 0};
                break;
            case 1:
                result[x] = {checker + 1, ntz(us) + 1};
                break;
            default:
                result[x] = {0,0};
                break;
        }
    }
    return result;
}

std::array<std::pair<uint8_t,uint8_t>, (1 << 24)> CAP = compute_cap();

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

std::vector<Bitboard> computebishopthreats() {
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

std::vector<uint64_t> computeinterpositions() {
    // 64x64 table of bitboards expressing next allowed targets
    // if the goal is to interpose an attack from s to t.
    // (note: capture of s is allowed, capture of t is not.)
    std::vector<uint64_t> result;
    for (uint8_t ti = 0; ti < 64; ++ ti) {
        uint8_t tc = ti & 7; uint8_t tr = ti >> 3;
        for (uint8_t si = 0; si < 64; ++ si) {
            uint8_t sc = si & 7; uint8_t sr = si >> 3;
            if (sc == tc) {
                uint64_t F = 0x0101010101010101UL << sc;
                if (sr < tr) {
                    result.push_back((F >> (8*(8-tr))) & (F << (8*(sr))));
                } else { // sr >= tr
                    result.push_back((F >> (8*(7-sr))) & (F << (8*(tr+1))));
                }
            } else if (sr == tr) {
                uint64_t R = 0x00000000000000FFUL << (8*sr);
                if (sc < tc) {
                    result.push_back(R & (R >> (8-tc)) & (R << (sc)));
                } else { // sr >= tr
                    result.push_back(R & (R >> (7-sc)) & (R << (tc+1)));
                }
            } else if (sr + sc == tr + tc) {
                uint64_t A = (sr + sc < 7) ? (0x0102040810204080UL >> (8*(7-sr-sc))) : (0x0102040810204080UL << (8*(sr+sc-7)));
                if (sr < tr) {
                    result.push_back(A & (A << (7*(sr))) & (A >> (7*(8-tr))));
                } else { // sr >= tr
                    result.push_back(A & (A << (7*(tr+1))) & (A >> (7*(7-sr))));
                }
            } else if (sr + tc == tr + sc) {
                uint64_t D = (sr < sc) ? (0x8040201008040201UL >> (8*(sc-sr))) : (0x8040201008040201UL << (8*(sr-sc)));
                if (sr < tr) {
                    result.push_back(D & (D << (9*(sr))) & (D >> (9*(8-tr))));
                } else {
                    result.push_back(D & (D << (9*(tr+1))) & (D >> (9*(7-sr))));
                }
            } else {
                result.push_back(-1);
            }
        }
    }
    return result;
}

std::vector<Bitboard> INTERPOSITIONS = computeinterpositions();

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

struct Move {
    uint8_t tc_tr_bqcr_bkcr;
    uint8_t sc_sr_wqcr_wkcr;
    uint8_t cp_sp_pr_c;
    uint8_t epc0_ep0_epc1_ep1;

    // 0x00 - 0x02 | tc     | target col idx into abcdefgh
    // 0x03 - 0x05 | tr     | target row idx into abcdefgh
    // 0x06        | bqcr   | change black queen castling rights
    // 0x07        | bkcr   | change black king castling rights
    // 0x08 - 0x0A | sc     | source col idx into abcdefgh
    // 0x0B - 0x0D | sr     | source row idx into 87654321
    // 0x0E        | wqcr   | change white queen castling rights
    // 0x0F        | wkcr   | change white king castling rights
    // 0x10 - 0x12 | cp     | capture piece idx into .PNBRQK
    // 0x13 - 0x15 | sp     | source piece idx into .PNBRQK
    // 0x16        | pr     | promotion bit
    // 0x17        | c      | 0 if white moving, else 1
    // 0x18 - 0x1A | epc0   | en passant square col idx into abcdefgh
    // 0x1B        | ep0    | 1 if last move was double push
    // 0x1C - 0x1E | epc1   | en passant square col idx into abcdefgh
    // 0x1F        | ep1    | 1 of this move is a double push

    constexpr Move () :
        tc_tr_bqcr_bkcr(0),
        sc_sr_wqcr_wkcr(0),
        cp_sp_pr_c(0),
        epc0_ep0_epc1_ep1(0) {}

    constexpr Move (uint32_t X) :
        tc_tr_bqcr_bkcr(X & 0xFF),
        sc_sr_wqcr_wkcr((X >> 0x08) & 0xFF),
        cp_sp_pr_c((X >> 0x10) & 0xFF),
        epc0_ep0_epc1_ep1((X >> 0x18) & 0xFF) {}

    constexpr Move (uint8_t tc, uint8_t tr, bool bqcr, bool bkcr, uint8_t sc, uint8_t sr, bool wqcr, bool wkcr, uint8_t cp, uint8_t sp, bool pr, bool c, uint8_t epc0, bool ep0, uint8_t epc1, bool ep1) :
        tc_tr_bqcr_bkcr((tc & 7) | ((tr & 7) << 3) | (bqcr << 6) | (bkcr << 7)),
        sc_sr_wqcr_wkcr((sc & 7) | ((sr & 7) << 3) | (wqcr << 6) | (wkcr << 7)),
        cp_sp_pr_c((cp & 7) | ((sp & 7) << 3) | (pr << 6) | (c << 7)),
        epc0_ep0_epc1_ep1((epc0 & 7) | (ep0 << 3) | ((epc1 & 7) << 4) | (ep1 << 7)) {}

    constexpr uint8_t tc() const {return tc_tr_bqcr_bkcr & 0x07;}
    constexpr uint8_t tr() const {return (tc_tr_bqcr_bkcr >> 3) & 0x07;}
    constexpr uint8_t ti() const {return tc_tr_bqcr_bkcr & 0x3F;}
    constexpr bool bqcr() const {return (tc_tr_bqcr_bkcr & 0x40) != 0;}
    constexpr bool bkcr() const {return (tc_tr_bqcr_bkcr & 0x80) != 0;}
    constexpr uint8_t sc() const {return sc_sr_wqcr_wkcr & 0x07;}
    constexpr uint8_t sr() const {return (sc_sr_wqcr_wkcr >> 3) & 0x07;}
    constexpr uint8_t si() const {return sc_sr_wqcr_wkcr & 0x3F;}
    constexpr bool wqcr() const {return (sc_sr_wqcr_wkcr & 0x40) != 0;}
    constexpr bool wkcr() const {return (sc_sr_wqcr_wkcr & 0x80) != 0;}
    constexpr uint8_t cp() const {return (cp_sp_pr_c) & 0x07;}
    constexpr uint8_t sp() const {return (cp_sp_pr_c >> 0x03) & 0x07;}
    constexpr bool pr() const {return cp_sp_pr_c & 0x40;}
    constexpr bool c() const {return cp_sp_pr_c & 0x80;}
    constexpr uint8_t epc0() const {return epc0_ep0_epc1_ep1 & 0x07;}
    constexpr bool ep0() const {return epc0_ep0_epc1_ep1 & 0x08;}
    constexpr uint8_t epc1() const {return (epc0_ep0_epc1_ep1 >> 0x04) & 0x07;}
    constexpr bool ep1() const {return (epc0_ep0_epc1_ep1 >> 0x04) & 0x08;}

    // queries
    constexpr uint64_t s() const {return 1UL << si();}
    constexpr uint64_t t() const {return 1UL << ti();}
    constexpr uint64_t st() const {return s() | t();}
    constexpr uint64_t ui() const {return (tc() << 3) | sr();}
    constexpr uint64_t u() const {return 1UL << ui();}
    constexpr uint8_t cr() const {return (wkcr() ? 0x01 : 0x00) | (wqcr() ? 0x02 : 0x00) | (bkcr() ? 0x04 : 0x00) | (bqcr() ? 0x08 : 0x00);}

    // feasibility (optional? might need it for future tables)
    constexpr bool kcr() const {return wkcr() && bkcr();}
    constexpr bool qcr() const {return wkcr() && bkcr();}
    constexpr bool wcr() const {return wkcr() && wqcr();}
    constexpr bool bcr() const {return bkcr() && bqcr();}

    // TODO: repair this
    constexpr bool feasible() const {

        // sp must be a Piece enum idx
        if (sp() > 6) return false;

        // can't move from an empty square
        if (sp() == SPACE) return false;

        // cp must be a Piece enum idx
        if (cp() > 6) return false;

        // cp may not name KING
        if (cp() == KING) return false;

        // source != target
        if ((sc() == tc()) && (sr() == tr())) return false;

        // only pawns promote, and it must be properly positioned
        // and cannot promote to pawn or king
        if (pr() && ((sr() != (c() ? 6 : 1)) || (tr() != (c() ? 7 : 0)) || (sp() == PAWN) || (sp() == KING))) return false;

        // pawns are never on rank 8 or rank 1 (row 0 or row 7)
        if ((sp() == PAWN) && ((sr() == 0) ||
            (sr() == 7))) return false;
        if ((cp() == PAWN) && ((tr() == 0) ||
            (tr() == 7))) return false;

        if ((sp() == PAWN) || pr()) {
            // pawns can only move forward one rank at a time,
            // except for their first move
            if (sr() != tr() + (c() ? -1 : 1)) {
                if ((sr() != (c() ? 1 : 6)) ||
                    (tr() != (c() ? 3 : 4))) return false;
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
                if (tr() != (c() ? 5 : 2)) return false;
            }
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
            if (kcr() && !((sc() == 7) && (sr() == (c() ? 0 : 7))) && !((tc() == 7) && (tr() == (c() ? 7 : 0)))) return false;
            // if losing kingside rights, cannot move to a rook to files a-e
            if (kcr() && (tc() < 5)) return false;
            // conditions where queenside castle right may change
            if (qcr() && !((sc() == 0) && (sr() == (c() ? 0 : 7))) && !((tc() == 0) && (tr() == (c() ? 7 : 0)))) return false;
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
            if ((sc() == 4) && (sr() == (c() ? 0 : 7)) && (tc() == 6) && (tr() == (c() ? 0 : 7)) && !kcr()) return false;
            // if queenside castle, must be losing queenside rights
            if ((sc() == 4) && (sr() == (c() ? 0 : 7)) && (tc() == 2) && (tr() == (c() ? 0 : 7)) && !qcr()) return false;
            // king takes rook losing castling rights:
            //   only diagonal/antidiagonal captures could
            //   possibly occur during play:
            if ((cp() == ROOK) && kcr() && (tr() == (c() ? 7 : 0)) && (tc() == 7) && !((sr() == (c() ? 6 : 1)) && (sc() == 6))) return false;
            if ((cp() == ROOK) && qcr() && (tr() == (c() ? 7 : 0)) && (tc() == 0) && !((sr() == (c() ? 6 : 1)) && (sc() == 1))) return false;
            // castling cannot capture, must be properly positioned
            if (sc()*sc() + tc()*tc() > 1 + 2*sc()*tc()) {
                if (!((tc() == 6) && kcr()) && !((tc() == 2) && qcr())) return false;
                if (cp() != SPACE) return false;
                if (sc() != 4) return false;
                if (sr() != (c() ? 0 : 7)) return false;
                if (tr() != (c() ? 0 : 7)) return false;
            }
            // kings move to neighboring squares
            if (((sc()*sc() + tc()*tc() + sr()*sr()) + tr()*tr() >
                2*(1 + sc()*tc() + sr()*tr())) && !((sc() == 4) &&
                (sr() == (c() ? 0 : 7)) && (tr()==sr()) &&
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
        bool kingmove = (sp() == KING) && (sr() == (c() ? 0 : 7)) && (sc() == 4);
        bool a1rookcapture = (cp() == ROOK) && (ti() == 56) && c();
        bool a8rookcapture = (cp() == ROOK) && (ti() == 0) && !c();
        bool h1rookcapture = (cp() == ROOK) && (ti() == 63) && c();
        bool h8rookcapture = (cp() == ROOK) && (ti() == 7) && !c();

        bool a1rookmove = (sp() == ROOK) && (si() == 56) && !c() && (tc() < 4);
        bool a8rookmove = (sp() == ROOK) && (si() == 0) && c() && (tc() < 4);
        bool h1rookmove = (sp() == ROOK) && (si() == 63) && !c() && (tc() > 4);
        bool h8rookmove = (sp() == ROOK) && (si() == 7) && c() && (tc() > 4);
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
                if ((sp() == KNIGHT) && (sr() == (c() ? 6 : 1)) && (sc() == 2)) return false;
                if ((sp() == KING) && ((sr() == tr()) || (sc() == tc()))) return false;
            } else {
                    return false;
            }
        }
        return true;
    }
};

struct Position {
    // We store a chess position with 8 bitboards, two for
    // colors and six for pieces. We keep castling rights
    // bitwise in uint8_t cr, the en passant column in
    // epc_, whether a double push occurred last move in
    // ep_, and the active color in c_:

    Bitboard pawn; // 0 for empty, 1 for pawns
    Bitboard knight; // 0 for empty, 1 for knights
    Bitboard bishop; // 0 for empty, 1 for bishops
    Bitboard rook; // 0 for empty, 1 for rooks
    Bitboard queen; // 0 for empty, 1 for queens
    Bitboard king;  // 0 for empty, 1 for kings
    Bitboard white; // 0 for empty, 1 for white pieces
    Bitboard black; // 0 for empty, 1 for black pieces
    uint8_t cr; // castling rights. bit 0 1 2 3 ~ wk wq bk bq
    uint8_t epc_; // if ep_, col of last double push; else 0
    bool ep_; // true if last move was double push
    bool c_; // false when white to move, true when black to move

    Position() {
        white = 0xFFFF000000000000; // rank_1 | rank_2;
        black = 0x000000000000FFFF; // rank_7 | rank_8;
        king = 0x1000000000000010; // e1 | e8;
        pawn = 0x00FF00000000FF00; // rank_2 | rank_7
        queen = 0x0800000000000008; // d1 | d8
        rook = 0x8100000000000081; // a1 | a8 | h1 | h8;
        bishop = 0x2400000000000024; // c1 | c8 | f1 | f8;
        knight = 0x4200000000000042; // b1 | b8 | g1 | g8;
        cr = 0x00; // castling rights
        epc_ = 0x00; // en passant column (meaningless when ep_ == false
        ep_ = false; // true if previous move was double push
        c_ = false; // true if black to move
    }

    constexpr bool wkcr() const { return cr & 1; }
    constexpr bool wqcr() const { return cr & 2; }
    constexpr bool bkcr() const { return cr & 4; }
    constexpr bool bqcr() const { return cr & 8; }
    constexpr uint8_t epc() const { return epc_; }
    constexpr bool ep() const { return ep_; }
    constexpr uint8_t epi() const { return epc_ | (c() ? 40 : 16); }
    constexpr bool c() const { return c_; }

    void play(Position rhs) {
        pawn ^= rhs.pawn;
        knight ^= rhs.knight;
        bishop ^= rhs.bishop;
        rook ^= rhs.rook;
        queen ^= rhs.queen;
        king ^= rhs.king;
        white ^= rhs.white;
        black ^= rhs.black;
        cr ^= rhs.cr;
        epc_ ^= rhs.epc_;
        ep_ ^= rhs.ep_;
        c_ ^= rhs.c_;
    }

    void play(Move const& move) {
        auto pr = move.pr();
        auto sc = move.sc();
        auto tc = move.tc();
        auto sp = move.sp();
        auto cp = move.cp();
        auto color = move.c();

        uint64_t s = move.s();
        uint64_t t = move.t();
        uint64_t st = move.st();

        uint64_t & us = color ? black : white;
        uint64_t & them = color ? white : black;

        c_ = !c_;

        ep_ ^= move.ep0();
        ep_ ^= move.ep1();

        epc_ ^= move.epc0();
        epc_ ^= move.epc1();

        cr ^= move.cr();

        us ^= st;

        if (pr) {
            pawn ^= s;
            switch (sp) {
                case KNIGHT: knight ^= t; break;
                case BISHOP: bishop ^= t; break;
                case ROOK: rook ^= t; break;
                case QUEEN: queen ^= t; break;
                default: break; // actually, HCF
            }
        } else {
            switch (sp) {
                case PAWN: pawn ^= st; break;
                case KNIGHT: knight ^= st; break;
                case BISHOP: bishop ^= st; break;
                case ROOK: rook ^= st; break;
                case QUEEN: queen ^= st; break;
                case KING: king ^= st; break;
            }
        }

        switch (cp) {
            case PAWN: pawn ^= t; them ^= t; break;
            case KNIGHT: knight ^= t; them ^= t; break;
            case BISHOP: bishop ^= t; them ^= t; break;
            case ROOK: rook ^= t; them ^= t; break;
            case QUEEN: queen ^= t; them ^= t; break;
            default: break;
        }

        if ((sp == PAWN) && (cp == SPACE) && (sc != tc)) {
            // en passant capture
            Bitboard u = move.u();
            pawn ^= u;
            them ^= u;
        }

        if ((sp == KING) && (tc == sc + 2)) {
            rook ^= color ? 0x00000000000000A0 : 0xA000000000000000;
            us ^= color ? 0x00000000000000A0 : 0xA000000000000000;
        }

        if ((sp == KING) && (tc + 2 == sc)) {
            rook ^= color ? 0x0000000000000009 : 0x0900000000000000;
            us ^= color ? 0x0000000000000009 : 0x0900000000000000;
        }
    }

    void undo(Move const& move) {
        play(move); // undoes itself
    }

    void add_move_s_t(std::vector<Move> & moves, bool pr, Piece sp, uint8_t si, uint8_t ti) {
        uint64_t t = 1UL << ti;
        Piece cp;
        if (pawn & t) {
            cp = PAWN;
        } else if (knight & t) {
            cp = KNIGHT;
        } else if (bishop & t) {
            cp = BISHOP;
        } else if (rook & t) {
            cp = ROOK;
        } else if (queen & t) {
            cp = QUEEN;
        } else {
            cp = SPACE;
        }
        uint8_t sc = si & 7;
        uint8_t sr = si >> 3;
        uint8_t tc = ti & 7;
        uint8_t tr = ti >> 3;
        bool bqcr1 = bqcr() && ((si == 4) || (ti == 2) || (si == 0));
        bool bkcr1 = bkcr() && ((si == 4) || (ti == 7) || (si == 7));
        bool wqcr1 = wqcr() && ((si == 60) || (ti == 58) || (si == 7));
        bool wkcr1 = wkcr() && ((si == 60) || (ti == 62) || (si == 63));
        bool ep1 = (sp == PAWN) && ((si == ti + 16) || (ti == si + 16));
        uint8_t epc1 = ep1 ? tc : 0;

        moves.push_back(Move(tc, tr, bqcr1, bkcr1, sc, sr, wqcr1, wkcr1, cp, sp, pr, c(), epc(), ep(), epc1, ep1));
    }

    void add_move_s_T(std::vector<Move> & moves, bool pr, Piece sp, uint8_t si, Bitboard T) {
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            add_move_s_t(moves, pr, sp, si, ti);
        }
    }

    bool check(uint8_t si) {
        // determine if si is attacked by "them" as the board stands
        Bitboard const& us = c() ? black : white;
        Bitboard const& them = c() ? white : black;

        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;

        uint8_t sc = si & 7;
        uint8_t sr = si >> 3;

        for (auto const& f : {n_hash, s_hash, w_hash, e_hash}) {
            uint32_t address = (f(them, sc, sr) << 16) | (f(us, sc, sr) << 8) | f(qr, sc, sr);
            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                std::cout << "Check Type 1\n";
                return true;
            }
        }

        for (auto const& f : {nw_hash, ne_hash, sw_hash, se_hash}) {
            uint32_t address = (f(them, sc, sr) << 16) | (f(us, sc, sr) << 8) | f(qb, sc, sr);
            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                std::cout << "Check Type 2\n";
                std::cout << address << "\n";
                std::cout << us << "\n";
                std::cout << them << "\n";
                std::cout << int(si) << "\n";
                std::cout << "sliders " << f(them, sc, sr) << "\n";
                std::cout << "
                return true;
            }
        }

        // knight threats
        if (knightthreats(si) & knight & them) {
            std::cout << "Check Type 3\n";
             return true;
        }
        // pawn threats
        if (pawnthreats(1UL << si, c()) & pawn & them) {
            std::cout << "Check Type 4\n";
            return true;
        }

        // safe!
        return false;
    }

    bool in_check() {
        return check(ntz(king & (c() ? black : white)));
    }

    std::vector<Move> legal_moves() { //uint16_t *out, uint8_t *moves_written) {
        // Step 1. Which player is active? (i.e. Whose turn?)
        // Take the perspective of the moving player, so
        // it becomes 'us' vs 'them'.
        Bitboard & us = c() ? black : white;
        Bitboard & them = c() ? white : black;
        Bitboard empty = ~(us | them);
        std::vector<Move> moves {};

        // Let's do king moves first.
        Bitboard ok = us & king;
        uint8_t oki = ntz(ok);
        uint8_t okr = oki >> 3;
        uint8_t okc = oki & 0x07;

        // take our king off the board
        us ^= ok;
        king ^= ok;

        // loop through the possibilities
        uint64_t S, T;

        T = kingthreats(oki) & ~us;
        while (T) {
            uint8_t ti = ntz(T);
            T &= (T-1);
            if (!check(ti)) {
                // std::cout << "king move\n";
                add_move_s_t(moves, false, KING, oki, ti);
            }
        }

        // put the king back on the board
        us ^= ok;
        king ^= ok;

        // Are we in check?

        // We compute a bitboard we name "targets" which in the case of a
        // check gives us the bitboard of acceptable target squares for
        // non-king pieces. (i.e. they have to interpose or capture)
        Bitboard targets = ~us;

        // We also compute a bitboard we name "pinned" which gives the
        // bitboard marking squares holding pieces pinned to the active
        // king by a would-be checker.
        Bitboard pinned = uint64_t(0);

        // Our algorithm proceeds by searching in each of the 8 rays
        // from the king's position. Each ray is processed with a hash
        // table CAP and we receive information (checker, pin) which we
        // use to key into the INTERPOSITIONS table, which gives a
        // bitboard telling us the allowed squares a non-king piece
        // must target in order to deal with the check (i.e. either
        // interpose or capture the attacker giving check).

        auto check_and_pin_search = [&](auto&& f, uint8_t x, int8_t step) {
            auto const& [checker, pin] = CAP[(f(them, okc, okr) << 16) | (f(us, okc, okr) << 8) | f(x, okc, okr)];
            if (checker != 0) {
               if (pin == 0) {
                 uint8_t ci = oki + step * checker;
                 targets &= INTERPOSITIONS[(oki << 6) | ci];
               } else {
                 pinned |= 1UL << (oki + step * pin);
               }
            }
        };

        Bitboard qr = (queen | rook) & them;
        check_and_pin_search(n_hash, -8, qr);
        check_and_pin_search(s_hash,  8, qr);
        check_and_pin_search(w_hash, -1, qr);
        check_and_pin_search(e_hash,  1, qr);

        Bitboard qb = (queen | bishop) & them;
        check_and_pin_search(nw_hash, -9, qb);
        check_and_pin_search(ne_hash, -7, qb);
        check_and_pin_search(sw_hash,  7, qb);
        check_and_pin_search(se_hash,  9, qb);

        // knight checks
        S = knightthreats(oki) & knight & them;
        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          targets &= (1UL << si);
        }

        // pawn checks
        S = pawnthreats(ok, c()) & pawn & them;
        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          targets &= (1UL << si);
        }

        if (targets == 0) { // king must move
            return moves;
        }

        if (targets == -1) { // no checks
            // Kingside Castle
            if (c() ? bkcr() : wkcr()) {
                Bitboard conf = (c() ? 240UL : (240UL << 56));
                if (((us & conf) == (c() ? 144UL : (144UL << 56))) &&
                    ((empty & conf) == (c() ? 96UL : (96UL << 56))) &&
                    !check(oki+1) && !check(oki+2)) {
                    // std::cout << "kingside castle move\n";
                    add_move_s_t(moves, false, KING, oki, oki + 2);
                }
            }

            // Queenside Castle
            if (c() ? bqcr() : wqcr()) {
                Bitboard conf = (c() ? 31UL : (31UL << 56));
                if (((us & conf) == (c() ? 17UL : (17UL << 56))) &&
                    ((empty & conf) == (c() ? 14UL : (14UL << 56))) &&
                    !check(oki-1) && !check(oki-2)) {
                    // std::cout << "queenside castle move\n";
                    add_move_s_t(moves, false, KING, oki, oki + 2);
                }
            }
        }

        // Queen Moves
        S = queen & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) {
                    Bitboard F = file_a << sc;
                    uint64_t T = F & rookthreats(si, empty) & ~us;
                    // if (T) std::cout << "file-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if (sr == okr) {
                    Bitboard R = rank_8 << (8*sr);
                    uint64_t T = R & rookthreats(si, empty) & ~us;
                    // if (T) std::cout << "rank-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if ((sr + sc) == (okr + okc)) {
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & ~us;
                    // if (T) std::cout << "antidiagonally-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else { // sr - sc == okr - okc
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & ~us;
                    // if (T) std::cout << "diagonally-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                }
            } else {
                uint64_t TR = rookthreats(si, empty) & ~us;
                // if (TR) std::cout << "unpinned queen rook-like move\n";
                add_move_s_T(moves, false, QUEEN, si, TR);
                uint64_t TB = bishopthreats(si, empty) & ~us;
                // if (TB) std::cout << "unpinned queen bishop-like move\n";
                add_move_s_T(moves, false, QUEEN, si, TB);
            }
        }

        // Rook moves
        S = rook & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) {
                    Bitboard F = file_a << okc;
                    uint64_t T = F & rookthreats(si, empty) & ~us;
                    // if (T) std::cout << "file-pinned rook moves\n";
                    add_move_s_T(moves, false, ROOK, si, T);
                } else { // sr == okr
                    Bitboard R = rank_8 << (8*okr);
                    uint64_t T = R & rookthreats(si, empty) & ~us;
                    // if (T) std::cout << "rank-pinned rook moves\n";
                    add_move_s_T(moves, false, ROOK, si, T);
                }
            } else {
                uint64_t T = rookthreats(si, empty) & ~us;
                // if (T) std::cout << "unpinned rook moves\n";
                add_move_s_T(moves, false, ROOK, si, T);
            }
        }

        // Bishop moves
        S = bishop & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1UL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc + sr == okr + okc) {
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & ~us;
                    // if (T) std::cout << "antidiagonally pinned bishop moves\n";
                    add_move_s_T(moves, false, BISHOP, si, T);
                } else { // sr - sc == okr - okc
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & ~us;
                    // if (T) std::cout << "diagonally pinned bishop moves\n";
                    add_move_s_T(moves, false, BISHOP, si, T);
                }
            } else {
                uint64_t T = bishopthreats(si, empty) & ~us;
                // if (T) std::cout << "unpinned bishop moves\n";
                add_move_s_T(moves, false, BISHOP, si, T);
            }
        }

        // Knight moves
        S = knight & us & ~pinned;
        while (S) {
            uint8_t si = ntz(S);
            S &= S-1;
            uint64_t T = knightthreats(si) & targets;
            // if (T) std::cout << "knight moves\n";
            add_move_s_T(moves, false, KNIGHT, si, T);
        }

        // Pawn pushes
        Bitboard our_pawns = pawn & us;
        S = our_pawns & (c() ? (empty >> 8) : (empty << 8));
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1UL << si;
            uint8_t sc = si & 0x07;
            uint8_t ti = si + (c() ? 8 : -8);
            uint8_t tr = ti >> 3;
            if (((s & pinned) != 0) && (sc != okc)) continue;
            if (tr == 0 || tr == 7) {
                // std::cout << "pawn push promotion moves\n";
                add_move_s_t(moves, true, QUEEN, si, ti);
                add_move_s_t(moves, true, ROOK, si, ti);
                add_move_s_t(moves, true, BISHOP, si, ti);
                add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                // std::cout << "pawn push move\n";
                add_move_s_t(moves, false, PAWN, si, ti);
            }
        }

        // Pawn captures (except en passant)
        // loop over S first might be better
        T = pawnthreats(our_pawns, c()) & them;
        while (T) {
          auto ti = ntz(T);
          T &= T-1;
          Bitboard t = 1UL << ti;
          S = pawnthreats(t, !c()) & our_pawns;
          while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1UL << si;
            if (s & pinned) {
                uint8_t sc = si & 0x07;
                uint8_t sr = si >> 3;
                if ((sc == okc) || (sr == okr)) continue;
                uint8_t tc = ti & 0x07;
                uint8_t tr = ti >> 3;
                if ((sr + sc) == (okr + okc)) {
                    if ((sr + sc) != (tr + tc)) continue;
                } else { // sr - sc == okr - okc
                    if ((sr + sc) == (tr + tc)) continue;
                }
            }
            if ((1UL << ti) & (rank_1 | rank_8)) {
                // std::cout << "pawn capture promotion moves\n";
                add_move_s_t(moves, true, QUEEN, si, ti);
                add_move_s_t(moves, true, ROOK, si, ti);
                add_move_s_t(moves, true, BISHOP, si, ti);
                add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                // std::cout << "pawn capture move\n";
                add_move_s_t(moves, false, PAWN, si, ti);
            }
          }
        }

        // Double Pawn pushes
        S = our_pawns & (c() ? 0x000000000000FF00UL :
                                 0x00FF000000000000UL);
        T = empty & (c() ? ((S << 16) & (empty << 8))
                           : ((S >> 16) & (empty >> 8)));
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            uint8_t si = ti - (c() ? 16 : -16);
            Bitboard s = 1UL << si;
            if (((s & pinned) != 0) && ((si & 0x07) != okc)) continue;
            // std::cout << "double pawn push move " << int(si) << " " << int(ti) << " " << okc << " " << pinned << "\n";
            add_move_s_t(moves, false, PAWN, si, ti);
        }

        // A discovered check cannot be countered with
        // an en passant capture. ~The More You Know~

        // En Passant
        if (ep()) {
            S = pawnthreats(1UL << epi(), !c()) & our_pawns;
            while (S) {
                auto si = ntz(S);
                S &= S-1;
                //  Here we handle missed pawn pins of
                //  the following forms:
                //
                //   pp.p    White is prevented from
                //   ..v.    en passant capture.
                //   rPpK  <-- row 4 = rank 5
                //
                //
                //   RpPk  <-- row 5 = rank 4
                //   ..^.    Black is prevented from
                //   PP.P    en passant capture.
                //
                //  (the v and ^ indicate ep square)
                uint8_t row = c() ? 4 : 3;
                bool pin = false;
                if (okr == row) {
                    auto R = (rook & them & (rank_8 << (8*row)));
                    while (R) {
                        auto ri = ntz(R);
                        R &= R-1;
                        uint64_t r = 1UL << ri;
                        // Notice that
                        //   bool expr = ((a < b) && (b <= c)) ||
                        //               ((c < b) && (b <= a));
                        // is equivalent to
                        //   bool expr = (a < b) == (b <= c);
                        if ((ri < si) == (si < oki)) {
                            uint8_t cnt = 0;
                            if (ri < oki) {
                                for (uint64_t x = r<<1; x != ok; x <<= 1) {
                                    if ((x & empty) == 0) cnt += 1;
                                }
                            } else { // ri > oki
                                for (uint64_t x = ok<<1; x != r; x <<= 1) {
                                    if ((x & empty) == 0) cnt += 1;
                                }
                            }
                            if (cnt == 2) pin = true; // the prohibited case
                        }
                    }
                }
                if (!pin) {
                    // std::cout << "en passant move\n";
                    add_move_s_t(moves, false, PAWN, si, epi());
                }
            }
        }



        return moves;
    }
};

std::array<Move,44304> compute_move_table() {
    std::array<Move,44304> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto move = Move(i);
        if (move.feasible()) result[j++] = move;
    }
    return result;
}

std::array<Position,44304> compute_posmove_table() {
    std::array<Position,44304> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto move = Move(i);
        if (move.feasible()){
          Position p;
          p.play(p);
          p.play(move);
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
        auto move = Move(i);
        if (move.feasible()) movetable[j++] = i;
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

// std::array<Move,44304> MOVETABLE = compute_move_table();
//
// std::array<Position,44304> POSMOVETABLE = compute_posmove_table();
//
// std::array<uint16_t,16777216> LOOKUP = compute_lookup_table();

void moves_csv_to_stdout() {
    uint32_t cnt = 0;
    uint32_t pcnt = 0;
    uint32_t ncnt = 0;
    uint32_t rcnt = 0;
    uint32_t bcnt = 0;
    uint32_t qcnt = 0;
    uint32_t kcnt = 0;
    std::cout << "turn, pr, sp, sc, sr, tc, tr, cp, wkcr, wqcr, bkcr, bqcr\n";
    for ( uint32_t i = 0; i < 256*256*256; ++i) {
        Move move(i);
        if (move.feasible()) {
            std::cout <<
                (move.c() ? "b" : "w") <<
                (move.pr() ? "*" : "-") <<
                GLYPHS[move.sp()] <<
                char('a'+move.sc()) <<
                (8-move.sr()) <<
                char('a'+move.tc()) <<
                (8-move.tr()) <<
                GLYPHS[move.cp()] <<
                (move.wkcr() ? "K" : "-") <<
                (move.wqcr() ? "Q" : "-") <<
                (move.bkcr() ? "k" : "-") <<
                (move.bqcr() ? "q" : "-") <<
                 "\n";
            ++ cnt;
            switch(move.sp()){
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

uint64_t perft(Position & board, uint8_t depth) {
    uint64_t result = 0;
    if (depth == 0) return 1;
    auto legal = board.legal_moves();
    if (depth == 1) return legal.size();
    for (auto move : legal) {
        board.play(move);
        result += perft(board, depth-1);
        board.undo(move);
    }
    return result;
}



// TODO: implement Move::is_ep()

// uint64_t capturetest(Position & board, int depth) {
//   if (depth == 0) return 0;
//   auto moves = board.legal_moves();
//   uint64_t result = 0;
//   for (auto move : moves) {
//     board.play(move);
//     if ((depth == 1) && (move.cp() != SPACE || move.is_ep())) result += 1;
//     result += capturetest(board, depth-1);
//     board.undo(move);
//   }
//   return result;
// }

// uint64_t enpassanttest(Position & board, int depth) {
//   if (depth == 0) return 0;
//   auto moves = board.legal_moves();
//   uint64_t result = 0;
//   for (auto move : moves) {
//     board.play(move);
//     if ((depth == 1) && (move.is_ep())) result += 1;
//     result += enpassanttest(board, depth-1);
//     board.undo(move);
//   }
//   return result;
// }

uint64_t checktest(Position & board, int depth) {
  if (depth == 0) return board.in_check() ? 1 : 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    result += checktest(board, depth-1);
    board.undo(move);
  }
  return result;
}

// TODO: implement Position:doublecheck

// uint64_t doublechecktest(Position & board, int depth) {
//   if (depth == 0) return board.doublecheck() ? 1 : 0;
//   auto moves = board.legal_moves();
//   uint64_t result = 0;
//   for (auto move : moves) {
//     board.play(move);
//     result += doublechecktest(board, depth-1);
//     board.undo(move);
//   }
//   return result;
// }

// TODO: implement Position::mate

// uint64_t matetest(Position & board, int depth) {
//   if (depth == 0) return board.mate() ? 1 : 0;
//   auto moves = board.legal_moves();
//   uint64_t result = 0;
//   for (auto move : moves) {
//     board.play(move);
//     result += matetest(board, depth-1);
//     board.play(move);
//   }
//   return result;
// }

int main(int argc, char * argv []) {

    for (int d = 0; d < 2; ++ d) {
        auto P = Position(); // new chessboard
        std::cout << "\n----------\ndepth " << d << "\n";
        std::cout << "perft "; std::cout.flush();
        std::cout << perft(P, d) << "\n";
        std::cout << "checks "; std::cout.flush();
        std::cout << checktest(P, d) << "\n";
    }

    // test

    // auto P = Position();
    // auto legal = P.legal_moves();
    // for (auto move : legal) {
    //     if (!move.pr() && (move.sp() != PAWN)) {
    //         std::cout << GLYPHS[move.sp()];
    //     }
    //     std::cout << char('a' + move.sc()) << char('8' - move.sr()) << ((move.cp() == SPACE) ? "" : "x" ) << char('a' + move.tc()) << char('8' - move.tr());
    //     if (move.pr()) {
    //         std::cout << GLYPHS[move.sp()];
    //     }
    //     std::cout << "\n";
    // }

    // test (broken)

    // std::cout << "MOVETABLE = [";
    // for (uint16_t code = 0; code < 44304; ++ code) {
    //     if (code > 0) std::cout << ", ";
    //     std::cout << MOVETABLE[code].X;
    // }
    // std::cout << "]\n";

    // test

    // Position P;
    // for (int x = 0; x < 100000; ++ x) {
    //   for (uint16_t code = 0; code < 44304; ++ code) {
    //       P.play2(POSMOVETABLE[code]);
    //   }
    // }
    return 0;
}
