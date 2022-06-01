// version4.cpp
// Shaun Harker
// BSD ZERO CLAUSE LICENSE
// forked from version3.cpp at 2022-05-30-2139

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <sstream>
#include <cassert>

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
typedef uint8_t Square;

std::string square(Square s) {
    std::ostringstream ss;
    ss << char('a' + (s & 0x07));
    ss << char('8' - (s >> 3));
    return ss.str();
}

// debugging tool: type wrapper for cout
struct Vizboard {Bitboard x;};

std::ostream & operator << (std::ostream & stream, Vizboard x) {
  //stream << "as bitset: " << std::bitset<64>(x.x) << "\n";
  for (int row = 0; row < 8; ++ row) {
    for (int col = 0; col < 8; ++ col) {
      stream << ((x.x & (1ULL << (8*row+col))) ? "1" : "0");
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
constexpr Bitboard rank_8       = 0x00000000000000FFULL;
// constexpr Bitboard rank_6       = 0x0000000000FF0000ULL;
// constexpr Bitboard rank_3       = 0x0000FF0000000000ULL;
constexpr Bitboard rank_1       = 0xFF00000000000000ULL;
constexpr Bitboard file_a       = 0x0101010101010101ULL;
constexpr Bitboard file_h       = 0x8080808080808080ULL;
constexpr Bitboard diagonal     = 0x8040201008040201ULL;
constexpr Bitboard antidiagonal = 0x0102040810204080ULL;

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

constexpr uint64_t nw_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((row >= 0) && (col >= 0)) {
        result |= s;
        s >>= 9;
        row -= 1;
        col -= 1;
    }
    return result;
}

constexpr uint64_t n_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while (row >= 0) {
        result |= s;
        s >>= 8;
        row -= 1;
    }
    return result;
}

constexpr uint64_t ne_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((row >= 0) && (col <= 7)) {
        result |= s;
        s >>= 7;
        row -= 1;
        col += 1;
    }
    return result;
}

constexpr uint64_t w_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((col >= 0)) {
        result |= s;
        s >>= 1;
        col -= 1;
    }
    return result;
}

constexpr uint64_t e_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((col <= 7)) {
        result |= s;
        s <<= 1;
        col += 1;
    }
    return result;
}

constexpr uint64_t sw_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((row <= 7) && (col >= 0)) {
        result |= s;
        s <<= 7;
        row += 1;
        col -= 1;
    }
    return result;
}

constexpr uint64_t s_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while (row <= 7) {
        result |= s;
        s <<= 8;
        row += 1;
    }
    return result;
}

constexpr uint64_t se_ray(int8_t row, int8_t col) {
    uint64_t result = 0;
    uint64_t s = 1ULL << (col + 8*row);
    while ((row <= 7) && (col <= 7)) {
        result |= s;
        s <<= 9;
        row += 1;
        col += 1;
    }
    return result;
}

template <typename RayFun>
std::array<uint64_t, 64>
compute_ray(RayFun ray){
    std::array<uint64_t, 64> result;
    for (uint8_t i = 0; i < 64; ++ i) {
        result[i] = ray(i >> 3, i & 7);
    }
    return result;
}

std::array<uint64_t, 64> NW_RAY = compute_ray(nw_ray);

std::array<uint64_t, 64> N_RAY = compute_ray(n_ray);

std::array<uint64_t, 64> NE_RAY = compute_ray(ne_ray);

std::array<uint64_t, 64> W_RAY = compute_ray(w_ray);

std::array<uint64_t, 64> E_RAY = compute_ray(e_ray);

std::array<uint64_t, 64> SW_RAY = compute_ray(sw_ray);

std::array<uint64_t, 64> S_RAY = compute_ray(s_ray);

std::array<uint64_t, 64> SE_RAY = compute_ray(se_ray);

Bitboard rookcollisionfreehash(Square i, Bitboard const& E) {

    if (i >= 64) {
        std::cout << "rcfh i = " << int(i) << "\n";
        std::cout.flush();
        abort();
    }
    // Given a chessboard square i and the Bitboard of empty squares
    // on it's "+"-mask, this function determines those squares
    // a rook or queen is "attacking".
    // E is empty squares intersected with rook "+"-mask
    auto constexpr A = antidiagonal;
    auto constexpr T = rank_8;
    auto constexpr L = file_a;
    auto X = T & (E >> (i & 0b111000));
    auto Y = (A * (L & (E >> (i & 0b000111)))) >> 56;
    return (Y << 14) | (X << 6) | i;
}

Bitboard bishopcollisionfreehash(Square i, Bitboard const& E) {
    if (i >= 64) {
        std::cout << "bcfh i = " << int(i) << "\n";
        std::cout.flush();
        abort();
    }
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
  return (x * 0x0202020202ULL & 0x010884422010ULL) % 1023;
  // return __builtin_bitreverse8(x);
}

uint8_t nw_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= NW_RAY[(row << 3) | col];
    x <<= (8*(7-row) + (7-col)); // <<= 63 - i   (0x3F ^ i)
    return bitreverse8((file_a * x) >> 56);
}

uint8_t n_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= N_RAY[(row << 3) | col];
    x <<= (8 * (7 - row) + (7 - col));
    x >>= 7;
    return (diagonal * x) >> 56;
}

uint8_t ne_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= NE_RAY[(row << 3) | col];
    x <<= 8 * (7 - row);
    x >>= col;
    return (file_a * x) >> 56;
}

uint8_t w_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= W_RAY[(row << 3) | col];
    x <<= (8 * (7 - row) + (7 - col));
    x >>= 56;
    return bitreverse8(x);
}

uint8_t e_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= E_RAY[(row << 3) | col];
    x >>= (8 * row + col);
    return x;
}

uint8_t sw_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= SW_RAY[(row << 3) | col];
    x >>= 8 * row;
    x <<= 7 - col;
    return bitreverse8((file_a * x) >> 56);
}

uint8_t s_scan (Bitboard x, uint8_t row, uint8_t col) {
    x &= S_RAY[(row << 3) | col];
    x >>= (8 * row + col);
    return (antidiagonal * x) >> 56;
}

uint8_t se_scan (Bitboard x, uint8_t row, uint8_t col) {
    uint64_t z = (file_a * ((x & SE_RAY[(row << 3) | col]) >> (8 * row + col))) >> 56;
    return z;
}

std::array<std::pair<uint8_t,uint8_t>, (1 << 24)> compute_cap() {
    // compute checks and pins on every possible ray
    //
    // table indexing scheme:
    // +------------------------------------+
    // | cap address bits                   |
    // +-----------+-----------+------------+
    // | 0x00-0x07 | 0x08-0x0F | 0x10-0x018 |
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
    for (uint32_t x = 0; x < (1 << 24); ++ x) {
        // 0xFE not 0xFF to ignore zeroeth square
        uint8_t slider = x & 0xFE;
        uint8_t us = (x >> 8) & 0xFE;
        uint8_t them = (x >> 16) & 0xFE;
        uint8_t checker;
        if (slider & them) {
            checker = ntz(slider & them);
        } else {
            result[x] = {0,0};
            continue;
        }
        uint8_t mask = ((1 << checker) - 1) ^ 1;
        if (them & mask) {
            result[x] = {0,0};
            continue;
        }
        uint8_t pcnt = popcount(us & mask);
        switch (pcnt) {
            case 0:
                result[x] = {checker, 0};
                break;
            case 1:
                result[x] = {checker, ntz(us & mask)};
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
    std::vector<Bitboard> result (1ULL << 22);
    for (Square i = 0; i < 64; ++ i) {
        Bitboard x = 1ULL << i;
        auto const row = i >> 3;
        auto const col = i & 7;
        for (int k = 0x0000; k <= 0xFFFF; k += 0x0001) {
            Bitboard E = Bitboard(0);
            for (int d = 0; d < 8; ++d) {
                E |= (k & (1 << d)) ? (1ULL << (8*row + d)) : 0;
            }
            for (int d = 0; d < 8; ++d) {
                E |= (k & (1 << (8+d))) ? (1ULL << (8*d + col)) : 0;
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
                E |= (k & (1 << d)) ? (1ULL << (8*r + d)) : 0;
            }
            for (int d = 0; d < 8; ++d) {
                Square r = row - col + d;
                if (r < 0 || r >= 8) continue;
                E |= (k & (1 << (8+d))) ? (1ULL << (8*r + d)) : 0;
            }
            // E is empty squares intersected with bishop "x"-mask possibility
            auto idx = bishopcollisionfreehash(i, E);
            for (auto f: {nw, ne, sw, se}) {
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
    // (i.e. intersect rays and exclude square t but not square s)
    std::vector<uint64_t> result;
    for (uint8_t ti = 0; ti < 64; ++ ti) {
        uint8_t tc = ti & 7; uint8_t tr = ti >> 3;
        uint64_t t = 1ULL << ti;
        for (uint8_t si = 0; si < 64; ++ si) {
            uint8_t sc = si & 7; uint8_t sr = si >> 3;
            if (sc == tc) {
                if (sr < tr) {
                    result.push_back(t ^ (N_RAY[ti] & S_RAY[si]));
                } else { // sr >= tr
                    result.push_back(t ^ (N_RAY[si] & S_RAY[ti]));
                }
            } else if (sr == tr) {
                if (sc < tc) {
                    result.push_back(t ^ (W_RAY[ti] & E_RAY[si]));
                } else { // sr >= tr
                    result.push_back(t ^ (W_RAY[si] & E_RAY[ti]));
                }
            } else if (sr + sc == tr + tc) {
                if (sc < tc) {
                    result.push_back(t ^ (SW_RAY[ti] & NE_RAY[si]));
                } else { // sr >= tr
                    result.push_back(t ^ (SW_RAY[si] & NE_RAY[ti]));
                }
            } else if (sr + tc == tr + sc) {
                if (sr < tr) {
                    result.push_back(t ^ (SE_RAY[si] & NW_RAY[ti]));
                } else { // sr >= tr
                    result.push_back(t ^ (SE_RAY[ti] & NW_RAY[si]));
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
    uint64_t rcfh = rookcollisionfreehash(i, empty & ROOKMASK[i]);
    return ROOKTHREATS[rcfh];
}

Bitboard const& bishopthreats(Square i, Bitboard const& empty) {
    uint64_t bcfh = bishopcollisionfreehash(i, empty & BISHOPMASK[i]);
    return BISHOPTHREATS[bcfh];
}

Bitboard queenthreats(Square i, Bitboard const& empty) {
    uint64_t rcfh = rookcollisionfreehash(i, empty & ROOKMASK[i]);
    uint64_t bcfh = bishopcollisionfreehash(i, empty & BISHOPMASK[i]);
    return ROOKTHREATS[rcfh] | BISHOPTHREATS[bcfh];
}

Bitboard const& knightthreats(Square i) {
    return KNIGHTTHREATS[i];
}

Bitboard const& kingthreats(Square i) {
    return KINGTHREATS[i];
}

template <bool color>
Bitboard pawnthreats(Bitboard const& X) {
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

    constexpr Move (uint8_t tc, uint8_t tr, bool bqcr, bool bkcr, uint8_t sc, uint8_t sr, bool wqcr, bool wkcr, uint8_t cp, uint8_t sp, bool pr, bool color, uint8_t epc0, bool ep0, uint8_t epc1, bool ep1) :
        tc_tr_bqcr_bkcr((tc & 7) | ((tr & 7) << 3) | (bqcr << 6) | (bkcr << 7)),
        sc_sr_wqcr_wkcr((sc & 7) | ((sr & 7) << 3) | (wqcr << 6) | (wkcr << 7)),
        cp_sp_pr_c((cp & 7) | ((sp & 7) << 3) | (pr << 6) | (color << 7)),
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

    std::string repr() const {
        std::ostringstream ss;
        if (pr() || (sp() == PAWN)) {
            if (sc() != tc()) {
                ss << char('a' + sc()) << "x";
            }
            ss << square(ti());
            if (pr()) {
                ss << "=" << GLYPHS[sp()];
            }
        } else if (sp() == KING) {
            if (sc() == 4) {
                if (tc() == 6) {
                    ss << "O-O";
                } else if (tc() == 2) {
                    ss << "O-O-O";
                } else {
                    ss << "K";
                    if(cp() != SPACE) ss << "x";
                    ss << square(ti());
                }
            } else {
                ss << "K";
                if(cp() != SPACE) ss << "x";
                ss << square(ti());
            }
        } else {
            switch (sp()) {
                case KNIGHT: ss << "N"; break;
                case BISHOP: ss << "B"; break;
                case ROOK: ss << "R"; break;
                case QUEEN: ss << "Q"; break;
            }
            ss << square(si());
            if (cp() != SPACE) ss << "x";
            ss << square(ti());
        }
        return ss.str();
    }

    // queries
    constexpr uint64_t s() const {return 1ULL << si();}
    constexpr uint64_t t() const {return 1ULL << ti();}
    constexpr uint64_t st() const {return s() | t();}
    constexpr uint64_t ui() const {return (sr() << 3) | tc();}
    constexpr uint64_t u() const {return 1ULL << ui();}
    constexpr uint8_t cr() const {return (wkcr() ? 0x01 : 0x00) | (wqcr() ? 0x02 : 0x00) | (bkcr() ? 0x04 : 0x00) | (bqcr() ? 0x08 : 0x00);}
    constexpr bool is_ep() const {
        bool result = (sp() == PAWN) && (sc() != tc()) && (cp() == SPACE);
        return result;
    }
};

struct Chessboard {
    // We store a chess position with 8 bitboards, two for
    // colors and six for pieces. We keep castling rights
    // bitwise in uint8_t cr_, the en passant column in
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
    uint8_t cr_; // castling rights. bit 0 1 2 3 ~ wk wq bk bq
    uint8_t epc_; // if ep_, col of last double push; else 0
    bool ep_; // true if last move was double push
    bool c_; // false when white to move, true when black to move
    uint8_t halfmove_; // number of halfmoves (plies) since last capture or pawn push
    uint16_t fullmove_; // number of full moves (starts at 1 by convention)

    Chessboard() {
        white = 0xFFFF000000000000; // rank_1 | rank_2;
        black = 0x000000000000FFFF; // rank_7 | rank_8;
        king = 0x1000000000000010; // e1 | e8;
        pawn = 0x00FF00000000FF00; // rank_2 | rank_7
        queen = 0x0800000000000008; // d1 | d8
        rook = 0x8100000000000081; // a1 | a8 | h1 | h8;
        bishop = 0x2400000000000024; // c1 | c8 | f1 | f8;
        knight = 0x4200000000000042; // b1 | b8 | g1 | g8;
        cr_ = 0x0F; // castling rights
        epc_ = 0x00; // en passant column (should be zero when ep_ == false)
        ep_ = false; // true if previous move was double push
        c_ = false; // true if black to move
        halfmove_ = 0;
        fullmove_ = 1;
    }

    Chessboard(std::string fen) {
        auto n = fen.size();
        uint64_t i = 1;
        int reading_mode = 0;
        int clk_start = fen.size();
        bool fen1 = false; // if active color read
        bool fen2 = false; // if castling rights read
        bool fen3 = false; // if ep square read
        bool fen4 = false; // if halfmove clock read
        white = black = king = pawn = queen = rook = bishop = knight = 0ULL;
        cr_ = 0;
        epc_ = 0;
        ep_ = true;
        for (uint8_t k = 0; k < n; ++ k) {
            switch (reading_mode) {
                case 0:  // read board
                    switch (fen[k]) {
                        case 'p':
                            pawn |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'n':
                            knight |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'b':
                            bishop |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'r':
                            rook |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'q':
                            queen |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'k':
                            king |= i;
                            black |= i;
                            i <<= 1;
                            break;
                        case 'P':
                            pawn |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case 'N':
                            knight |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case 'B':
                            bishop |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case 'R':
                            rook |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case 'Q':
                            queen |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case 'K':
                            king |= i;
                            white |= i;
                            i <<= 1;
                            break;
                        case '/':
                            break;
                        case '1':
                            i <<= 1;
                            break;
                        case '2':
                            i <<= 2;
                            break;
                        case '3':
                            i <<= 3;
                            break;
                        case '4':
                            i <<= 4;
                            break;
                        case '5':
                            i <<= 5;
                            break;
                        case '6':
                            i <<= 6;
                            break;
                        case '7':
                            i <<= 7;
                            break;
                        case '8':
                            i <<= 8;
                            break;
                        case ' ':
                            if (i == 1) break; // ignore leading whitespace
                            if (i != 0) std::runtime_error("invalid fen");
                            reading_mode = 1;
                            break;
                        default:
                            throw std::runtime_error("invalid fen");
                    }
                    break;
                case 1: // read active color
                    switch (fen[k]) {
                        case 'w':
                            c_ = false;
                            fen1 = true;
                            break;
                        case 'b':
                            c_ = true;
                            fen1 = true;
                            break;
                        case ' ':
                            if (!fen1) std::runtime_error("invalid fen");
                            reading_mode = 2;
                            break;
                        default:
                            throw std::runtime_error("invalid fen");
                    }
                    break;
                case 2: // read castling rights
                    switch (fen[k]) {
                        case 'K':
                            cr_ |= 1;
                            fen2 = true;
                            break;
                        case 'Q':
                            cr_ |= 2;
                            fen2 = true;
                            break;
                        case 'k':
                            cr_ |= 4;
                            fen2 = true;
                            break;
                        case 'q':
                            cr_ |= 8;
                            fen2 = true;
                            break;
                        case '-':
                            fen2 = true;
                            break;
                        case ' ':
                            if (!fen2) throw std::runtime_error("invalid fen");
                            reading_mode = 3;
                            break;
                    }
                    break;
                case 3: // read ep square
                    switch (fen[k]) {
                        case '-':
                            epc_ = 0;
                            ep_ = false;
                            fen3 = true;
                            break;
                        case 'a':
                            epc_ = 0;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'b':
                            epc_ = 1;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'c':
                            epc_ = 2;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'd':
                            epc_ = 3;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'e':
                            epc_ = 4;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'f':
                            epc_ = 5;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'g':
                            epc_ = 6;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case 'h':
                            epc_ = 7;
                            ep_ = true;
                            fen3 = true;
                            break;
                        case ' ':
                            if (!fen3) throw std::runtime_error("invalid fen");
                            reading_mode = 4;
                            clk_start = k+1;
                            break;
                        default:
                            break;
                    }
                default:
                    break;
            }
        }

        auto clockstr = fen.substr(clk_start, std::string::npos);
        std::istringstream iss(clockstr);
        if (clockstr.size() > 0) {
            int x;
            iss >> x;
            halfmove_ = x;
            iss >> x;
            fullmove_ = x;
        } else {
            halfmove_ = 0;
            fullmove_ = 1;
        }
    }

    Chessboard clone() {
        Chessboard result = *this;
        return result;
    }

    constexpr bool wkcr() const { return cr_ & 1; }
    constexpr bool wqcr() const { return cr_ & 2; }
    constexpr bool bkcr() const { return cr_ & 4; }
    constexpr bool bqcr() const { return cr_ & 8; }
    constexpr uint8_t epc() const { return epc_; }
    constexpr bool ep() const { return ep_; }
    constexpr uint8_t epi() const { return epc_ | (c() ? 0x28 : 0x10); }
    constexpr bool c() const { return c_; }

    void play_move(Move const& move) {
        auto pr = move.pr();
        auto sc = move.sc();
        auto tc = move.tc();
        auto sp = move.sp();
        auto cp = move.cp();
        auto color = move.c();

        if (color == c()) { // forward
            halfmove_ += 1;
            if (color) { // black move
                fullmove_ += 1;
            }
        } else {
            halfmove_ -= 1;
            if (color) {
                fullmove_ -= 1;
            }
        }

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

        cr_ ^= move.cr();

        us ^= st;

        if (pr) {
            pawn ^= s;
            switch (sp) {
                case KNIGHT: knight ^= t; break;
                case BISHOP: bishop ^= t; break;
                case ROOK: rook ^= t; break;
                case QUEEN: queen ^= t; break;
                default: abort(); break; // actually, HCF
            }
        } else {
            switch (sp) {
                case PAWN: pawn ^= st; break;
                case KNIGHT: knight ^= st; break;
                case BISHOP: bishop ^= st; break;
                case ROOK: rook ^= st; break;
                case QUEEN: queen ^= st; break;
                case KING: king ^= st; break;
                default: abort(); break;
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

    void undo_move(Move const& move) {
        play_move(move); // undoes itself
    }

    std::string board() const {
      // display board
      uint64_t s = 1;
      std::ostringstream ss;
      for (int i=0; i<8; ++i) {
        for (int j=0; j<8; ++j) {
          if (s & white) {
            if (s & king) {
              ss << "K";
            } else if (s & queen) {
              ss << "Q";
            } else if (s & bishop) {
              ss << "B";
            } else if (s & knight) {
              ss << "N";
            } else if (s & rook) {
              ss << "R";
            } else if (s & pawn) {
              ss << "P";
            } else {
              ss << "?"; // debug
            }
          } else if (s & black) {
            if (s & king) {
              ss << "k";
            } else if (s & queen) {
              ss << "q";
            } else if (s & bishop) {
              ss << "b";
            } else if (s & knight) {
              ss << "n";
            } else if (s & rook) {
              ss << "r";
            } else if (s & pawn) {
              ss << "p";
            }
          } else {
            ss << ".";
          }
          s <<= 1;
        }
        ss << "\n";
      }
      return ss.str();
    }

    std::string fen() const {
      // display board
      uint64_t s = 1;
      std::ostringstream ss;
      // the board string,
      // e.g. rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR
      char b='0', cc='1';
      for (int i=0; i<8; ++i) {
        if (i > 0) ss << "/";
        for (int j=0; j<8; ++j) {
          if (s & white) {
            if (s & king) {
              cc = 'K';
            } else if (s & queen) {
              cc = 'Q';
            } else if (s & bishop) {
              cc = 'B';
            } else if (s & knight) {
              cc = 'N';
            } else if (s & rook) {
              cc = 'R';
            } else if (s & pawn) {
              cc = 'P';
            }
          } else
          if (s & black) {
            if (s & king) {
              cc = 'k';
            } else if (s & queen) {
              cc = 'q';
            } else if (s & bishop) {
              cc = 'b';
            } else if (s & knight) {
              cc = 'n';
            } else if (s & rook) {
              cc = 'r';
            } else if (s & pawn) {
              cc = 'p';
            }
          }
          if (cc == '1') {
            ++b;
          } else {
            (b > '0') ? (ss << b << cc) : (ss << cc);
            b = '0';
            cc = '1';
          }
          s <<= 1;
        }
        if (b > '0') {
          ss << b;
          b = '0';
        }
      }
      ss << " " << (c() ? "b " : "w ");
      if (wkcr()) ss << "K";
      if (wqcr()) ss << "Q";
      if (bkcr()) ss << "k";
      if (bqcr()) ss << "q";
      ep() ? (ss << " " << square(epi()) << " ") : (ss << " - ");
      ss << int(halfmove_) << " " << int(fullmove_);
      return ss.str();
    }

    template <bool color>
    bool safe(uint8_t si) {
        // determine if si is attacked by "them" as the board stands
        Bitboard const& us = color ? black : white;
        Bitboard const& them = color ? white : black;
        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;
        uint8_t sc = si & 7;
        uint8_t sr = si >> 3;
        for (auto const& f : {n_scan, s_scan, w_scan, e_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qr = f(qr, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qr;
            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                return false;
            }
        }

        //int debugint = 0;
        for (auto const& f : {nw_scan, ne_scan, sw_scan, se_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qb = f(qb, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qb;
            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                return false;
            }
        }

        // knight threats
        if (knightthreats(si) & knight & them) {
            return false;
        }
        // pawn threats
        if (pawnthreats<color>(1ULL << si) & pawn & them) {
            return false;
        }

        // safe!
        return true;
    }

    std::string move_to_san(Move const& move) {
        uint64_t empty = ~(white | black);
        std::ostringstream ss;
        auto disambiguate = [&](uint64_t S) {
            if (popcount(S) == 0) {
                throw std::runtime_error("disambiguating nonexistent move");
            }
            if (popcount(S) == 1) return;
            uint64_t T = S;
            // rebuild S to only care about legal ambiguity
            S = 0;
            while (T) {
                uint8_t ti = ntz(T);
                T &= T-1;
                if (ti == move.si()) {
                    S |= (1ULL << ti);
                } else {
                    auto m = Move(move.tc(), move.tr(), move.bqcr(), move.bkcr(), ti & 7, ti >> 3, move.wqcr(), move.wkcr(), move.cp(), move.sp(), move.pr(), move.c(), move.epc0(), move.ep0(), move.epc1(), move.ep1());
                    play_move(m);
                    uint64_t ok = move.c() ? (black & king) : (white & king);
                    uint8_t oki = ntz(ok);
                    bool moved_into_check = !(move.c() ? safe<true>(oki) : safe<false>(oki));
                    if (!moved_into_check) S |= (1ULL << ti);
                    undo_move(m);
                }
            }
            if (popcount(S) == 1) return;
            if (popcount(S & (file_a << move.sc())) == 1) {
                ss << char('a' + move.sc());
                return;
            }
            if (popcount(S & (rank_8 << (8*move.sr()))) == 1) {
                ss << char('8' - move.sr());
                return;
            }
            ss << char('a' + move.sc()) << char('8' - move.sr());
        };
        if (move.pr() || (move.sp() == PAWN)) {
            if (move.sc() != move.tc()) {
                ss << char('a' + move.sc()) << "x";
            }
            ss << square(move.ti());
            if (move.pr()) {
                if (move.sp() > 6) {
                    throw std::runtime_error("invalid piece");
                }
                ss << "=" << GLYPHS[move.sp()];
            }
        } else if (move.sp() == KING) {
            if (move.sc() == 4) {
                if (move.tc() == 6) {
                    ss << "O-O";
                } else if (move.tc() == 2) {
                    ss << "O-O-O";
                } else {
                    ss << "K";
                    if(move.cp() != SPACE) ss << "x";
                    ss << square(move.ti());
                }
            } else {
                ss << "K";
                if(move.cp() != SPACE) ss << "x";
                ss << square(move.ti());
            }
        } else {
            switch (move.sp()) {
                case KNIGHT:
                    ss << "N";
                    disambiguate(knightthreats(move.ti()) & knight & (move.c() ? black : white));
                    break;
                case BISHOP:
                    ss << "B";
                    disambiguate(bishopthreats(move.ti(), empty) & bishop & (move.c() ? black : white));
                    break;
                case ROOK:
                    ss << "R";
                    disambiguate(rookthreats(move.ti(), empty) & rook & (move.c() ? black : white));
                    break;
                case QUEEN:
                    ss << "Q";
                    disambiguate(queenthreats(move.ti(), empty) & queen & (move.c() ? black : white));
                    break;
            }
            if (move.cp() != SPACE) ss << "x";
            ss << square(move.ti());
        }
        // is the move a check? (note: don't forget discovered checks!)
        play_move(move);
        if (checked()) {
            if (legal_moves().size() == 0) {
                ss << "#";
            } else {
                ss << "+";
            }
        }
        undo_move(move);
        return ss.str();
    }

    Move san_to_move(std::string const& san) {
        std::vector<Move> moves = legal_moves();
        for (Move move : moves) {
            if (move_to_san(move) == san) {
                return move;
            }
        }
        throw std::runtime_error("illegal move");
    }

    void play(std::string moves) {
        std::istringstream iss(moves);
        std::string m;
        while(iss >> m) play_move(san_to_move(m));
    }

    bool mated() {
        bool result = checked() && (legal_moves().size() == 0);
        return result;
    }

    bool checked() {
        uint8_t oki = ntz(king & (c() ? black : white));
        return !(c() ? safe<true>(oki) : safe<false>(oki));
    }

    void add_move_s_t(std::vector<Move> & moves, bool pr, Piece sp, uint8_t si, uint8_t ti) {
        uint64_t t = 1ULL << ti;
        Piece cp;

        // cont
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
        // rights change if they are present and then disturbed
        bool bqcr1 = bqcr() && ((si == 4) || (si == 0) || (ti == 0));
        bool bkcr1 = bkcr() && ((si == 4) || (si == 7) || (ti == 7));
        bool wqcr1 = wqcr() && ((si == 60) || (si == 56) || (ti == 56));
        bool wkcr1 = wkcr() && ((si == 60) || (si == 63) || (ti == 63));
        bool ep1 = (sp == PAWN) && ((si == ti + 16) || (ti == si + 16));
        uint8_t epc1 = ep1 ? tc : 0;
        auto move = Move(tc, tr, bqcr1, bkcr1, sc, sr, wqcr1, wkcr1, cp, sp, pr, c(), epc(), ep(), epc1, ep1);
        //(uint8_t tc, uint8_t tr, bool bqcr, bool bkcr, uint8_t sc, uint8_t sr, bool wqcr, bool wkcr, uint8_t cp, uint8_t sp, bool pr, bool color, uint8_t epc0, bool ep0, uint8_t epc1, bool ep1)
        moves.push_back(move);
    }

    void add_move_s_T(std::vector<Move> & moves, bool pr, Piece sp, uint8_t si, Bitboard T) {
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            add_move_s_t(moves, pr, sp, si, ti);
        }
    }

    template <bool color>
    std::vector<Move> legal_moves_() {
        // Which player is active? (i.e. Whose turn?)
        // Take the perspective of the active player, so
        // it becomes 'us' vs 'them'.
        Bitboard & us = color ? black : white;
        Bitboard & them = color ? white : black;
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
            if (safe<color>(ti)) {
                add_move_s_t(moves, false, KING, oki, ti);
            }
        }

        // put the king back on the board
        us ^= ok;
        king ^= ok;

        // Are we in check?
        uint8_t num_checkers = 0;

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

        auto check_and_pin_search = [&](auto&& f, uint64_t x, int8_t step) {
            uint64_t cap_address = (f(them, okr, okc) << 16) | (f(us, okr, okc) << 8) | f(x, okr, okc);
            auto const& [checker, pin] = CAP[cap_address];
            if (checker != 0) {
               if (pin == 0) {
                 uint8_t ci = oki + step * checker;
                 uint64_t interp_address = (uint64_t(oki) << 6) | uint64_t(ci);
                 targets &= INTERPOSITIONS[interp_address];
                 num_checkers += 1;
               } else {
                 pinned |= 1ULL << (oki + step * pin);
               }
            }
        };

        Bitboard qr = (queen | rook) & them;
        check_and_pin_search(n_scan, qr, -8);
        check_and_pin_search(s_scan, qr,  8);
        check_and_pin_search(w_scan, qr, -1);
        check_and_pin_search(e_scan, qr,  1);

        Bitboard qb = (queen | bishop) & them;
        check_and_pin_search(nw_scan, qb, -9);
        check_and_pin_search(ne_scan, qb, -7);
        check_and_pin_search(sw_scan, qb,  7);
        check_and_pin_search(se_scan, qb,  9);

        // todo: exploit the fact one can't be
        // checked by two non-sliders simultaneously

        // knight checks
        S = knightthreats(oki) & knight & them;

        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          targets &= (1ULL << si);
          num_checkers += 1;
        }

        // pawn checks
        S = pawnthreats<color>(ok) & pawn & them;
        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          uint64_t s = 1ULL << si;
          targets &= s;
          num_checkers += 1;
        }

        if (targets == 0) { // king must move
            return moves;
        }

        if (num_checkers == 0) { // no checks
            // Kingside Castle
            if (color ? bkcr() : wkcr()) {
                Bitboard conf = (color ? 240ULL : (240ULL << 56));
                if (((us & conf) == (color ? 144ULL : (144ULL << 56))) &&
                    ((empty & conf) == (color ? 96ULL : (96ULL << 56))) &&
                    safe<color>(oki+1) && safe<color>(oki+2)) {
                    add_move_s_t(moves, false, KING, oki, oki + 2);
                }
            }

            // Queenside Castle
            if (color ? bqcr() : wqcr()) {
                Bitboard conf = (color ? 31ULL : (31ULL << 56));
                if (((us & conf) == (color ? 17ULL : (17ULL << 56))) &&
                    ((empty & conf) == (color ? 14ULL : (14ULL << 56))) &&
                    safe<color>(oki-1) && safe<color>(oki-2)) {
                    add_move_s_t(moves, false, KING, oki, oki - 2);
                }
            }
        }

        // Queen Moves
        S = queen & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) { // file-pinned queen moves
                    Bitboard F = file_a << sc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if (sr == okr) { // rank-pinned queen moves
                    Bitboard R = rank_8 << (8*sr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if ((sr + sc) == (okr + okc)) { // antidiagonally-pinned queen moves
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else { // sr - sc == okr - okc // diagonally-pinned queen moves
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    add_move_s_T(moves, false, QUEEN, si, T);
                }
            } else { // unpinned queen moves
                uint64_t TR = rookthreats(si, empty) & targets;
                add_move_s_T(moves, false, QUEEN, si, TR);
                uint64_t TB = bishopthreats(si, empty) & targets;
                add_move_s_T(moves, false, QUEEN, si, TB);
            }
        }

        // Rook moves
        S = rook & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) { // file-pinned rook moves
                    Bitboard F = file_a << okc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    add_move_s_T(moves, false, ROOK, si, T);
                } else if (sr == okr) { // rank-pinned rook moves
                    Bitboard R = rank_8 << (8*okr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    add_move_s_T(moves, false, ROOK, si, T);
                }
            } else { // unpinned rook moves
                uint64_t T = rookthreats(si, empty) & targets;
                add_move_s_T(moves, false, ROOK, si, T);
            }
        }

        // Bishop moves
        S = bishop & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc + sr == okr + okc) { // antidiagonally pinned bishop moves
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    add_move_s_T(moves, false, BISHOP, si, T);
                } else if (sr - sc == okr - okc) { // diagonally pinned bishop moves
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    add_move_s_T(moves, false, BISHOP, si, T);
                }
            } else { // unpinned bishop moves
                uint64_t T = bishopthreats(si, empty) & targets;
                add_move_s_T(moves, false, BISHOP, si, T);
            }
        }

        // Knight moves
        S = knight & us & ~pinned;
        while (S) {
            uint8_t si = ntz(S);
            S &= S-1;
            uint64_t T = knightthreats(si) & targets;
            add_move_s_T(moves, false, KNIGHT, si, T);
        }

        // Pawn pushes
        Bitboard our_pawns = pawn & us;
        S = our_pawns & (color ? (empty >> 8) : (empty << 8));
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1ULL << si;
            uint8_t sc = si & 0x07;
            uint8_t ti = si + (color ? 8 : -8);
            uint8_t tr = ti >> 3;
            Bitboard t = 1ULL << ti;
            if (((s & pinned) != 0) && (sc != okc)) continue;
            if ((targets & t) == 0) continue;
            if (tr == 0 || tr == 7) {
                add_move_s_t(moves, true, QUEEN, si, ti);
                add_move_s_t(moves, true, ROOK, si, ti);
                add_move_s_t(moves, true, BISHOP, si, ti);
                add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                add_move_s_t(moves, false, PAWN, si, ti);
            }
        }

        // Pawn captures (except en passant)
        // loop over S first might be better
        T = pawnthreats<color>(our_pawns) & them;
        while (T) {
          auto ti = ntz(T);
          T &= T-1;
          Bitboard t = 1ULL << ti;
          if ((targets & t) == 0) continue;
          S = pawnthreats<!color>(t) & our_pawns;
          while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1ULL << si;
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
            if ((1ULL << ti) & (rank_1 | rank_8)) {
                add_move_s_t(moves, true, QUEEN, si, ti);
                add_move_s_t(moves, true, ROOK, si, ti);
                add_move_s_t(moves, true, BISHOP, si, ti);
                add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                add_move_s_t(moves, false, PAWN, si, ti);
            }
          }
        }

        // Double Pawn pushes
        S = our_pawns & (color ? 0x000000000000FF00ULL :
                               0x00FF000000000000ULL);
        T = empty & (color ? ((S << 16) & (empty << 8))
                           : ((S >> 16) & (empty >> 8)));
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            uint64_t t = (1ULL << ti);
            if ((targets & t) == 0) continue;
            uint8_t si = ti - (color ? 16 : -16);
            Bitboard s = 1ULL << si;
            if (((s & pinned) != 0) && ((si & 0x07) != okc)) continue;
            add_move_s_t(moves, false, PAWN, si, ti);
        }

        // En Passant
        if (ep()) {
            uint64_t t = 1ULL << epi();
            if (!(targets & (color ? (t >> 8) : (t << 8)))) {
                // if in check, ep must capture checker
                // en passant is never an interposition, so this works
                return moves;
            }
            S = pawnthreats<!color>(t) & our_pawns;
            while (S) {
                auto si = ntz(S);
                S &= S-1;
                //  Handle pins:
                Bitboard s = 1ULL << si;
                if (s & pinned) {
                    uint8_t sc = si & 0x07;
                    uint8_t sr = si >> 3;
                    if ((sc == okc) || (sr == okr)) continue;
                    uint8_t ti = epi();
                    uint8_t tc = ti & 0x07;
                    uint8_t tr = ti >> 3;
                    if ((sr + sc) == (okr + okc)) {
                        if ((sr + sc) != (tr + tc)) continue;
                    } else { // sr - sc == okr - okc
                        if ((sr + sc) == (tr + tc)) continue;
                    }
                }
                //  Here we handle missed pawn pins of
                //  the following forms:
                //
                //   pp.p    White is prevented from
                //   ..v.    en passant capture.
                //   rPpK  <-- row 3 = rank 5
                //
                //
                //   RpPk  <-- row 4 = rank 4
                //   ..^.    Black is prevented from
                //   PP.P    en passant capture.
                //
                //  (the v and ^ indicate ep square)
                uint8_t row = color ? 4 : 3;
                bool pin = false;
                if (okr == row) {
                    auto R = (rook & them & (rank_8 << (8*row)));
                    while (R) {
                        auto ri = ntz(R);
                        R &= R-1;
                        uint64_t r = 1ULL << ri;
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
                    add_move_s_t(moves, false, PAWN, si, epi());
                }
            }
        }

        return moves;
    }

    std::vector<Move> legal_moves() {
        return c() ? legal_moves_<true>() : legal_moves_<false>();
    }

    std::vector<std::string> legal() {
        auto moves = legal_moves();
        std::vector<std::string> result {};
        for (auto move : moves) {
            result.push_back(move_to_san(move));
        }
        return result;
    }

    std::string repr_html() {
        std::ostringstream oss;
        std::string tddark("<td style=\"width: 45px; height: 45px; border: 1px solid #000; position: relative; background-color: #b58863;\">");
        std::string tdlight("<td style=\"width: 45px; height: 45px; border: 1px solid #000; position: relative; background-color: #f0d9b5;\">");
        std::string whiteking("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\"> <path d=\"M 22.5,11.63 L 22.5,6\" style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\"/> <path d=\"M 20,8 L 25,8\" style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\"/> <path d=\"M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25\" style=\"fill:#ffffff; stroke:#000000; stroke-linecap:butt; stroke-linejoin:miter;\"/> <path d=\"M 12.5,37 C 18,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 20,16 10.5,13 6.5,19.5 C 3.5,25.5 12.5,30 12.5,30 L 12.5,37\" style=\"fill:#ffffff; stroke:#000000;\"/> <path d=\"M 12.5,30 C 18,27 27,27 32.5,30\" style=\"fill:none; stroke:#000000;\"/> <path d=\"M 12.5,33.5 C 18,30.5 27,30.5 32.5,33.5\" style=\"fill:none; stroke:#000000;\"/> <path d=\"M 12.5,37 C 18,34 27,34 32.5,37\" style=\"fill:none; stroke:#000000;\"/> </g></svg>");
        std::string blackking("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\"> <path d=\"M 22.5,11.63 L 22.5,6\" style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\" id=\"path6570\"/> <path d=\"M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25\" style=\"fill:#000000;fill-opacity:1; stroke-linecap:butt; stroke-linejoin:miter;\"/> <path d=\"M 12.5,37 C 18,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 20,16 10.5,13 6.5,19.5 C 3.5,25.5 12.5,30 12.5,30 L 12.5,37\" style=\"fill:#000000; stroke:#000000;\"/> <path d=\"M 20,8 L 25,8\" style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\"/> <path d=\"M 32,29.5 C 32,29.5 40.5,25.5 38.03,19.85 C 34.15,14 25,18 22.5,24.5 L 22.5,26.6 L 22.5,24.5 C 20,18 10.85,14 6.97,19.85 C 4.5,25.5 13,29.5 13,29.5\" style=\"fill:none; stroke:#ffffff;\"/> <path d=\"M 12.5,30 C 18,27 27,27 32.5,30 M 12.5,33.5 C 18,30.5 27,30.5 32.5,33.5 M 12.5,37 C 18,34 27,34 32.5,37\" style=\"fill:none; stroke:#ffffff;\"/> </g></svg>");
        std::string whitequeen("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"fill:#ffffff;stroke:#000000;stroke-width:1.5;stroke-linejoin:round\"> <path d=\"M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z\"/> <path d=\"M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z\"/> <path d=\"M 11.5,30 C 15,29 30,29 33.5,30\" style=\"fill:none\"/> <path d=\"M 12,33.5 C 18,32.5 27,32.5 33,33.5\" style=\"fill:none\"/> <circle cx=\"6\" cy=\"12\" r=\"2\" /> <circle cx=\"14\" cy=\"9\" r=\"2\" /> <circle cx=\"22.5\" cy=\"8\" r=\"2\" /> <circle cx=\"31\" cy=\"9\" r=\"2\" /> <circle cx=\"39\" cy=\"12\" r=\"2\" /> </g></svg>");
        std::string blackqueen("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\"height=\"45\"> <g style=\"fill:#000000;stroke:#000000;stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round\"> <path d=\"M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z\" style=\"stroke-linecap:butt;fill:#000000\" /> <path d=\"m 9,26 c 0,2 1.5,2 2.5,4 1,1.5 1,1 0.5,3.5 -1.5,1 -1,2.5 -1,2.5 -1.5,1.5 0,2.5 0,2.5 6.5,1 16.5,1 23,0 0,0 1.5,-1 0,-2.5 0,0 0.5,-1.5 -1,-2.5 -0.5,-2.5 -0.5,-2 0.5,-3.5 1,-2 2.5,-2 2.5,-4 -8.5,-1.5 -18.5,-1.5 -27,0 z\" /> <path d=\"M 11.5,30 C 15,29 30,29 33.5,30\" /> <path d=\"m 12,33.5 c 6,-1 15,-1 21,0\" /> <circle cx=\"6\" cy=\"12\" r=\"2\" /> <circle cx=\"14\" cy=\"9\" r=\"2\" /> <circle cx=\"22.5\" cy=\"8\" r=\"2\" /> <circle cx=\"31\" cy=\"9\" r=\"2\" /> <circle cx=\"39\" cy=\"12\" r=\"2\" /> <path d=\"M 11,38.5 A 35,35 1 0 0 34,38.5\" style=\"fill:none; stroke:#000000;stroke-linecap:butt;\" /> <g style=\"fill:none; stroke:#ffffff;\">  <path d=\"M 11,29 A 35,35 1 0 1 34,29\" />  <path d=\"M 12.5,31.5 L 32.5,31.5\" />  <path d=\"M 11.5,34.5 A 35,35 1 0 0 33.5,34.5\" />  <path d=\"M 10.5,37.5 A 35,35 1 0 0 34.5,37.5\" /> </g> </g></svg>");
        std::string whitebishop("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:none; fill-rule:evenodd; fill-opacity:1; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.6)\">> <g style=\"fill:#ffffff; stroke:#000000; stroke-linecap:butt;\">  <path d=\"M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.65,38.99 6.68,38.97 6,38 C 7.35,36.54 9,36 9,36 z\"/>  <path d=\"M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z\"/>  <path d=\"M 25 8 A 2.5 2.5 0 1 1 20,8 A 2.5 2.5 0 1 1 25 8 z\"/> </g> <path d=\"M 17.5,26 L 27.5,26 M 15,30 L 30,30 M 22.5,15.5 L 22.5,20.5 M 20,18 L 25,18\" style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\"/> </g></svg>");
        std::string blackbishop("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:none; fill-rule:evenodd; fill-opacity:1; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.6)\"> <g style=\"fill:#000000; stroke:#000000; stroke-linecap:butt;\">  <path d=\"M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.65,38.99 6.68,38.97 6,38 C 7.35,36.54 9,36 9,36 z\"/>  <path d=\"M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z\"/>  <path d=\"M 25 8 A 2.5 2.5 0 1 1 20,8 A 2.5 2.5 0 1 1 25 8 z\"/> </g> <path d=\"M 17.5,26 L 27.5,26 M 15,30 L 30,30 M 22.5,15.5 L 22.5,20.5 M 20,18 L 25,18\" style=\"fill:none; stroke:#ffffff; stroke-linejoin:miter;\"/> </g></svg>");
        std::string whiteknight("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.3)\"> <path  d=\"M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18\"  style=\"fill:#ffffff; stroke:#000000;\" /> <path  d=\"M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10\"  style=\"fill:#ffffff; stroke:#000000;\" /> <path  d=\"M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z\"  style=\"fill:#000000; stroke:#000000;\" /> <path  d=\"M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z\"  transform=\"matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)\"  style=\"fill:#000000; stroke:#000000;\" /> </g></svg>");
        std::string blackknight("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.3)\"> <path  d=\"M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18\"  style=\"fill:#000000; stroke:#000000;\" /> <path  d=\"M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10\"  style=\"fill:#000000; stroke:#000000;\" /> <path  d=\"M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z\"  style=\"fill:#ffffff; stroke:#ffffff;\" /> <path  d=\"M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z\"  transform=\"matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)\"  style=\"fill:#ffffff; stroke:#ffffff;\" /> <path  d=\"M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z \"  style=\"fill:#ffffff; stroke:none;\" /> </g></svg>");
        std::string whiterook("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:#ffffff; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.3)\"> <path  d=\"M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14\"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 34,14 L 31,17 L 14,17 L 11,14\" /> <path  d=\"M 31,17 L 31,29.5 L 14,29.5 L 14,17\"  style=\"stroke-linecap:butt; stroke-linejoin:miter;\" /> <path  d=\"M 31,29.5 L 32.5,32 L 12.5,32 L 14,29.5\" /> <path  d=\"M 11,14 L 34,14\"  style=\"fill:none; stroke:#000000; stroke-linejoin:miter;\" /> </g></svg>");
        std::string blackrook("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <g style=\"opacity:1; fill:#000000; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\" transform=\"translate(0,0.3)\"> <path  d=\"M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 12.5,32 L 14,29.5 L 31,29.5 L 32.5,32 L 12.5,32 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 14,29.5 L 14,16.5 L 31,16.5 L 31,29.5 L 14,29.5 z \"  style=\"stroke-linecap:butt;stroke-linejoin:miter;\" /> <path  d=\"M 14,16.5 L 11,14 L 34,14 L 31,16.5 L 14,16.5 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14 L 11,14 z \"  style=\"stroke-linecap:butt;\" /> <path  d=\"M 12,35.5 L 33,35.5 L 33,35.5\"  style=\"fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;\" /> <path  d=\"M 13,31.5 L 32,31.5\"  style=\"fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;\" /> <path  d=\"M 14,29.5 L 31,29.5\"  style=\"fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;\" /> <path  d=\"M 14,16.5 L 31,16.5\"  style=\"fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;\" /> <path  d=\"M 11,14 L 34,14\"  style=\"fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;\" /> </g></svg>");
        std::string whitepawn("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <path d=\"m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z\" style=\"opacity:1; fill:#ffffff; fill-opacity:1; fill-rule:nonzero; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:miter; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\"/></svg>");
        std::string blackpawn("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"45\" height=\"45\"> <path d=\"m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z\" style=\"opacity:1; fill:#000000; fill-opacity:1; fill-rule:nonzero; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:miter; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;\"/></svg>");

        oss << "<table>";
        uint64_t s = 1;
        for (uint8_t row = 0; row < 8; ++ row) {
            oss << "<tr>";
            for (uint8_t col = 0; col < 8; ++ col) {
                if ((row + col) & 1) {
                    oss << tddark;
                } else {
                    oss << tdlight;
                }
                if (s & white) {
                    if (s & king) {
                        oss << whiteking;
                    } else if (s & queen) {
                        oss << whitequeen;
                    } else if (s & bishop) {
                        oss << whitebishop;
                    } else if (s & knight) {
                        oss << whiteknight;
                    } else if (s & rook) {
                        oss << whiterook;
                    } else if (s & pawn) {
                        oss << whitepawn;
                    }
                } else if (s & black) {
                    if (s & king) {
                        oss << blackking;
                    } else if (s & queen) {
                        oss << blackqueen;
                    } else if (s & bishop) {
                        oss << blackbishop;
                    } else if (s & knight) {
                        oss << blackknight;
                    } else if (s & rook) {
                        oss << blackrook;
                    } else if (s & pawn) {
                        oss << blackpawn;
                    }
                }
                s <<= 1;
                oss << "</td>";
            }
            oss << "</tr>";
        }
        oss << "</table>";
        return oss.str();
    }

    // some unfortunately repeated code just for perft bragging rights
    template <bool color>
    uint64_t count_moves_() {
        uint64_t result = 0;

        // Which player is active? (i.e. Whose turn?)
        // Take the perspective of the active player, so
        // it becomes 'us' vs 'them'.
        Bitboard & us = color ? black : white;
        Bitboard & them = color ? white : black;
        Bitboard empty = ~(us | them);

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
            if (safe<color>(ti)) result += 1;
        }

        // put the king back on the board
        us ^= ok;
        king ^= ok;

        // Are we in check?
        uint8_t num_checkers = 0;

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

        auto check_and_pin_search = [&](auto&& f, uint64_t x, int8_t step) {
            uint64_t cap_address = (f(them, okr, okc) << 16) | (f(us, okr, okc) << 8) | f(x, okr, okc);
            auto const& [checker, pin] = CAP[cap_address];
            if (checker != 0) {
               if (pin == 0) {
                 uint8_t ci = oki + step * checker;
                 uint64_t interp_address = (uint64_t(oki) << 6) | uint64_t(ci);
                 targets &= INTERPOSITIONS[interp_address];
                 num_checkers += 1;
               } else {
                 pinned |= 1ULL << (oki + step * pin);
               }
            }
        };

        Bitboard qr = (queen | rook) & them;
        check_and_pin_search(n_scan, qr, -8);
        check_and_pin_search(s_scan, qr,  8);
        check_and_pin_search(w_scan, qr, -1);
        check_and_pin_search(e_scan, qr,  1);

        Bitboard qb = (queen | bishop) & them;
        check_and_pin_search(nw_scan, qb, -9);
        check_and_pin_search(ne_scan, qb, -7);
        check_and_pin_search(sw_scan, qb,  7);
        check_and_pin_search(se_scan, qb,  9);

        // todo: exploit the fact one can't be
        // checked by two non-sliders simultaneously

        // knight checks
        S = knightthreats(oki) & knight & them;

        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          targets &= (1ULL << si);
          num_checkers += 1;
        }

        // pawn checks
        S = pawnthreats<color>(ok) & pawn & them;
        while (S) {
          uint8_t si = ntz(S);
          S &= S - 1;
          uint64_t s = 1ULL << si;
          targets &= s;
          num_checkers += 1;
        }

        if (targets == 0) { // king must move
            return result;
        }

        if (num_checkers == 0) { // no checks
            // Kingside Castle
            if (color ? bkcr() : wkcr()) {
                Bitboard conf = (color ? 240ULL : (240ULL << 56));
                if (((us & conf) == (color ? 144ULL : (144ULL << 56))) &&
                    ((empty & conf) == (color ? 96ULL : (96ULL << 56))) &&
                    safe<color>(oki+1) && safe<color>(oki+2)) {
                    result += 1;
                }
            }

            // Queenside Castle
            if (color ? bqcr() : wqcr()) {
                Bitboard conf = (color ? 31ULL : (31ULL << 56));
                if (((us & conf) == (color ? 17ULL : (17ULL << 56))) &&
                    ((empty & conf) == (color ? 14ULL : (14ULL << 56))) &&
                    safe<color>(oki-1) && safe<color>(oki-2)) {
                    result += 1;
                }
            }
        }

        // Queen Moves
        S = queen & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) { // file-pinned queen moves
                    Bitboard F = file_a << sc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    //add_move_s_T(moves, false, QUEEN, si, T);
                    result += popcount(T);
                } else if (sr == okr) { // rank-pinned queen moves
                    Bitboard R = rank_8 << (8*sr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    //add_move_s_T(moves, false, QUEEN, si, T);
                    result += popcount(T);
                } else if ((sr + sc) == (okr + okc)) { // antidiagonally-pinned queen moves
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    //add_move_s_T(moves, false, QUEEN, si, T);
                    result += popcount(T);
                } else { // sr - sc == okr - okc // diagonally-pinned queen moves
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    //add_move_s_T(moves, false, QUEEN, si, T);
                    result += popcount(T);
                }
            } else { // unpinned queen moves
                uint64_t TR = rookthreats(si, empty) & targets;
                // add_move_s_T(moves, false, QUEEN, si, TR);
                result += popcount(TR);
                uint64_t TB = bishopthreats(si, empty) & targets;
                // add_move_s_T(moves, false, QUEEN, si, TB);
                result += popcount(TB);
            }
        }

        // Rook moves
        S = rook & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc == okc) { // file-pinned rook moves
                    Bitboard F = file_a << okc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    // add_move_s_T(moves, false, ROOK, si, T);
                    result += popcount(T);
                } else if (sr == okr) { // rank-pinned rook moves
                    Bitboard R = rank_8 << (8*okr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    // add_move_s_T(moves, false, ROOK, si, T);
                    result += popcount(T);
                }
            } else { // unpinned rook moves
                uint64_t T = rookthreats(si, empty) & targets;
                // add_move_s_T(moves, false, ROOK, si, T);
                result += popcount(T);
            }
        }

        // Bishop moves
        S = bishop & us;
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            if ((1ULL << si) & pinned) {
                uint8_t sc = si & 7;
                uint8_t sr = si >> 3;
                if (sc + sr == okr + okc) { // antidiagonally pinned bishop moves
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    // add_move_s_T(moves, false, BISHOP, si, T);
                    result += popcount(T);
                } else if (sr - sc == okr - okc) { // diagonally pinned bishop moves
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    // add_move_s_T(moves, false, BISHOP, si, T);
                    result += popcount(T);
                }
            } else { // unpinned bishop moves
                uint64_t T = bishopthreats(si, empty) & targets;
                // add_move_s_T(moves, false, BISHOP, si, T);
                result += popcount(T);
            }
        }

        // Knight moves
        S = knight & us & ~pinned;
        while (S) {
            uint8_t si = ntz(S);
            S &= S-1;
            uint64_t T = knightthreats(si) & targets;
            // add_move_s_T(moves, false, KNIGHT, si, T);
            result += popcount(T);
        }

        // Pawn pushes
        Bitboard our_pawns = pawn & us;
        S = our_pawns & (color ? (empty >> 8) : (empty << 8));
        while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1ULL << si;
            uint8_t sc = si & 0x07;
            uint8_t ti = si + (color ? 8 : -8);
            uint8_t tr = ti >> 3;
            Bitboard t = 1ULL << ti;
            if (((s & pinned) != 0) && (sc != okc)) continue;
            if ((targets & t) == 0) continue;
            if (tr == 0 || tr == 7) {
                result += 4;
                // add_move_s_t(moves, true, QUEEN, si, ti);
                // add_move_s_t(moves, true, ROOK, si, ti);
                // add_move_s_t(moves, true, BISHOP, si, ti);
                // add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                result += 1;
                // add_move_s_t(moves, false, PAWN, si, ti);
            }
        }

        // Pawn captures (except en passant)
        // loop over S first might be better
        T = pawnthreats<color>(our_pawns) & them;
        while (T) {
          auto ti = ntz(T);
          T &= T-1;
          Bitboard t = 1ULL << ti;
          if ((targets & t) == 0) continue;
          S = pawnthreats<!color>(t) & our_pawns;
          while (S) {
            auto si = ntz(S);
            S &= S-1;
            Bitboard s = 1ULL << si;
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
            if ((1ULL << ti) & (rank_1 | rank_8)) {
                result += 4;
                // add_move_s_t(moves, true, QUEEN, si, ti);
                // add_move_s_t(moves, true, ROOK, si, ti);
                // add_move_s_t(moves, true, BISHOP, si, ti);
                // add_move_s_t(moves, true, KNIGHT, si, ti);
            } else {
                result += 1;
                // add_move_s_t(moves, false, PAWN, si, ti);
            }
          }
        }

        // Double Pawn pushes
        S = our_pawns & (color ? 0x000000000000FF00ULL :
                               0x00FF000000000000ULL);
        T = empty & (color ? ((S << 16) & (empty << 8))
                           : ((S >> 16) & (empty >> 8)));
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            uint64_t t = (1ULL << ti);
            if ((targets & t) == 0) continue;
            uint8_t si = ti - (color ? 16 : -16);
            Bitboard s = 1ULL << si;
            if (((s & pinned) != 0) && ((si & 0x07) != okc)) continue;
            // add_move_s_t(moves, false, PAWN, si, ti);
            result += 1;
        }

        // En Passant
        if (ep()) {
            uint64_t t = 1ULL << epi();
            if (!(targets & (color ? (t >> 8) : (t << 8)))) {
                // if in check, ep must capture checker
                // en passant is never an interposition, so this works
                return result;
            }
            S = pawnthreats<!color>(t) & our_pawns;
            while (S) {
                auto si = ntz(S);
                S &= S-1;
                //  Handle pins:
                Bitboard s = 1ULL << si;
                if (s & pinned) {
                    uint8_t sc = si & 0x07;
                    uint8_t sr = si >> 3;
                    if ((sc == okc) || (sr == okr)) continue;
                    uint8_t ti = epi();
                    uint8_t tc = ti & 0x07;
                    uint8_t tr = ti >> 3;
                    if ((sr + sc) == (okr + okc)) {
                        if ((sr + sc) != (tr + tc)) continue;
                    } else { // sr - sc == okr - okc
                        if ((sr + sc) == (tr + tc)) continue;
                    }
                }
                //  Here we handle missed pawn pins of
                //  the following forms:
                //
                //   pp.p    White is prevented from
                //   ..v.    en passant capture.
                //   rPpK  <-- row 3 = rank 5
                //
                //
                //   RpPk  <-- row 4 = rank 4
                //   ..^.    Black is prevented from
                //   PP.P    en passant capture.
                //
                //  (the v and ^ indicate ep square)
                uint8_t row = color ? 4 : 3;
                bool pin = false;
                if (okr == row) {
                    auto R = (rook & them & (rank_8 << (8*row)));
                    while (R) {
                        auto ri = ntz(R);
                        R &= R-1;
                        uint64_t r = 1ULL << ri;
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
                    result += 1;
                    // add_move_s_t(moves, false, PAWN, si, epi());
                }
            }
        }
        return result;
    }

    uint64_t count_moves() {
        return c() ? count_moves_<true>() : count_moves_<false>();
    }

};

// Tests -- see https://www.chessprogramming.org/Perft_Results

uint64_t perft(Chessboard & board, uint8_t depth) {
    uint64_t result = 0;
    if (depth == 0) return 1;
    if (depth == 1) return board.count_moves();
    auto legal = board.legal_moves();
    for (auto move : legal) {
        board.play_move(move);
        uint64_t subcnt = perft(board, depth-1);
        result += subcnt;
        board.undo_move(move);
    }
    return result;
}

Chessboard kiwipete() {
    return Chessboard("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
}

Chessboard position3() {
    return Chessboard("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
}

Chessboard position4() {
    return Chessboard("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
}

Chessboard position4R() {
    return Chessboard("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1");
}

Chessboard position5() {
    return Chessboard("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
}

Chessboard position6() {
    return Chessboard("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
}

// main

int main(int argc, char * argv []) {
    std::vector<Chessboard> boards = {Chessboard(), kiwipete(), position3(), position4(), position4R(), position5(), position6()};
    for (auto board : boards) {
        std::cout << board.fen() << "\n";
        for (int d = 0; d < 7; ++ d) {
            auto n = perft(board, d);
            std::cout << d << " " << n << "\n";
            if (n > 100000000) break;
        }
    }
    return 0;
}

// pybind11
// Python Bindings
//
// #include <fstream>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;
//
// PYBIND11_MODULE(chessboard, m) {
//     py::class_<Move>(m, "Move")
//         .def(py::init<>())
//         .def("tc", &Move::tc)
//         .def("tr", &Move::tr)
//         .def("ti", &Move::ti)
//         .def("sc", &Move::sc)
//         .def("sr", &Move::sr)
//         .def("si", &Move::si)
//         .def("sp", &Move::sp)
//         .def("cp", &Move::cp)
//         .def("bqcr", &Move::bqcr)
//         .def("wqcr", &Move::wqcr)
//         .def("bkcr", &Move::bkcr)
//         .def("wkcr", &Move::wkcr)
//         .def("ep0", &Move::ep0)
//         .def("epc0", &Move::epc0)
//         .def("ep1", &Move::ep1)
//         .def("epc1", &Move::epc1)
//         .def("__repr__", &Move::repr);
//
//     py::class_<Chessboard>(m, "Chessboard")
//         .def(py::init<>())
//         .def(py::init<std::string>())
//         .def("fen", &Chessboard::fen)
//         .def("play", &Chessboard::play)
//         .def("legal", &Chessboard::legal)
//         .def("legal_moves", &Chessboard::legal_moves)
//         .def("move_to_san", &Chessboard::move_to_san)
//         .def("san_to_move", &Chessboard::san_to_move)
//         .def("play_move", &Chessboard::play_move)
//         .def("undo_move", &Chessboard::undo_move)
//         .def("board", &Chessboard::board)
//         .def("clone", &Chessboard::clone)
//         .def("checked", &Chessboard::checked)
//         .def("mated", &Chessboard::mated)
//         .def("epc", &Chessboard::epc)
//         .def("ep", &Chessboard::ep)
//         .def("epi", &Chessboard::epi)
//         .def("c", &Chessboard::c)
//         .def("wkcr", &Chessboard::wkcr)
//         .def("wqcr", &Chessboard::wqcr)
//         .def("bkcr", &Chessboard::bkcr)
//         .def("bqcr", &Chessboard::bqcr)
//         .def("_repr_html_", &Chessboard::repr_html)
//         .def("__repr__", &Chessboard::fen);
//
//     m.def("perft", &perft);
//     m.def("kiwipete", &kiwipete);
//     m.def("position3", &position3);
//     m.def("position4", &position4);
//     m.def("position4R", &position4R);
//     m.def("position5", &position5);
//     m.def("position6", &position6);

// }
