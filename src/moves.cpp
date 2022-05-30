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
// constexpr uint64_t SE_MASK = file_a * antidiagonal;
// constexpr uint64_t NW_MASK = (~SE_MASK) | antidiagonal;
// constexpr uint64_t SW_MASK = file_a * diagonal;
// constexpr uint64_t NE_MASK = (~SW_MASK) | diagonal;

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
    auto X = T & (E >> (i & 0b111000));  // 3
    auto Y = (A * (L & (E >> (i & 0b000111)))) >> 56;  // 5
    return (Y << 14) | (X << 6) | i; // 4
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

// constexpr Bitboard off_diagonal(uint8_t row, uint8_t col) {
//   if (row > col) {
//     return diagonal >> (8*(row-col));
//   } else {
//     return diagonal << (8*(col-row));
//   }
// }
//
// constexpr Bitboard off_antidiagonal(uint8_t row, uint8_t col) {
//   if (row + col < 7) {
//     return antidiagonal << (8*(7 - row+col));
//   } else {
//     return antidiagonal >> (8*(row+col - 7));
//   }
// }

uint8_t nw_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "nw_scan\n";
        std::cout.flush();
        abort();
    }
    x &= NW_RAY[(row << 3) | col];
    x <<= (8*(7-row) + (7-col)); // <<= 63 - i   (0x3F ^ i)
    return bitreverse8((file_a * x) >> 56);
}

uint8_t n_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "n_scan\n";
        std::cout.flush();
        abort();
    }
    x &= N_RAY[(row << 3) | col];
    x <<= (8 * (7 - row) + (7 - col));
    x >>= 7;
    //std::cout << "[n_scan " << x << "]";
    return (diagonal * x) >> 56;
}

uint8_t ne_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "ne_scan\n";
        std::cout.flush();
        abort();
    }
    x &= NE_RAY[(row << 3) | col];
    x <<= 8 * (7 - row);
    x >>= col;
    return (file_a * x) >> 56;
}

uint8_t w_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "w_scan\n";
        std::cout.flush();
        abort();
    }
    x &= W_RAY[(row << 3) | col];
    x <<= (8 * (7 - row) + (7 - col));
    x >>= 56;
    return bitreverse8(x);
}

uint8_t e_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "e_scan\n";
        std::cout.flush();
        abort();
    }
    x &= E_RAY[(row << 3) | col];
    x >>= (8 * row + col);
    return x;
}

uint8_t sw_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "sw_scan\n";
        std::cout.flush();
        abort();
    }
    x &= SW_RAY[(row << 3) | col];
    x >>= 8 * row;
    x <<= 7 - col;
    return bitreverse8((file_a * x) >> 56);
}

uint8_t s_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "s_scan\n";
        std::cout.flush();
        abort();
    }
    x &= S_RAY[(row << 3) | col];
    x >>= (8 * row + col);
    return (antidiagonal * x) >> 56;
}

uint8_t se_scan (Bitboard x, uint8_t row, uint8_t col) {
    if ( (row > 7) || (col > 7) ) {
        std::cout << "se_scan\n";
        std::cout.flush();
        abort();
    }
    //std::cout << "se_scan " << std::bitset<64>(x) << " " << int(row) << " " << int(col) << "\n";
    //std::cout << Vizboard({x}) << "\n";
    //std::cout << Vizboard({SE_RAY[(row << 3) | col]}) << "\n";
    uint64_t z = (file_a * ((x & SE_RAY[(row << 3) | col]) >> (8 * row + col))) >> 56;
    //std::cout << " = " << std::bitset<8>(z) << "]\n";
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
    if (i >= 64) {
        std::cout << "rookthreats invalid square " << i << "\n";
        std::cout.flush();
        abort();
    }
    uint64_t rcfh = rookcollisionfreehash(i, empty & ROOKMASK[i]);
    if (rcfh >= ROOKTHREATS.size()) {
        std::cout << "rookthreats rcfh\n";
        std::cout.flush();
        abort();
    }
    return ROOKTHREATS[rcfh];
}

Bitboard const& bishopthreats(Square i, Bitboard const& empty) {
    if (i >= 64) {
        std::cout << "bishopthreats invalid square " << i << "\n";
        std::cout.flush();
        abort();
    }
    uint64_t bcfh = bishopcollisionfreehash(i, empty & BISHOPMASK[i]);
    if (bcfh >= BISHOPTHREATS.size()) {
        std::cout << "bishopthreats bcfh\n";
        std::cout.flush();
        abort();
    }
    return BISHOPTHREATS[bcfh];
}

Bitboard queenthreats(Square i, Bitboard const& empty) {
    if (i >= 64) {
        std::cout << "queenthreats invalid square " << i << "\n";
        std::cout.flush();
        abort();
    }
    uint64_t rcfh = rookcollisionfreehash(i, empty & ROOKMASK[i]);
    uint64_t bcfh = bishopcollisionfreehash(i, empty & BISHOPMASK[i]);
    if (rcfh >= ROOKTHREATS.size()) {
        std::cout << "queenrookthreats rcfh\n";
        std::cout.flush();
        abort();
    }
    if (bcfh >= BISHOPTHREATS.size()) {
        std::cout << "queenbishopthreats bcfh\n";
        std::cout.flush();
        abort();
    }
    return ROOKTHREATS[rcfh] | BISHOPTHREATS[bcfh];
}

Bitboard const& knightthreats(Square i) {
    if (i >= 64) {
        std::cout << "knightthreats invalid square " << i << "\n";
        std::cout.flush();
        abort();
    }
    return KNIGHTTHREATS[i];
}

Bitboard const& kingthreats(Square i) {
    if (i >= 64) {
        std::cout << "kingthreats invalid square " << i << "\n";
        std::cout.flush();
        abort();
    }
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
                    //std::cout << "repr. K" << ((cp()!=SPACE)?"x":"") << square(ti()) << "\n";
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

        // debug
        if (result && (tc() != epc0())) {
            std::cout << "Move::is_ep() contradiction\n";
            std::cout.flush();
            abort();
        }
        return result;
    }
    // feasibility (optional? might need it for future tables)
    // constexpr bool kcr() const {return wkcr() && bkcr();}
    // constexpr bool qcr() const {return wkcr() && bkcr();}
    // constexpr bool wcr() const {return wkcr() && wqcr();}
    // constexpr bool bcr() const {return bkcr() && bqcr();}
    //
    // // TODO: repair this
    // constexpr bool feasible() const {
    //
    //     // sp must be a Piece enum idx
    //     if (sp() > 6) return false;
    //
    //     // can't move from an empty square
    //     if (sp() == SPACE) return false;
    //
    //     // cp must be a Piece enum idx
    //     if (cp() > 6) return false;
    //
    //     // cp may not name KING
    //     if (cp() == KING) return false;
    //
    //     // source != target
    //     if ((sc() == tc()) && (sr() == tr())) return false;
    //
    //     // only pawns promote, and it must be properly positioned
    //     // and cannot promote to pawn or king
    //     if (pr() && ((sr() != (c() ? 6 : 1)) || (tr() != (c() ? 7 : 0)) || (sp() == PAWN) || (sp() == KING))) return false;
    //
    //     // pawns are never on rank 8 or rank 1 (row 0 or row 7)
    //     if ((sp() == PAWN) && ((sr() == 0) ||
    //         (sr() == 7))) return false;
    //     if ((cp() == PAWN) && ((tr() == 0) ||
    //         (tr() == 7))) return false;
    //
    //     if ((sp() == PAWN) || pr()) {
    //         // pawns can only move forward one rank at a time,
    //         // except for their first move
    //         if (sr() != tr() + (c() ? -1 : 1)) {
    //             if ((sr() != (c() ? 1 : 6)) ||
    //                 (tr() != (c() ? 3 : 4))) return false;
    //             // can't capture on double push
    //             if (cp() != SPACE) return false;
    //         }
    //         // pawns stay on file when not capturing,
    //         // and move over one file when capturing.
    //         // i) can't move over more than one file
    //         if (sc()*sc() + tc()*tc() > 1 + 2*sc()*tc()) return false;
    //         // ii) can't capture forward
    //         if ((sc() == tc()) && (cp() != SPACE)) return false;
    //         // iii) can't move diagonal without capture
    //         if ((sc() != tc()) && (cp() == SPACE)) {
    //             // invalid unless possible en passant
    //             if (tr() != (c() ? 5 : 2)) return false;
    //         }
    //     }
    //
    //     if (sp() == KNIGHT) {
    //         // i know how horsies move
    //         if ((sc()*sc() + tc()*tc() + sr()*sr() + tr()*tr())
    //             != 5 + 2*(sc()*tc() + sr()*tr())) return false;
    //     }
    //     if (sp() == BISHOP) {
    //         // bishops move on diagonals and antidiagonals
    //         if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
    //                 (sc() + tr() != tc() + sr())) // not on same diagonal
    //             return false;
    //     }
    //     if (sp() == ROOK) {
    //         // rooks move on ranks and files (rows and columns)
    //         if ((sc() != tc()) && (sr() != tr())) return false;
    //         // conditions where kingside castle right may change
    //         if (kcr() && !((sc() == 7) && (sr() == (c() ? 0 : 7))) && !((tc() == 7) && (tr() == (c() ? 7 : 0)))) return false;
    //         // if losing kingside rights, cannot move to a rook to files a-e
    //         if (kcr() && (tc() < 5)) return false;
    //         // conditions where queenside castle right may change
    //         if (qcr() && !((sc() == 0) && (sr() == (c() ? 0 : 7))) && !((tc() == 0) && (tr() == (c() ? 7 : 0)))) return false;
    //         // if losing queenside rights, cannot move a rook to files e-h
    //         if (qcr() && ((tc() > 3))) return false;
    //     }
    //     if (sp() == QUEEN) {
    //         // queens move on ranks, files, diagonals, and
    //         // antidiagonals.
    //         if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
    //                 (sc() + tr() != tc() + sr()) && // not on same diagonal
    //                 (sc() != tc()) && // not on same file
    //                 (sr() != tr())) // not on same rank
    //             return false;
    //         if ((sc() == tc()) && (sr() == tr())) return false;
    //     }
    //     if (sp() == KING) {
    //         // if kingside castle, must be losing kingside rights
    //         if ((sc() == 4) && (sr() == (c() ? 0 : 7)) && (tc() == 6) && (tr() == (c() ? 0 : 7)) && !kcr()) return false;
    //         // if queenside castle, must be losing queenside rights
    //         if ((sc() == 4) && (sr() == (c() ? 0 : 7)) && (tc() == 2) && (tr() == (c() ? 0 : 7)) && !qcr()) return false;
    //         // king takes rook losing castling rights:
    //         //   only diagonal/antidiagonal captures could
    //         //   possibly occur during play:
    //         if ((cp() == ROOK) && kcr() && (tr() == (c() ? 7 : 0)) && (tc() == 7) && !((sr() == (c() ? 6 : 1)) && (sc() == 6))) return false;
    //         if ((cp() == ROOK) && qcr() && (tr() == (c() ? 7 : 0)) && (tc() == 0) && !((sr() == (c() ? 6 : 1)) && (sc() == 1))) return false;
    //         // castling cannot capture, must be properly positioned
    //         if (sc()*sc() + tc()*tc() > 1 + 2*sc()*tc()) {
    //             if (!((tc() == 6) && kcr()) && !((tc() == 2) && qcr())) return false;
    //             if (cp() != SPACE) return false;
    //             if (sc() != 4) return false;
    //             if (sr() != (c() ? 0 : 7)) return false;
    //             if (tr() != (c() ? 0 : 7)) return false;
    //         }
    //         // kings move to neighboring squares
    //         if (((sc()*sc() + tc()*tc() + sr()*sr()) + tr()*tr() >
    //             2*(1 + sc()*tc() + sr()*tr())) && !((sc() == 4) &&
    //             (sr() == (c() ? 0 : 7)) && (tr()==sr()) &&
    //             (((tc()==2) && qcr()) || ((tc()==6) && kcr()))))
    //             return false;
    //     }
    //     // to change castling rights there are nine cases:
    //     // 1. king move from its home square
    //     // 2. a1 rook captured by black
    //     // 3. h1 rook captured by black
    //     // 4. a8 rook captured by white
    //     // 5. h8 rook captured by white
    //     // 6. a1 rook moved by white
    //     // 7. h1 rook moved by white
    //     // 8. a8 rook moved by black
    //     // 9. h8 rook moved black
    //     // White could capture an unmoved a8 rook with its unmoved a1 rook,
    //     // and similar scenarios, so the cases aren't mutually exclusive.
    //     // it isn't possible to remove castling rights via a Rf8 x Rh8 because the enemy king would be in check. Similarly for other exceptions
    //     bool kingmove = (sp() == KING) && (sr() == (c() ? 0 : 7)) && (sc() == 4);
    //     bool a1rookcapture = (cp() == ROOK) && (ti() == 56) && c();
    //     bool a8rookcapture = (cp() == ROOK) && (ti() == 0) && !c();
    //     bool h1rookcapture = (cp() == ROOK) && (ti() == 63) && c();
    //     bool h8rookcapture = (cp() == ROOK) && (ti() == 7) && !c();
    //
    //     bool a1rookmove = (sp() == ROOK) && (si() == 56) && !c() && (tc() < 4);
    //     bool a8rookmove = (sp() == ROOK) && (si() == 0) && c() && (tc() < 4);
    //     bool h1rookmove = (sp() == ROOK) && (si() == 63) && !c() && (tc() > 4);
    //     bool h8rookmove = (sp() == ROOK) && (si() == 7) && c() && (tc() > 4);
    //     if (kcr() && !(kingmove || h1rookmove || h8rookmove)) {
    //         if (h1rookcapture || h8rookcapture) {
    //             // exclude moves implying a king is en prise
    //             if ((sp() == ROOK) && (sc() < 6)) return false;
    //             if ((sp() == QUEEN) && (sr() == tr()) && (sc() < 6)) return false;
    //             if ((sp() == KING) && ((sr() == tr()) || (sc() == tc()))) return false;
    //         } else {
    //             return false;
    //         }
    //     }
    //     if (qcr() && !(kingmove || a1rookmove || a8rookmove)) {
    //         if (a1rookcapture || a8rookcapture) {
    //             // exclude moves implying a king is en prise
    //             if ((sp() == ROOK) && (sc() > 2)) return false;
    //             if ((sp() == QUEEN) && (sr() == tr()) && (sc() > 2)) return false;
    //             if ((sp() == KNIGHT) && (sr() == (c() ? 6 : 1)) && (sc() == 2)) return false;
    //             if ((sp() == KING) && ((sr() == tr()) || (sc() == tc()))) return false;
    //         } else {
    //                 return false;
    //         }
    //     }
    //     return true;
    // }
};

struct Position {
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

    // DEBUG
    std::vector<Move> tape;

    Position() : tape() {
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
    }

    Position(std::string fen) {
        auto n = fen.size();
        uint64_t i = 1;
        int reading_mode = 0;
        bool fen1 = false; // if active color read
        bool fen2 = false; // if castling rights read
        //bool fen3 = false; // if ep square read
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
                            break;
                        case 'a':
                            epc_ = 0;
                            ep_ = true;
                            break;
                        case 'b':
                            epc_ = 1;
                            ep_ = true;
                            break;
                        case 'c':
                            epc_ = 2;
                            ep_ = true;
                            break;
                        case 'd':
                            epc_ = 3;
                            ep_ = true;
                            break;
                        case 'e':
                            epc_ = 4;
                            ep_ = true;
                            break;
                        case 'f':
                            epc_ = 5;
                            ep_ = true;
                            break;
                        case 'g':
                            epc_ = 6;
                            ep_ = true;
                            break;
                        case 'h':
                            epc_ = 7;
                            ep_ = true;
                            break;
                        default:
                            break;
                    }
                default:
                    break;
            }
        }
    }

    void reset() {
        *this = Position();
    }

    Position clone() {
        Position result = *this;
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

    // void play(Position rhs) {
    //     pawn ^= rhs.pawn;
    //     knight ^= rhs.knight;
    //     bishop ^= rhs.bishop;
    //     rook ^= rhs.rook;
    //     queen ^= rhs.queen;
    //     king ^= rhs.king;
    //     white ^= rhs.white;
    //     black ^= rhs.black;
    //     cr_ ^= rhs.cr_;
    //     epc_ ^= rhs.epc_;
    //     ep_ ^= rhs.ep_;
    //     c_ ^= rhs.c_;
    // }

    void play(Move const& move) {
        auto pr = move.pr();
        auto sc = move.sc();
        auto tc = move.tc();
        auto sp = move.sp();
        auto cp = move.cp();
        auto color = move.c();

        // DEBUG
        if (color == c()) {
            tape.push_back(move);
        } else {
            if (tape.empty()) {
                std::cout << "invalid tape pop\n";
                std::cout.flush();
                abort();
            }
            tape.pop_back();
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

    void undo(Move const& move) {
        play(move); // undoes itself
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
      //ss << halfmove << " " << fullmove; { HOLE }
      return ss.str();
    }

    void add_move_s_t(std::vector<Move> & moves, bool pr, Piece sp, uint8_t si, uint8_t ti) {
        uint64_t t = 1ULL << ti;
        Piece cp;

        // debug
        if (sp > 6) {
            std::cout << "invalid piece\n";
            std::cout.flush();
            abort();
        }
        if (si >= 64) {
            std::cout << "invalid si\n";
            std::cout.flush();
            abort();
        }
        if (ti >= 64) {
            std::cout << "invalid ti\n";
            std::cout.flush();
            abort();
        }

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

    bool safe(uint8_t si, bool color) {
        // determine if si is attacked by "them" as the board stands
        Bitboard const& us = color ? black : white;
        Bitboard const& them = color ? white : black;

        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;

        if (si >= 64) {
            std::cout << "safe(" << int(si) << (color ? " black)\n" : " white)\n");
            print_tape();
            std::cout << "US\n";
            std::cout << Vizboard({us}) << "\n";
            std::cout << "THEM\n";
            std::cout << Vizboard({them}) << "\n";
            std::cout << "BOARD\n";
            std::cout << board() << "\n";
            std::cout << "FEN\n";
            std::cout << fen() << "\n";
            std::cout.flush();
            abort();
        }

        uint8_t sc = si & 7;
        uint8_t sr = si >> 3;
        for (auto const& f : {n_scan, s_scan, w_scan, e_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qr = f(qr, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qr;

            if (address >= CAP.size()) {
                std::cout << "check. CAP whoopsie\n";
                std::cout << "address = " << int(address) << "\n";
                std::cout.flush();
                abort();
            }

            auto const& [checker, pin] = CAP[address];

            if (checker != 0 && pin == 0) {
                //std::cout << "rook " << int(checker) << " " << int(pin) << "\n";
                return false;
            }
        }

        //int debugint = 0;
        for (auto const& f : {nw_scan, ne_scan, sw_scan, se_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qb = f(qb, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qb;

            // std::cout << debugint << "...\n"; ++ debugint;
            // std::cout << std::bitset<8>(f_them) << "\n";
            // std::cout << std::bitset<8>(f_us) << "\n";
            // std::cout << std::bitset<8>(f_qb) << "\n";

            if (address >= CAP.size()) {
                std::cout << "check bish. CAP whoopsie\n";
                std::cout << "address = " << int(address) << "\n";
                std::cout.flush();
                abort();
            }

            auto const& [checker, pin] = CAP[address];


            if (checker != 0 && pin == 0) {
                // std::cout << "bish " << int(checker) << " " << int(pin) << "\n";
                // std::cout << "sr = " << int(sr) << "\n";
                // std::cout << "sc = " << int(sc) << "\n";
                // std::cout << std::bitset<8>(f_them) << "\n";
                // std::cout << std::bitset<8>(f_us) << "\n";
                // std::cout << std::bitset<8>(f_qb) << "\n";
                // std::cout << Vizboard({them}) << "\n";
                // std::cout << Vizboard({us}) << "\n";
                // std::cout << Vizboard({qb}) << "\n";
                // std::cout << Vizboard({SW_RAY[49]}) << "\n";
                return false;
            }
        }

        // knight threats
        if (knightthreats(si) & knight & them) {
            // std::cout << "Check Type 3\n";
            return false;
        }
        // pawn threats
        if (pawnthreats(1ULL << si, color) & pawn & them) {
            // std::cout << "Check Type 4\n";
            return false;
        }

        // safe!
        return true;
    }

    bool king_en_prise() {
        // Used for debugging; if true this implies
        // the board is in an invalid state and
        // the enemy king is already in check.

        // the position is possible only if:
        // 1. the non-active colored king is not en prise
        // 2. there are no pawns on rank 1 or rank 8
        // 3. moved rooks/kings imply missing castling rights
        // 4. ep square implies non-active colored pawn
        //    on that square
        // 5. other conditions such as "each side has
        //    at most 8 pawns, 1 king, and 16 pieces"
        // We only bother with (1) here, and call
        // that "legal".

        // their king
        uint64_t tk = king & (c() ? white : black);
        uint8_t tki = ntz(tk);
        return !safe(tki, !c());
    }

    uint8_t num_attackers(uint8_t si, bool color) {
        // almost identical to safe... DRY?
        // determine if si is attacked by "them" as the board stands
        uint8_t result = 0;

        Bitboard const& us = color ? black : white;
        Bitboard const& them = color ? white : black;

        Bitboard qr = (queen | rook) & them;
        Bitboard qb = (queen | bishop) & them;

        if (si >= 64) {
            std::cout << "safe(" << int(si) << (color ? " black)\n" : " white)\n");
            print_tape();
            std::cout << "US\n";
            std::cout << Vizboard({us}) << "\n";
            std::cout << "THEM\n";
            std::cout << Vizboard({them}) << "\n";
            std::cout << "BOARD\n";
            std::cout << board() << "\n";
            std::cout << "FEN\n";
            std::cout << fen() << "\n";
            std::cout.flush();
            abort();
        }

        uint8_t sc = si & 7;
        uint8_t sr = si >> 3;
        for (auto const& f : {n_scan, s_scan, w_scan, e_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qr = f(qr, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qr;

            if (address >= CAP.size()) {
                std::cout << "check. CAP whoopsie\n";
                std::cout << "address = " << int(address) << "\n";
                std::cout.flush();
                abort();
            }

            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                result += 1;
            }
        }

        for (auto const& f : {nw_scan, ne_scan, sw_scan, se_scan}) {
            uint8_t f_them = f(them, sr, sc);
            uint8_t f_us = f(us, sr, sc);
            uint8_t f_qb = f(qb, sr, sc);
            uint32_t address = (f_them << 16) | (f_us << 8) | f_qb;

            if (address >= CAP.size()) {
                std::cout << "check bish. CAP whoopsie\n";
                std::cout << "address = " << int(address) << "\n";
                std::cout.flush();
                abort();
            }

            auto const& [checker, pin] = CAP[address];
            if (checker != 0 && pin == 0) {
                result += 1;
            }
        }

        // knight threats
        if (knightthreats(si) & knight & them) {
            // std::cout << "Check Type 3\n";
            result += 1;
        }
        // pawn threats
        if (pawnthreats(1ULL << si, color) & pawn & them) {
            // std::cout << "Check Type 4\n";
            result += 1;
        }

        return result;
    }

    std::string move_to_san(Move const& move) {
        //std::cout << "move_to_san start\n";
        //std::cout.flush();
        uint64_t empty = ~(white | black);
        std::ostringstream ss;
        auto disambiguate = [&](uint64_t S) {
            if (popcount(S) == 0) {
                throw std::runtime_error("disambiguating nonexistent move");
            }
            if (popcount(S) == 1) return;
            //std::cout << "disambiguate\n";
            //std::cout.flush();
            uint64_t T = S;
            // rebuild S to only care about legal ambiguity
            S = 0;
            while (T) {
                uint8_t ti = ntz(T);
                T &= T-1;
                //std::cout << "disambiguate loop top with ti = " << ti << "\n";
                //std::cout.flush();
                if (ti == move.si()) {
                    S |= (1ULL << ti);
                } else {
                    auto m = Move(move.tc(), move.tr(), move.bqcr(), move.bkcr(), ti & 7, ti >> 3, move.wqcr(), move.wkcr(), move.cp(), move.sp(), move.pr(), move.c(), move.epc0(), move.ep0(), move.epc1(), move.ep1());
                    //std::cout << "Investigating alternate " << m.repr() << "\n";
                    //std::cout.flush();
                    play(m);
                    //std::cout << "played alternative\n";
                    uint64_t ok = move.c() ? (black & king) : (white & king);
                    uint8_t oki = ntz(ok);
                    //std::cout << "move_to_san 3 1\n";
                    //std::cout.flush();
                    if (safe(oki, move.c())) {
                        S |= (1ULL << ti);
                    }
                    //std::cout << "move_to_san lookahead 4\n";
                    //std::cout.flush();
                    undo(m);
                    //std::cout << "move_to_san lookahead 5\n";
                    //std::cout.flush();

                }
            }
            if (popcount(S) == 1) return;
            //std::cout << "disambiguate end A\n";
            //std::cout.flush();
            if (popcount(S & (file_a << move.sc())) == 1) {
                ss << char('a' + move.sc());
                return;
            }
            //std::cout << "disambiguate emd B\n";
            //std::cout.flush();
            if (popcount(S & (rank_8 << (8*move.sr()))) == 1) {
                ss << char('8' - move.sr());
                return;
            }
            //std::cout << "disambiguate end C\n";
            //std::cout.flush();
            ss << char('a' + move.sc()) << char('8' - move.sr());
            //std::cout << "disambiguate end D\n";
            //std::cout.flush();
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
        play(move);
        if (checked()) {
            if (legal_moves().size() == 0) {
                ss << "#";
            } else {
                ss << "+";
            }
        }
        undo(move);
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
        // char c = san[0];
        // uint8_t sp, si, ti
        // switch (c) {
        //     case 'K':
        //
        //         break;
        //     case 'Q':
        //         break;
        //     case 'R':
        //         break;
        //     case 'B':
        //         break;
        //     case 'N':
        //         break;
        //     default:
        // }
    }

    void print_tape() {
        std::cout << "print_tape\n";
        std::cout.flush();
        for (auto move : tape) {
            if (move.sp() > 6) {
                std::cout << "INVALID GLYPH\n";
                abort();
            }
            if(!move.pr() && (move.sp() != PAWN)) std::cout << GLYPHS[move.sp()];
            std::cout << char('a' + move.sc()) << char('8' - move.sr()) << ((move.cp() == SPACE) ? "" : "x") << char('a' + move.tc()) << char('8' - move.tr());
            if(move.pr()) std::cout << "=" << GLYPHS[move.sp()];
            std::cout << " ";
        }
        std::cout << "\n";
    }

    bool mated() {
        bool result = checked() && (legal_moves().size() == 0);

        // if (!result && checked()) {
        //     std::cout << "This is recorded as a check but not a mate:\n";
        //     print_tape();
        //     std::cout << "The claimed allowed moves are:\n";
        //     for (auto move : legal_moves()) {
        //         std::cout << san_from_move(move) << " ";
        //     }
        //     std::cout << "\n";
        // }
        return result;
    }

    bool checked() {
        uint8_t oki = ntz(king & (c() ? black : white));
        if (oki >= 64) {
            std::cout << "checked. " << int(oki) << (c() ? " black)\n" : " white)\n");
            std::cout << Vizboard({king}) << "\n";
            std::cout << Vizboard({(c() ? black : white)}) << "\n";

            std::cout.flush();
            abort();
        }
        return !safe(oki, c());
    }

    bool doublechecked() {
        uint8_t oki = ntz(king & (c() ? black : white));
        if (oki >= 64) {
            std::cout << "doublechecked. " << int(oki) << (c() ? " black)\n" : " white)\n");
            std::cout << Vizboard({king}) << "\n";
            std::cout << Vizboard({(c() ? black : white)}) << "\n";

            std::cout.flush();
            abort();
        }
        return num_attackers(oki, c()) == 2;
    }

    std::vector<Move> legal_moves() {
        // Step 1. Which player is active? (i.e. Whose turn?)
        // Take the perspective of the active player, so
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

        if (oki >= 64) {
            std::cout << "legal_moves(" << int(oki) << (c() ? " black)\n" : " white)\n");
            print_tape();
            std::cout << "US\n";
            std::cout << Vizboard({us}) << "\n";
            std::cout << "THEM\n";
            std::cout << Vizboard({them}) << "\n";
            std::cout << "BOARD\n";
            std::cout << board() << "\n";
            std::cout << "FEN\n";
            std::cout << fen() << "\n";
            std::cout.flush();
            abort();
        }

        // take our king off the board
        us ^= ok;
        king ^= ok;

        // loop through the possibilities
        uint64_t S, T;

        T = kingthreats(oki) & ~us;
        while (T) {
            uint8_t ti = ntz(T);
            T &= (T-1);

            if (ti >= 64) {
                std::cout << "legal_moves kingthreats(" << int(ti) << (c() ? " black)\n" : " white)\n");
                print_tape();
                std::cout << "US\n";
                std::cout << Vizboard({us}) << "\n";
                std::cout << "THEM\n";
                std::cout << Vizboard({them}) << "\n";
                std::cout << "BOARD\n";
                std::cout << board() << "\n";
                std::cout << "FEN\n";
                std::cout << fen() << "\n";
                std::cout.flush();
                abort();
            }
            //std::cout << "A " << int(oki) << " " << int(ti) << "\n";
            if (safe(ti, c())) {
                //std::cout << "B " << int(oki) << " " << int(ti) << "\n";
                // std::cout << "king move\n";
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

            uint64_t address = ((f(them, okr, okc) << 16) | (f(us, okr, okc) << 8) | f(x, okr, okc));

            if (address >= CAP.size()) {
                std::cout << "CAP whoopsie\n";
                std::cout << "address = " << int(address) << "\n";
                std::cout.flush();
                abort();
            }

            auto const& [checker, pin] = CAP[(f(them, okr, okc) << 16) | (f(us, okr, okc) << 8) | f(x, okr, okc)];
            if (checker != 0) {
               if (pin == 0) {
                 uint8_t ci = oki + step * checker;
                 uint64_t address = (uint64_t(oki) << 6) | uint64_t(ci);
                 if (address > INTERPOSITIONS.size()) {
                     std::cout << "INTERPOSITION whoopsie\n";
                     std::cout << "oki = " << int(oki) << "\n";
                     std::cout << "ci = " << int(ci) << "\n";
                     std::cout.flush();
                     abort();
                 }
                 targets &= INTERPOSITIONS[address];
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

        //std::cout << "check and pin search complete\n";

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
        S = pawnthreats(ok, c()) & pawn & them;
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

        //std::cout << "Targets = \n" << Vizboard({targets}) << "\n";

        if (num_checkers == 0) { // no checks
            // Kingside Castle
            if (c() ? bkcr() : wkcr()) {
                if (oki+2 >= 64) {
                    std::cout << "legal_moves kingthreats(" << int(oki+2) << (c() ? " black)\n" : " white)\n");
                    print_tape();
                    std::cout << "US\n";
                    std::cout << Vizboard({us}) << "\n";
                    std::cout << "THEM\n";
                    std::cout << Vizboard({them}) << "\n";
                    std::cout << "BOARD\n";
                    std::cout << board() << "\n";
                    std::cout << "FEN\n";
                    std::cout << fen() << "\n";
                    std::cout.flush();
                    abort();
                }

                Bitboard conf = (c() ? 240ULL : (240ULL << 56));
                if (((us & conf) == (c() ? 144ULL : (144ULL << 56))) &&
                    ((empty & conf) == (c() ? 96ULL : (96ULL << 56))) &&
                    safe(oki+1, c()) && safe(oki+2, c())) {
                    // std::cout << "kingside castle move\n";
                    add_move_s_t(moves, false, KING, oki, oki + 2);
                }
            }

            // Queenside Castle
            if (c() ? bqcr() : wqcr()) {

                if (oki-2 >= 64) {
                    std::cout << "legal_moves kingthreats(" << int(oki+2) << (c() ? " black)\n" : " white)\n");
                    print_tape();
                    std::cout << "US\n";
                    std::cout << Vizboard({us}) << "\n";
                    std::cout << "THEM\n";
                    std::cout << Vizboard({them}) << "\n";
                    std::cout << "BOARD\n";
                    std::cout << board() << "\n";
                    std::cout << "FEN\n";
                    std::cout << fen() << "\n";
                    std::cout.flush();
                    abort();
                }

                Bitboard conf = (c() ? 31ULL : (31ULL << 56));
                if (((us & conf) == (c() ? 17ULL : (17ULL << 56))) &&
                    ((empty & conf) == (c() ? 14ULL : (14ULL << 56))) &&
                    safe(oki-1, c()) && safe(oki-2, c())) {
                    // std::cout << "queenside castle move\n";
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
                if (sc == okc) {
                    Bitboard F = file_a << sc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    // if (T) std::cout << "file-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if (sr == okr) {
                    Bitboard R = rank_8 << (8*sr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    // if (T) std::cout << "rank-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else if ((sr + sc) == (okr + okc)) {
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    // if (T) std::cout << "antidiagonally-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                } else { // sr - sc == okr - okc
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    // if (T) std::cout << "diagonally-pinned queen moves\n";
                    add_move_s_T(moves, false, QUEEN, si, T);
                }
            } else {
                uint64_t TR = rookthreats(si, empty) & targets;
                // if (TR) std::cout << "unpinned queen rook-like move\n";
                add_move_s_T(moves, false, QUEEN, si, TR);
                uint64_t TB = bishopthreats(si, empty) & targets;
                // if (TB) std::cout << "unpinned queen bishop-like move\n";
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
                if (sc == okc) {
                    Bitboard F = file_a << okc;
                    uint64_t T = F & rookthreats(si, empty) & targets;
                    // if (T) std::cout << "file-pinned rook moves\n";
                    add_move_s_T(moves, false, ROOK, si, T);
                } else if (sr == okr) {
                    Bitboard R = rank_8 << (8*okr);
                    uint64_t T = R & rookthreats(si, empty) & targets;
                    // if (T) std::cout << "rank-pinned rook moves\n";
                    add_move_s_T(moves, false, ROOK, si, T);
                }
            } else {
                uint64_t T = rookthreats(si, empty) & targets;
                // if (T) std::cout << "unpinned rook moves\n";
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
                if (sc + sr == okr + okc) {
                    Bitboard A = (sr + sc < 7) ? (antidiagonal >> (8*(7-sr-sc))) : (antidiagonal << (8*(sr+sc-7)));
                    uint64_t T = A & bishopthreats(si, empty) & targets;
                    // if (T) std::cout << "antidiagonally pinned bishop moves\n";
                    add_move_s_T(moves, false, BISHOP, si, T);
                } else if (sr - sc == okr - okc) {
                    Bitboard D = (sr > sc) ? (diagonal << (8*(sr-sc))) : (diagonal >> (8*(sc-sr)));
                    uint64_t T = D & bishopthreats(si, empty) & targets;
                    // if (T) std::cout << "diagonally pinned bishop moves\n";
                    add_move_s_T(moves, false, BISHOP, si, T);
                }
            } else {
                uint64_t T = bishopthreats(si, empty) & targets;
                // if (T) std::cout << "unpinned bishop moves " << si << "\n";
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
            Bitboard s = 1ULL << si;
            uint8_t sc = si & 0x07;
            uint8_t ti = si + (c() ? 8 : -8);
            uint8_t tr = ti >> 3;
            Bitboard t = 1ULL << ti;
            if (((s & pinned) != 0) && (sc != okc)) continue;
            if ((targets & t) == 0) continue;
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
          Bitboard t = 1ULL << ti;
          if ((targets & t) == 0) continue;
          S = pawnthreats(t, !c()) & our_pawns;
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
        S = our_pawns & (c() ? 0x000000000000FF00ULL :
                               0x00FF000000000000ULL);
        T = empty & (c() ? ((S << 16) & (empty << 8))
                           : ((S >> 16) & (empty >> 8)));
        while (T) {
            uint8_t ti = ntz(T);
            T &= T-1;
            uint64_t t = (1ULL << ti);
            if ((targets & t) == 0) continue;
            uint8_t si = ti - (c() ? 16 : -16);
            Bitboard s = 1ULL << si;
            if (((s & pinned) != 0) && ((si & 0x07) != okc)) continue;
            // std::cout << "double pawn push move " << int(si) << " " << int(ti) << " " << okc << " " << pinned << "\n";
            add_move_s_t(moves, false, PAWN, si, ti);

            // debug
            Move m = moves.back();
            if (!m.ep1()) {
                std::cout << "ep1 problem, i think?\n";
                std::cout.flush();
                abort();
            }
        }

        // En Passant
        if (ep()) {
            uint64_t t = 1ULL << epi();
            if (!(targets & (c() ? (t >> 8) : (t << 8)))) {
                // if in check, ep must capture checker
                // en passant is never an interposition, so this works
                return moves;
            }
            S = pawnthreats(t, !c()) & our_pawns;
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
                uint8_t row = c() ? 4 : 3;
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
                    //std::cout << "en passant move " << si << " " << epi() << "\n";
                    add_move_s_t(moves, false, PAWN, si, epi());

                    // debug
                    Move m = moves.back();
                    if ((!m.ep0()) || m.ep1()) {
                        std::cout << "ep flags problem, i think?\n";
                        std::cout.flush();
                        abort();
                    }
                }
            }
        }



        return moves;
    }
};

#if 0

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
#endif

// Tests

uint64_t perft(Position & board,
    uint8_t depth, int vdepth=0) {
    uint64_t result = 0;
    if (depth == 0) return 1;
    auto legal = board.legal_moves();
    if (depth == 1) return legal.size();
    for (auto move : legal) {
        board.play(move);
        uint64_t subcnt = perft(board, depth-1, vdepth-1);
        result += subcnt;
        board.undo(move);
        if (vdepth > 0) std::cout << subcnt << ", ";
    }
    if (vdepth > 0) std::cout << "\n";
    return result;
}

uint64_t capturetest(Position & board, int depth) {
  if (depth == 0) return 0;
  auto moves = board.legal_moves();
  uint64_t result = 0;
  for (auto move : moves) {
    board.play(move);
    if ((depth == 1) && (move.cp() != SPACE || move.is_ep())) result += 1;
    result += capturetest(board, depth-1);
    board.undo(move);
  }
  return result;
}

uint64_t doublepushtest(Position & board, int depth) {
    if (depth == 0) return 0;
    auto moves = board.legal_moves();
    uint64_t result = 0;
    for (auto move : moves) {
        board.play(move);
        if ((depth == 1) && (board.ep())) result += 1;
        result += doublepushtest(board, depth-1);
        board.undo(move);
    }
    return result;
}

uint64_t enpassanttest(Position & board, int depth) {
    if (depth == 0) return 0;
    auto moves = board.legal_moves();
    uint64_t result = 0;
    for (auto move : moves) {
        board.play(move);
        if ((depth == 1) && (move.is_ep())) result += 1;
        result += enpassanttest(board, depth-1);
        board.undo(move);
    }
    return result;
}

uint64_t checktest(Position & board, int depth) {
    if (depth == 0) return board.checked() ? 1 : 0;
    auto moves = board.legal_moves();
    uint64_t result = 0;
    for (auto move : moves) {
        board.play(move);
        result += checktest(board, depth-1);
        board.undo(move);
    }
    return result;
}

uint64_t doublechecktest(Position & board, int depth) {
    if (depth == 0) return board.doublechecked() ? 1 : 0;
    auto moves = board.legal_moves();
    uint64_t result = 0;
    for (auto move : moves) {
        board.play(move);
        result += doublechecktest(board, depth-1);
        board.undo(move);
    }
    return result;
}

uint64_t matetest(Position & board, std::vector<Move> & prev, int depth) {
    //if (depth == 0) return 0;
    if (depth == 0) return (board.mated()) ? 1 : 0;
    auto moves = board.legal_moves();
    uint64_t result = 0;
    for (auto move : moves) {
        board.play(move);
        prev.push_back(move);
        result += matetest(board, prev, depth-1);
        prev.pop_back();
        board.play(move);
    }
    return result;
}

Position kiwipete() {
    return Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
}

Position position3() {
    return Position("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
}

Position position4() {
    return Position("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -");
}

Position position4R() {
    return Position("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ -");
}

Position position5() {
    return Position("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -");
}

// main

int main(int argc, char * argv []) {
    // for (int d = 0; d < 8; ++ d) {
    //     auto P = position3(); // new chessboard
    //     std::cout << "\n----------\ndepth " << d << "\n";
    //     std::cout << "perft "; std::cout.flush();
    //     std::cout << perft(P, d, 0) << "\n";
    // }

    int d = 5;
    auto P = position3(); // new chessboard
    // P.play(P.legal_moves()[2]);
    // P.play(P.legal_moves()[0]);
    // P.play(P.legal_moves()[1]);
    // P.play(P.legal_moves()[16]);
    std::cout << "\n----------\ndepth " << d << "\n";
    std::cout << "perft "; std::cout.flush();
    std::cout << perft(P, d, 1) << "\n";
    for (auto move : P.legal_moves()) {
        std::cout << P.move_to_san(move) << " ";
    }
    std::cout << "\n";
    std::cout << P.board() << "\n";
    return 0;
}

// pybind11
// Python Bindings

#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(chessboard2, m) {
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def("tc", &Move::tc)
        .def("tr", &Move::tr)
        .def("ti", &Move::ti)
        .def("sc", &Move::sc)
        .def("sr", &Move::sr)
        .def("si", &Move::si)
        .def("sp", &Move::sp)
        .def("cp", &Move::cp)
        .def("bqcr", &Move::bqcr)
        .def("wqcr", &Move::wqcr)
        .def("bkcr", &Move::bkcr)
        .def("wkcr", &Move::wkcr)
        .def("ep0", &Move::ep0)
        .def("epc0", &Move::epc0)
        .def("ep1", &Move::ep1)
        .def("epc1", &Move::epc1)
        .def("__repr__", &Move::repr);

    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def("reset", &Position::reset)
        .def("fen", &Position::fen)
        .def("legal_moves", &Position::legal_moves)
        .def("move_to_san", &Position::move_to_san)
        .def("san_to_move", &Position::san_to_move)
        .def("play", &Position::play)
        .def("board", &Position::board)
        .def("clone", &Position::clone)
        .def("safe", &Position::safe)
        .def("num_attackers", &Position::num_attackers)
        .def("checked", &Position::checked)
        .def("mated", &Position::mated)
        .def("epc", &Position::epc)
        .def("ep", &Position::ep)
        .def("epi", &Position::epi)
        .def("c", &Position::c)
        .def("wkcr", &Position::wkcr)
        .def("wqcr", &Position::wqcr)
        .def("bkcr", &Position::bkcr)
        .def("bqcr", &Position::bqcr)
        .def("__repr__", &Position::fen);

    m.def("perft", &perft);
    m.def("kiwipete", &kiwipete);
    m.def("position3", &position3);
    m.def("position4", &position4);
    m.def("position4R", &position4R);
    m.def("position5", &position5);
    m.def("capturetest", &capturetest);
    m.def("checktest", &checktest);
    m.def("enpassanttest", &enpassanttest);
    m.def("doublechecktest", &doublechecktest);
    m.def("matetest", &matetest);
}
