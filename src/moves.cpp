// moves.cpp
// Shaun Harker
// BSD ZERO CLAUSE LICENSE

#include <cstdint>
#include <array>
#include <iostream>

template <typename T> constexpr T sqr(T x) {return x*x;}

constexpr std::array<char,7>
    GLYPHS {'.','P','N','B','R','Q','K'};

struct ThreeByteMove {
    uint32_t X;
    //  bit range  | mode0 |
    // 0x17        | c     | ply % 2
    // 0x14 - 0x16 | spi   | source piece index
    // 0x11 - 0x13 | sr    | source row 87654321
    // 0x0E - 0x10 | sc    | source col abcdefgh
    // 0x0B - 0x0D | tr    | target row 87654321
    // 0x08 - 0x0A | tc    | target col abcdefgh
    // 0x05 - 0x07 | tpi   | target piece idx into .PNBRQK
    // 0x02 - 0x04 | cpi   | capture piece idx into .PNBRQK
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
    constexpr uint8_t spi() const {return (X >> 0x14) & 0x07;}
    constexpr uint8_t si() const {return (X >> 0x0E) & 0x3F;}
    constexpr uint8_t ti() const {return (X >> 0x08) & 0x3F;}
    constexpr uint8_t tpi() const {return (X >> 0x05) & 0x07;}
    constexpr uint8_t cpi() const {return (X >> 0x02) & 0x07;}
    constexpr bool wkrr() const {
      return (sp() == 'R') && (cp() == 'R') && (
       (si() == 63) && (ti() == 7)) && !c();
    }
    constexpr bool bkrr() const {
      return (sp() == 'R') && (cp() == 'R') && (
       (si() == 7) && (ti() == 63)) && c();
    }
    constexpr bool wqrr() const {
      return (sp() == 'R') && (cp() == 'R') && (
       (si() == 56) && (ti() == 0)) && !c();
    }
    constexpr bool bqrr() const {
      return (sp() == 'R') && (cp() == 'R') && (
       (si() == 0) && (ti() == 56)) && c();
    }
    constexpr bool rr() const {
      return wkrr() || bkrr() || wqrr() || bqrr();
    }
    constexpr bool qcr() const {
      // in the rr() case we may compute it:
      if (rr()) return (X & 0x03) && (wqrr() || bqrr());
      // Otherwise it is stored in the 0x02 bit.
      return X & 0x02;
    }
    constexpr bool kcr() const {
      // in the rr() case we may compute it:
      if (rr()) return (X & 0x03) && (wkrr() || bkrr());
      // Otherwise it is stored in the 0x01 bit.
      return X & 0x01;
    }
    constexpr bool wcr() const {
      bool white = !c();
      // in the rr() case this is stored directly:
      if (rr()) return X & 0x01;
      // if no rights are lost then false:
      if (!kcr() && !qcr()) return false;
      // if both rights are lost then it must
      // be a castling or a king move; just
      // return whose turn it is:
      if (kcr() && qcr()) return white;
      // a move from the king home square
      // never removes the enemies castling right
      // so, the removed right must be white's in
      // that case:
      if (si() == 60) return true;
      // we know the move isn't RxR, so:
      if ((sp() == 'R') && ((si() == 56) || (si() == 63)))
        return true;
      // we know the move isn't RxR, so:
      if ((cp() == 'R') && ((ti() == 56) || (ti() == 63)))
        return true;
      // still here?
      return false;
    }
    constexpr bool bcr() const {
      bool white = !c();
      // in the rr() case this is stored directly:
      if (rr()) return X & 0x02;
      // if no rights are lost then false:
      if (!kcr() && !qcr()) return false;
      // if both rights are lost then it must
      // be a castling or a king move; just
      // return whose turn it is:
      if (kcr() && qcr()) return !white;
      // a move from the king home square
      // never removes the enemies castling right
      // so, the removed right must be white's in
      // that case:
      if (si() == 4) return true;
      // we know the move isn't RxR, so:
      if ((sp() == 'R') && ((si() == 0) || (si() == 7)))
        return true;
      // we know the move isn't RxR, so:
      if ((cp() == 'R') && ((ti() == 0) || (ti() == 7)))
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
    constexpr char srk() const {return '8'-sr();}
    constexpr char sf() const {return 'a'+sc();}
    constexpr char trk() const {return '8'-tr();}
    constexpr char tf() const {return 'a'+tc();}
    constexpr bool pr() const {return spi() != tpi();}
    constexpr char sp() const {return GLYPHS[spi()];}
    constexpr char tp() const {return GLYPHS[tpi()];}
    constexpr char cp() const {return GLYPHS[cpi()];}
    constexpr bool ep() const {
        return sp() == 'P' && (sc() != tc()) && cp() == '.';
    }
    constexpr bool x() const {return (cp() != '.') || ep();}
    constexpr bool wkcr() const {return wcr() && kcr();}
    constexpr bool bkcr() const {return bcr() && kcr();}
    constexpr bool wqcr() const {return wcr() && qcr();}
    constexpr bool bqcr() const {return bcr() && qcr();}

    constexpr bool feasible() const {

        bool white = !c();

        // spi must name a glyph
        if (spi() > 6) return false;

        // can't move from an empty square
        if (sp() == '.') return false;

        // tpi must name a glyph
        if (tpi() > 6) return false;

        // tpi can't name '.'
        if (tp() == '.') return false;

        // cpi must name a glyph
        if (cpi() > 6) return false;

        // cpi may not name 'K'
        if (cp() == 'K') return false;

        // source != target
        if ((sc() == tc()) && (sr() == tr())) return false;

        // only pawns promote, and it must be properly positioned
        if ((sp() != tp()) && ((srk() != (white ? '7' : '2'))
            || (trk() != (white ? '8' : '1')) || (sp() != 'P'))) {
            return false;}

        // pawns can't promote to space, pawns, or kings
        if ((sp() != tp()) && ((tp() == '.') ||
            (tp() == 'P') || (tp() == 'K'))) return false;

        // pawns are never on rank 8 or rank 1 (row 0 or row 7)
        if ((sp() == 'P') && ((srk() == '8') ||
            (srk() == '1'))) return false;
        if ((tp() == 'P') && ((trk() == '8') ||
            (trk() == '1'))) return false;
        if ((cp() == 'P') && ((trk() == '8') ||
            (trk() == '1'))) return false;

        if (sp() == 'P') {
            // pawns can only move forward one rank at a time,
            // except for their first move
            uint8_t level_a = white ? 7-sr() : sr();
            uint8_t level_b = white ? 7-tr() : tr();
            if (level_b != level_a + 1) {
                if ((srk() != (white ? '2' : '7')) ||
                    (trk() != (white ? '4' : '5'))) return false;
                // can't capture on double push
                if (cp() != '.') return false;
            }
            // pawns stay on file when not capturing,
            // and move over one file when capturing.
            // i) can't move over more than one file
            if (sqr(sc()) + sqr(tc()) > 1 + 2*sc()*tc()) return false;
            // ii) can't capture forward
            if ((sc() == tc()) && (cp() != '.')) return false;
            // iii) can't move diagonal without capture
            if ((sc() != tc()) && (cp() == '.')) {
                // invalid unless possible en passant
                if (trk() != (white ? '6' : '3')) return false;
            }
        }

        if (sp() != tp()) {
            // can only promote on the endrank
            if (trk() != (white ? '8' : '1')) return false;
            // can only promote to N, B, R, Q
            if ((tp() == '.') || (tp() == 'P') ||
                    (tp() == 'K')) return false;
        }

        // that ought to cover it for pawns.

        if (sp() == 'N') {
            // i know how horsies move
            if ((sc()*sc() + tc()*tc() + sr()*sr() + tr()*tr())
                != 5 + 2*(sc()*tc() + sr()*tr())) return false;
        }


        if (sp() == 'B') {
            // bishops move on diagonals and antidiagonals
            if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
                    (sc() + tr() != tc() + sr())) // not on same diagonal
                return false;
        }


        if (sp() == 'R') {
            // rooks move on ranks and files (rows and columns)
            if ((sc() != tc()) && // not on same file
                    (sr() != tr())) // not on same rank
                return false;
            // conditions where kingside castle right may change
            if (kcr() && !((sf() == 'h') && (srk() == (white ? '1' : '8')))
                                && !((tf() == 'h') && (trk() == (white ? '8' : '1'))))
                return false;
            // if losing kingside rights, cannot move to files a-e
            if (kcr() && ((tf() == 'a') || (tf() == 'b') || (tf() == 'c') || (tf() == 'd') || (tf() == 'e')))
                return false;
            // conditions where queenside castle right may change
            if (qcr() && !((sf() == 'a') && (srk() == (white ? '1' : '8')))
                                && !((tf() == 'a') && (trk() == (white ? '8' : '1'))))
                return false;
            // if losing queenside rights, cannot move to files e-h
            if (qcr() && ((tf() == 'e') || (tf() == 'f') || (tf() == 'g') || (tf() == 'h')))
                return false;
        }

        if (sp() == 'Q') {
            // queens move on ranks, files, diagonals, and
            // antidiagonals.
            if ((sc() + sr() != tc() + tr()) && // not on same antidiagonal
                    (sc() + tr() != tc() + sr()) && // not on same diagonal
                    (sc() != tc()) && // not on same file
                    (sr() != tr())) // not on same rank
                return false;
            if ((sc() == tc()) && (sr() == tr())) return false;
        }

      if (sp() == 'K') {
          // if kingside castle, must be losing kingside rights
          if ((sf() == 'e') && (srk() == (white ? '1' : '8')) && (tf() == 'g') && (trk() == (white ? '1' : '8')) && !kcr()) return false;
          // if queenside castle, must be losing queenside rights
          if ((sf() == 'e') && (srk() == (white ? '1' : '8')) && (tf() == 'c') && (trk() == (white ? '1' : '8')) && !qcr()) return false;
          // the weirdest moves: (objectively, too. the last to have to debug and add code to handle) king takes rook losing castling rights.
          // Only diagonal/antidiagonal captures could possibly occur during play.
          if ((cp() == 'R') && kcr() && (trk() == (white ? '8' : '1')) && (tf() == 'h') && !((srk() == (white ? '7' : '2')) && (sf() == 'g'))) return false;
          if ((cp() == 'R') && qcr() && (trk() == (white ? '8' : '1')) && (tf() == 'a') && !((srk() == (white ? '7' : '2')) && (sf() == 'b'))) return false;
          // castling cannot capture, must be properly positioned
          if ((sqr(sc()) + sqr(tc()) > 1 + 2*sc()*tc())) {
              if (!((tf() == 'g') && kcr()) && !((tf() == 'c') && qcr())) return false;
              if (cp() != '.') return false;
              if (sf() != 'e') return false;
              if (srk() != (white ? '1' : '8')) return false;
              if (trk() != (white ? '1' : '8')) return false;
          }
          // kings move to neighboring squares
          if ((sqr(sc()) + sqr(tc()) + sqr(sr()) + sqr(tr()) >
                  2*(1 + sc()*tc() + sr()*tr())) &&
                  !((sc() == 4) && (sr() == (white ? 7 : 0)) && (tr()==sr()) &&
                      (((tc()==2) && qcr()) || ((tc()==6) && kcr())))) return false;
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

      bool kingmove = (sp() == 'K') && (sr() == (white ? 7 : 0)) && (sc()==4);
      bool a1rookcapture = (cp() == 'R') && (ti() == 56) && !white;
      bool a8rookcapture = (cp() == 'R') && (ti() == 0) && white;
      bool h1rookcapture = (cp() == 'R') && (ti() == 63) && !white;
      bool h8rookcapture = (cp() == 'R') && (ti() == 7) && white;

      bool a1rookmove = (sp() == 'R') && (si() == 56) && white && ((sc() == 0) || (sc() == 1) || (sc() == 2) || (sc() == 3));
      bool a8rookmove = (sp() == 'R') && (si() == 0) && !white && ((sc() == 0) || (sc() == 1) || (sc() == 2) || (sc() == 3));
      bool h1rookmove = (sp() == 'R') && (si() == 63) && white && ((sc() == 5) || (sc() == 6) || (sc() == 7));
      bool h8rookmove = (sp() == 'R') && (si() == 7) && !white && ((sc() == 5) || (sc() == 6) || (sc() == 7));
      if (kcr() && !(kingmove || h1rookmove || h8rookmove)) {
          if (h1rookcapture || h8rookcapture) {
              // exclude moves implying king is en prise
              if ((sp()=='R') && (sc()<6)) return false;
              if ((sp()=='Q') && (sr() == tr()) && (sc()<6)) return false;
              if ((sp()=='K') && ((sr() == tr()) || (sc() == tc()))) return false;
          } else {
              return false;
          }
      }
      if (qcr() && !(kingmove || a1rookmove || a8rookmove)) {
          if (a1rookcapture || a8rookcapture) {
              // exclude moves implying king is en prise
              if ((sp()=='R') && (sc() > 2)) return false;
              if ((sp()=='Q') && (sr() == tr()) && (sc() > 2)) return false;
              if ((sp()=='N') && (srk() == (white ? '7' : '2')) && (sc() == 2)) return false;
              if ((sp()=='K') && ((sr() == tr()) || (sc() == tc()))) return false;
          } else {
                  return false;
          }
      }

      return true;
    }
};


constexpr std::array<ThreeByteMove,44296>
compute_move_table() {
    std::array<ThreeByteMove,44296> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto tbm = ThreeByteMove(i);
        if (tbm.feasible()) result[j++] = tbm;
    }
    return result;
}

constexpr std::array<uint16_t,16777216> compute_lookup_table() {
    // to make this independent from compute_move_table()
    // we simply recompute it here. it's all compiler time
    // anyway.
    std::array<uint32_t,44296> movetable {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        if (ThreeByteMove(i).feasible()) movetable[j++] = i;
    }
    std::array<uint16_t,16777216> result {};
    int i = 0;
    for (uint32_t x : movetable) result[x] = i++;
    return result;
}

void moves_csv_to_stdout() {
    uint32_t cnt = 0;
    uint32_t pcnt = 0;
    uint32_t ncnt = 0;
    uint32_t rcnt = 0;
    uint32_t bcnt = 0;
    uint32_t qcnt = 0;
    uint32_t kcnt = 0;
    std::cout << "turn, sp, sf, srk, tp, tf, trk, cp, kcr, qcr\n";
    for ( uint32_t i = 0; i < 256*256*256; ++i) {
        ThreeByteMove tbm(i);
        if (tbm.feasible()) {
            std::cout << (tbm.c() ? '1' : '0') << ", " << tbm.sp() << ", " <<
                tbm.sf() << ", " << tbm.srk() << ", " << tbm.tp() << ", " <<
                tbm.tf() << ", " << tbm.trk() << ", " <<
                tbm.cp() << ", " << (tbm.kcr() ? '1' : '0') << ", " <<
                (tbm.qcr() ? '1' : '0') << "\n";
            ++ cnt;
            switch(tbm.sp()){
                case 'P': ++pcnt; break;
                case 'N': ++ncnt; break;
                case 'R': ++rcnt; break;
                case 'B': ++bcnt; break;
                case 'Q': ++qcnt; break;
                case 'K': ++kcnt; break;
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
    // 44295 total
    // here's the breakdown:
    // P: (8+(8+14*5)*5+(8*4+14*4*4))*2+28+16 == 1352
    // N: ((2*4+3*8+4*16)+(4*4+6*16)+8*16)*2*6-2*4*2-3*4*2-4*8*2+8-2 == 3934
    // R: (8*5+6*6)*32+48*(2*5+12*6)*2+(3*5+6*6)*2+(4*5+6*6)*2+(7+8)*2 == 10548
    // B: (7*28+9*20+11*12+13*4)*2*6-7*32+28 == 6524
    // Q: (21*28+23*20+25*12+27*4)*6*2-21*16*2+21*4-11*2 == 16862
    // K: (3*4+5*24+8*36)*6*2-(3*4+5*12)*2+8+4+(3*6+2*5)*2*3 == 5076
}

typedef uint64_t Bitboard;
typedef uint8_t Square;

struct Position {
    Bitboard white;
    Bitboard black;
    Bitboard pawn;
    Bitboard knight;
    Bitboard bishop;
    Bitboard rook;
    Bitboard queen;
    Bitboard king;
    uint8_t rights;
    char board[65];
    Position() :
    board("rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR") {
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
    bool c() const {return rights & 1;}
    bool wkcr() const {return rights & 2;}
    bool wqcr() const {return rights & 4;}
    bool bkcr() const {return rights & 8;}
    bool bqcr() const {return rights & 16;}
};

std::array<ThreeByteMove,44296> MOVETABLE {};
std::array<uint16_t,16777216> LOOKUP {};

void playcode(Position & P, uint16_t n) {
    auto const& tbm = MOVETABLE[n];
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

    uint64_t & pawn = P.pawn;
    uint64_t & knight = P.knight;
    uint64_t & bishop = P.bishop;
    uint64_t & rook = P.rook;
    uint64_t & queen = P.queen;
    uint64_t & king = P.king;
    auto & rights = P.rights;
    auto & board = P.board;
    bool white = !c;
    uint64_t & us = white ? P.white : P.black;
    uint64_t & them = white ? P.black : P.white;

    rights ^= (wkcr ? 0x02 : 0x00) | (wqcr ? 0x04 : 0x00) |
              (bkcr ? 0x08 : 0x00) | (bqcr ? 0x10 : 0x00) |
              (0x01);
    us ^= s;
    auto bsp = white ? sp : (sp + 32);
    board[si] ^= ('.' ^ bsp);

    switch (sp) {
        case 'P': pawn ^= s; break;
        case 'N': knight ^= s; break;
        case 'B': bishop ^= s; break;
        case 'R': rook ^= s; break;
        case 'Q': queen ^= s; break;
        case 'K': king ^= s; break;
    }

    // Remove capture piece (except en passant), if any
    if (cp != '.') them ^= t;

    switch (cp) {
        case 'P': pawn ^= t; break;
        case 'N': knight ^= t; break;
        case 'B': bishop ^= t; break;
        case 'R': rook ^= t; break;
        case 'Q': queen ^= t; break;
    }

    us ^= t;
    auto bcp = white ? (cp + 32) : cp;
    auto btp = white ? tp : (tp + 32);

    board[ti] ^= (bcp ^ btp);

    switch (tp) {
        case 'P': pawn ^= t; break;
        case 'N': knight ^= t; break;
        case 'B': bishop ^= t; break;
        case 'R': rook ^= t; break;
        case 'Q': queen ^= t; break;
    }

    if ((sp == 'P') && (tp == 'P') &&
            (cp == '.') && (sc != tc)) {
        // en passant capture
        pawn ^= tbm.u();
        them ^= tbm.u();
        board[ui] ^= ('.' ^ (white ? 'p' : 'P'));
    }

    if ((sp == 'K') && (tc == sc + 2)) {
        rook ^= white ? 0xA000000000000000 :
                        0x00000000000000A0;
        us ^= white ? 0xA000000000000000 :
                      0x00000000000000A0;
    }

    if ((sp == 'K') && (tc + 2 == sc)) {
        rook ^= white ? 0x0900000000000000 :
                        0x0000000000000009;
        us ^= white ? 0x0900000000000000 :
                      0x0000000000000009;
    }

}

#include "dispatcher.hpp"

template<uint16_t n>
struct PlayFun {
    typedef Position input_t;
    static void dispatch(input_t & t) {
        playcode(t, n);
    }
};

void play(Position & P, ThreeByteMove tbm) {
    uint16_t n = LOOKUP[tbm.X];
    dispatcher<PlayFun>(P, n);
}

int main(int argc, char * argv []) {
    compute_move_table();
    compute_lookup_table();
    // for (int i = 0; i < MOVETABLE.size(); ++ i) {
    //     if (LOOKUP[MOVETABLE[i].X] != i) {
    //         std::cerr << "Error at " << i << "\n";
    //     }
    // }
    std::cout << MOVETABLE[500].X << " " << MOVETABLE.size() << "\n";
    return 0;
}
