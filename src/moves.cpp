// moves.cpp
// Shaun Harker
// BSD ZERO CLAUSE LICENSE

#include <cstdint>
#include <array>
#include <iostream>
#include <string>

template <typename T> constexpr T sqr(T x) {return x*x;}

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

struct FourByteMove {
  // 0x17        | c      | ply % 2
  // 0x00 - 0x00 | xorepc | xor of prev epc col and new epc col
  // 0x01        | bqcr   | change black queen castling rights
  // 0x00        | bkcr   | change black king castling rights
  //             | wqcr   | change white queen castling rights
  //             | wkcr   | change white king castling rights
  // 0x14 - 0x16 | spi    | source piece index
  // 0x02 - 0x04 | cpi    | capture piece idx into .PNBRQK
  // 0x11 - 0x13 | sr     | source row 87654321
  // 0x0E - 0x10 | sc     | source col abcdefgh
  // 0x0B - 0x0D | tr     | target row 87654321
  // 0x08 - 0x0A | tc     | target col abcdefgh
  // 0x05 - 0x07 | tpi    | target piece idx into .PNBRQK

  uint32_t X;
};

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

        // that ought to cover it for pawns.

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



typedef uint64_t Bitboard;
typedef uint8_t Square;

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
    Position() //:
    //board("rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR")
    {
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
};

std::array<ThreeByteMove,44304>
compute_move_table() {
    std::array<ThreeByteMove,44304> result {};
    uint16_t j = 0;
    for (uint32_t i = 0; i < 256*256*256; ++i) {
        auto tbm = ThreeByteMove(i);
        if (tbm.feasible()) result[j++] = tbm;
    }
    return result;
}

std::array<uint16_t,16777216>
compute_lookup_table() {
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
    moves_csv_to_stdout();
    // std::cout << "Move application rate test.\n";
    // Position P;
    // for (int x = 0; x < 10000; ++ x) {
    //   for (uint16_t code = 0; code < 44304; ++ code) {
    //       P.play(MOVETABLE[code]);
    //   }
    // }

    return 0;
}
