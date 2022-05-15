#include <cstdint>
#include <array>
#include <iostream>

template <typename T> T sqr(T x) {return x*x;}

constexpr std::array<char,7>
  SS {'.','P','N','B','R','Q','K'};

struct ThreeByteMove {
  uint32_t X;
  // 0x17        |   c | 0 if white to move
  // 0x14 - 0x16 | spi | source piece index
  // 0x11 - 0x13 |  sr | source row 87654321
  // 0x0E - 0x10 |  sc | source col abcdefgh
  // 0x0B - 0x0D |  tr | target row 87654321
  // 0x08 - 0x0A |  tc | target col abcdefgh
  // 0x05 - 0x07 | tpi | target pi .PNBRQK
  // 0x02 - 0x04 | cpi | capture pi .PNBRQK
  // 0x01        | qcr | change queen castling rights
  // 0x00        | kcr | change king castling rights
  ThreeByteMove (uint32_t X):X(X){};
  bool c() const {return X & 0x800000;}
  uint8_t spi() const {return (X >> 0x14) & 0x07;}
  uint8_t si() const  {return (X >> 0x0E) & 0x3F;}
  uint8_t ti() const  {return (X >> 0x08) & 0x3F;}
  uint8_t tpi() const {return (X >> 0x05) & 0x07;}
  uint8_t cpi() const {return (X >> 0x02) & 0x07;}
  bool qcr() const {return X & 0x02;}
  bool kcr() const {return X & 0x01;}

  // queries
  uint8_t sr() const {return si() >> 3;}
  uint8_t sc() const {return si() & 0x07;}
  uint8_t tr() const {return ti() >> 3;}
  uint8_t tc() const {return ti() & 0x07;}
  char srk() const {return '8'-sr();}
  char sf() const {return 'a'+sc();}
  char trk() const {return '8'-tr();}
  char tf() const {return 'a'+tc();}
  bool pr() const {return spi() != tpi();}
  char sp() const {return SS[spi()];}
  char tp() const {return SS[tpi()];}
  char cp() const {return SS[cpi()];}
  bool ep() const {return sp() == 'P' &&
    (sc() != tc()) && cp() == '.';}
  bool x() const {return (cp() != '.') || ep();}
  // uint16_t to_u16() const {return twobyte[X];}
  // void from_u16(uint16_t x){X=threebyte[x]);}

  bool feasible() const {

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

    // to change castling rights there are nine cases:
    //  1. king move
    //  2. a1 rook captured by black
    //  3. h1 rook captured by black
    //  4. a8 rook captured by white
    //  5. h8 rook captured by white
    //  6. a1 rook moved by white
    //  7. h1 rook moved by white
    //  8. a8 rook moved by black
    //  9. h8 rook moved black
    // White could capture an unmoved a8 rook with its unmoved a1 rook,
    // and similar scenarios, so the cases aren't mutually exclusive.
    bool kingmove = (sp() == 'K');
    bool a1rookcapture = (cp() == 'R') && (ti() == 56) && !white;
    bool a8rookcapture = (cp() == 'R') && (ti() == 0) && white;
    bool h1rookcapture = (cp() == 'R') && (ti() == 63) && !white;
    bool h8rookcapture = (cp() == 'R') && (ti() == 7) && white;
    bool a1rookmove = (sp() == 'R') && (si() == 56) && white;
    bool a8rookmove = (sp() == 'R') && (si() == 0) && !white;
    bool h1rookmove = (sp() == 'R') && (si() == 63) && white;
    bool h8rookmove = (sp() == 'R') && (si() == 7) && !white;
    if (kcr() && !(kingmove || h1rookcapture || h1rookmove || h8rookcapture || h8rookmove)) return false;
    if (qcr() && !(kingmove || a1rookcapture || a1rookmove || a8rookcapture || a8rookmove)) return false;

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
      //   i) can't move over more than one file
      if (sqr(sc()) + sqr(tc()) > 1 + 2*sc()*tc()) return false;
      //   ii) can't capture forward
      if ((sc() == tc()) && (cp() != '.')) return false;
      //  iii) can't move diagonal without capture
      if ((sc() != tc()) && (cp() == '.')) {
        // invalid unless possible en passant
        if (trk() != (white ? '6' : '3')) return false;
      }
    }

    // redundant from above
    //if ((sp() != tp()) && (sp() != 'P')) return false;

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
      if ((sqr(sc()) + sqr(tc()) + sqr(sr()) + sqr(tr())) !=
        5 + 2*(sc()*tc() + sr()*tr()))
        return false;
    }


    if (sp() == 'B') {
      // bishops move on diagonals and antidiagonals
      if ((sc() + sr() != tc() + tr()) &&  // not on same antidiagonal
          (sc() + tr() != tc() + sr()))    // not on same diagonal
        return false;
    }


    if (sp() == 'R') {
      // rooks move on ranks and files (rows and columns)
      if ((sc() != tc()) &&  // not on same file
          (sr() != tr()))    // not on same rank
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
      if ((sc() + sr() != tc() + tr()) &&  // not on same antidiagonal
          (sc() + tr() != tc() + sr()) &&  // not on same diagonal
          (sc() != tc()) &&  // not on same file
          (sr() != tr()))  // not on same rank
        return false;
      if ((sc() == tc()) && (sr() == tr())) return false;
    }

   if (sp() == 'K') {
     // king move cannot lose castling rights unless starting on home square
     if ((sf() != 'e') && (kcr() || qcr())) return false;
     if ((srk() != (white ? '1' : '8')) && (kcr() || qcr())) return false;
     // if kingside castle, must be losing kingside rights
     if ((sf() == 'e') && (srk() == (white ? '1' : '8')) && (tf() == 'g') && (trk() == (white ? '1' : '8')) && !kcr()) return false;
     // if queenside castle, must be losing queenside rights
     if ((sf() == 'e') && (srk() == (white ? '1' : '8')) && (tf() == 'g') && (trk() == (white ? '1' : '8')) && !qcr()) return false;
     // castling cannot capture, must be properly positioned
     if ((sqr(sc()) + sqr(tc()) > 1 + 2*sc()*tc())) {
       if (!((tf() == 'g') && kcr()) && !((tf() == 'c') && qcr())) return false;
       if (cp() != '.') return false;
       if (sf() != 'e') return false;
       if (srk() != (white ? '1' : '8')) return false;
       if (trk() != (white ? '1' : '8')) return false;
     }
     // kings move to neighboring squares
     if ((sqr(sc()) + sqr(tc()) + sqr(sr()) + sqr(tr()) > 2*(1 + sc()*tc() + sr()*tr())) && !(kcr() || qcr())) return false;
   }
   if (sp() == 'Q') {
     std::cout << sp() << sf() << srk() << ((cp() != '.') ? "x" : "") << tp() << tf() << trk() << " captured: " << cp() << " X = " << X << " " <<
     (c() ? "b" : "w") << " spi: " << int(spi()) << " tpi: " << int(tpi()) << " cpi: " << int(cpi()) << " si: " << int(si()) << " ti: " << int(ti()) << " cr: " << (kcr() ? "K" : "") << (qcr() ? "Q" : "") << "\n";
   }
   return true;
  }
  // advance to next legitimate move

};

int main(int arc, char * argv []) {
  uint32_t cnt = 0;
  uint32_t pcnt = 0;
  uint32_t ncnt = 0;
  uint32_t rcnt = 0;
  uint32_t bcnt = 0;
  uint32_t qcnt = 0;
  uint32_t kcnt = 0;
  for ( uint32_t i = 0; i < 256*256*256; ++i) {
    ThreeByteMove tbm(i);
    if (tbm.feasible()) {
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
  std::cout << cnt << "\n";
  std::cout << "P " << pcnt << "\n";
  std::cout << "N " << ncnt << "\n";
  std::cout << "R " << rcnt << "\n";
  std::cout << "B " << bcnt << "\n";
  std::cout << "Q " << qcnt << "\n";
  std::cout << "K " << kcnt << "\n";
  return 0;
}
