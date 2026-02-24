#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern "C" void _cacumm_reg(void);
extern "C" void _cagk_reg(void);
extern "C" void _cal2_reg(void);
extern "C" void _can2_reg(void);
extern "C" void _cat_reg(void);
extern "C" void _h_reg(void);
extern "C" void _kadist_reg(void);
extern "C" void _kaprox_reg(void);
extern "C" void _kca_reg(void);
extern "C" void _kdrca1_reg(void);
extern "C" void _kmb_reg(void);
extern "C" void _naxn_reg(void);

extern "C" void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"cacumm.mod\"");
    fprintf(stderr, " \"cagk.mod\"");
    fprintf(stderr, " \"cal2.mod\"");
    fprintf(stderr, " \"can2.mod\"");
    fprintf(stderr, " \"cat.mod\"");
    fprintf(stderr, " \"h.mod\"");
    fprintf(stderr, " \"kadist.mod\"");
    fprintf(stderr, " \"kaprox.mod\"");
    fprintf(stderr, " \"kca.mod\"");
    fprintf(stderr, " \"kdrca1.mod\"");
    fprintf(stderr, " \"kmb.mod\"");
    fprintf(stderr, " \"naxn.mod\"");
    fprintf(stderr, "\n");
  }
  _cacumm_reg();
  _cagk_reg();
  _cal2_reg();
  _can2_reg();
  _cat_reg();
  _h_reg();
  _kadist_reg();
  _kaprox_reg();
  _kca_reg();
  _kdrca1_reg();
  _kmb_reg();
  _naxn_reg();
}
