//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void Edamp_cpu(real dt) {

//<USER_DEFINED>
  INPUT(Energy);
  INPUT(Density);
  OUTPUT(Energy);
//<\USER_DEFINED>

//<EXTERNAL>
  real* e    = Energy->field_cpu;
  real* rho  = Density->field_cpu;
  real  beta = BETA;
  real  gam  = GAMMA;
  real  r0   = R0;
  real  asp  = ASPECTRATIO;
  real  flar = FLARINGINDEX;
  real  bigg = G;
  real  mstar= MSTAR;
  int   pitch     = Pitch_cpu;
  int   stride    = Stride_cpu;
  int   size_x    = Nx;
  int   size_y    = Ny + 2*NGHY;
  int   size_z    = Nz + 2*NGHZ;
  int   fluidtype = Fluidtype;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;

  real bigrad;
  real Hg;
  real R3;
  real omk;
  real cs;
  real cs2;
  real tauc;
  real rho_cell;
  real e_old;
  real e_eq;
  real dt_over_tau;
  real denom;

#ifndef __GPU
  static int first = 1;
#endif
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

  for (k = 0; k < size_z; k++) {
    for (j = 0; j < size_y; j++) {
      for (i = 0; i < size_x; i++) {
//<#>
        ll = l;  /* global 1D index, defined in define.h */

#ifndef __GPU
        if (first) {
          printf("Edamp_cpu first call, dt = %g, rank = %d, Fluidtype = %d\n",
                 dt, CPU_Rank, fluidtype);
          fflush(stdout);
          first = 0;
        }
#endif

        /* cylindrical radius R at cell centre */
#ifdef SPHERICAL
        // Y=r, Z=theta
        real r_sph = ymed(j);
        real theta = zmed(k);
        bigrad = r_sph * sin(theta); // 这就是 R_cyl
#endif
#ifdef CYLINDRICAL
        bigrad = ymed(j);
#endif

        if (bigrad <= 0.0) {
          /* R<=0: skip cell */
        } else {

          /* H/R profile via asp, flar (local isothermal background) */
          Hg  = asp * pow(bigrad / r0, flar) * bigrad;
          R3  = bigrad * bigrad * bigrad;
          omk = sqrt(bigg * mstar / R3);
          cs  = omk * Hg;
          cs2 = cs * cs;

#ifdef BETACOOLING
          /* == Only cool the gas fluid == */
          if (fluidtype == GAS && beta > 0.0 && cs2 > 0.0 && omk > 0.0) {

            /* tau_c = beta / Omega_K */
            tauc = beta / omk;

            rho_cell = rho[ll];
            if (rho_cell > 0.0) {

              e_old = e[ll];
              if (e_old <= 0.0) e_old = 1e-30;

              /* local isothermal target energy:
                 e_eq = rho * cs^2 / (gamma - 1) */
              e_eq = rho_cell * cs2 / (gam - 1.0);
              if (e_eq <= 0.0) e_eq = 1e-30;

              dt_over_tau = dt / tauc;
              if (dt_over_tau > 100.0) dt_over_tau = 100.0;

              denom = 1.0 + dt_over_tau;
              if (denom > 0.0 && isfinite(denom)) {

                /* implicit beta–cooling update:
                   e^{n+1} = (e^n + (dt/tau) e_eq) / (1 + dt/tau) */
                e[ll] = (e_old + dt_over_tau * e_eq) / denom;

#ifndef __GPU
                if (!isfinite(e[ll]) || !isfinite(rho_cell) ||
                    !isfinite(omk)   || !isfinite(cs2)) {
                  printf("NaN/inf in Edamp at i=%d j=%d k=%d, "
                         "e=%g rho=%g R=%g tauc=%g omk=%g\n",
                         i, j, k,
                         (double)e[ll], (double)rho_cell,
                         (double)bigrad, (double)tauc, (double)omk);
                  fflush(stdout);
                  exit(1);
                }
#endif /* !__GPU */
              }
            }
          }
#endif /* BETACOOLING */
        }
//<\#>
      }
    }
  }

//<\MAIN_LOOP>
}
