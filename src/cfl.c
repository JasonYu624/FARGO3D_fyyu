//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
#include <math.h>

// ============================================================
// CFL / BAD-STATE abort monitor (GPU-safe, MPI-friendly)
// ============================================================
#define CFL_ABORT_THRESHOLD 5e-4

#define CFL_REASON_SMALL_DT        1
#define CFL_REASON_DT_NAN          2
#define CFL_REASON_DT_NONPOS       3

#define CFL_REASON_BAD_DX        101
#define CFL_REASON_BAD_DY        102
#define CFL_REASON_BAD_DZ        103

#define CFL_REASON_BAD_RHO       201
#define CFL_REASON_BAD_E         202
#define CFL_REASON_BAD_VX        203
#define CFL_REASON_BAD_VY        204
#define CFL_REASON_BAD_VZ        205

#define CFL_REASON_BAD_CS2       301
#define CFL_REASON_BAD_CS        302
#define CFL_REASON_BAD_VISC      303

#define CFL_REASON_BAD_CFL1      401
#define CFL_REASON_BAD_CFL2      402
#define CFL_REASON_BAD_CFL3      403
#define CFL_REASON_BAD_CFL4      404
#define CFL_REASON_BAD_CFL5      405
#define CFL_REASON_BAD_CFL6      406
#define CFL_REASON_BAD_CFL7      407
#define CFL_REASON_BAD_CFL8      408
#define CFL_REASON_BAD_CFL9      409
#define CFL_REASON_BAD_CFL10     410
#define CFL_REASON_BAD_DENOM     411

#ifdef __GPU
#include <cuda_runtime.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

static const char *CFLReasonName(int reason) {
  switch (reason) {
    case CFL_REASON_SMALL_DT:    return "small dt";
    case CFL_REASON_DT_NAN:      return "dt is NaN/Inf";
    case CFL_REASON_DT_NONPOS:   return "dt <= 0";

    case CFL_REASON_BAD_DX:      return "dx invalid";
    case CFL_REASON_BAD_DY:      return "dy invalid";
    case CFL_REASON_BAD_DZ:      return "dz invalid";

    case CFL_REASON_BAD_RHO:     return "rho invalid/non-positive";
    case CFL_REASON_BAD_E:       return "energy invalid";
    case CFL_REASON_BAD_VX:      return "vx invalid";
    case CFL_REASON_BAD_VY:      return "vy invalid";
    case CFL_REASON_BAD_VZ:      return "vz invalid";

    case CFL_REASON_BAD_CS2:     return "soundspeed2 invalid/negative";
    case CFL_REASON_BAD_CS:      return "soundspeed invalid";
    case CFL_REASON_BAD_VISC:    return "viscosity invalid/negative";

    case CFL_REASON_BAD_CFL1:    return "cfl1 invalid";
    case CFL_REASON_BAD_CFL2:    return "cfl2 invalid";
    case CFL_REASON_BAD_CFL3:    return "cfl3 invalid";
    case CFL_REASON_BAD_CFL4:    return "cfl4 invalid";
    case CFL_REASON_BAD_CFL5:    return "cfl5 invalid";
    case CFL_REASON_BAD_CFL6:    return "cfl6 invalid";
    case CFL_REASON_BAD_CFL7:    return "cfl7 invalid";
    case CFL_REASON_BAD_CFL8:    return "cfl8 invalid";
    case CFL_REASON_BAD_CFL9:    return "cfl9 invalid";
    case CFL_REASON_BAD_CFL10:   return "cfl10 invalid";
    case CFL_REASON_BAD_DENOM:   return "CFL denominator invalid/non-positive";
    default:                     return "unknown";
  }
}

#ifdef __GPU
__device__ int  cfl_abort_flag;
__device__ int  cfl_abort_reason;
__device__ int  cfl_abort_ijk[3];
__device__ real cfl_abort_dt;
__device__ int  cfl_abort_termid;
__device__ real cfl_abort_terms2[10];
__device__ real cfl_abort_phys[4];
__device__ real cfl_abort_cfl5cand2[3];
__device__ int  cfl_abort_cfl5dir;
__device__ real cfl_abort_geom[8];
__device__ real cfl_abort_thermo[10];
__device__ real cfl_abort_face[10];

static inline void CFLAbortResetDevice(void) {
  int  zeroi = 0;
  int  zero3[3] = {0,0,0};
  real zeror = (real)0.0;
  real zero4[4] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0};
  real zero3r[3] = {(real)0.0,(real)0.0,(real)0.0};
  real zero8[8] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,
                   (real)0.0,(real)0.0,(real)0.0,(real)0.0};
  real zero10[10] = {(real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0,
                     (real)0.0,(real)0.0,(real)0.0,(real)0.0,(real)0.0};

  cudaMemcpyToSymbol(cfl_abort_flag,     &zeroi,  sizeof(int),      0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_reason,   &zeroi,  sizeof(int),      0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_cfl5dir,  &zeroi,  sizeof(int),      0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_termid,   &zeroi,  sizeof(int),      0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_ijk,      zero3,   3*sizeof(int),    0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_dt,       &zeror,  sizeof(real),     0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_terms2,   zero10,  10*sizeof(real),  0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_phys,     zero4,   4*sizeof(real),   0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_cfl5cand2,zero3r,  3*sizeof(real),   0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_geom,     zero8,   8*sizeof(real),   0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_thermo,   zero10,  10*sizeof(real),  0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(cfl_abort_face,     zero10,  10*sizeof(real),  0, cudaMemcpyHostToDevice);
}

static void CFLAbortCheckAndExit(void) {
  int  flag   = 0;
  int  reason = 0;
  int  ijk[3] = {0,0,0};
  int  termid = 0;
  real dt     = (real)1e30;
  real terms2[10];
  real phys[4];
  real cfl5cand2[3];
  int  cfl5dir = 0;
  real geom[8];
  real thermo[10];
  real face[10];

  for (int q=0;q<10;q++) terms2[q] = (real)0.0;
  for (int q=0;q<4;q++)  phys[q]   = (real)0.0;
  for (int q=0;q<3;q++)  cfl5cand2[q] = (real)0.0;
  for (int q=0;q<8;q++)  geom[q]   = (real)0.0;
  for (int q=0;q<10;q++) thermo[q] = (real)0.0;
  for (int q=0;q<10;q++) face[q]   = (real)0.0;

  cudaMemcpyFromSymbol(&flag,   cfl_abort_flag,      sizeof(int),     0, cudaMemcpyDeviceToHost);
  if (!flag) return;

  cudaMemcpyFromSymbol(&reason, cfl_abort_reason,    sizeof(int),     0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(ijk,     cfl_abort_ijk,       3*sizeof(int),   0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&dt,     cfl_abort_dt,        sizeof(real),    0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&termid, cfl_abort_termid,    sizeof(int),     0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(terms2,  cfl_abort_terms2,    10*sizeof(real), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(phys,    cfl_abort_phys,      4*sizeof(real),  0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(cfl5cand2,cfl_abort_cfl5cand2,3*sizeof(real),  0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&cfl5dir,cfl_abort_cfl5dir,   sizeof(int),     0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(geom,    cfl_abort_geom,      8*sizeof(real),  0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(thermo,  cfl_abort_thermo,    10*sizeof(real), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(face,    cfl_abort_face,      10*sizeof(real), 0, cudaMemcpyDeviceToHost);

  double mydt;
  if (!isfinite((double)dt) || dt <= (real)0.0) mydt = -1.0e99;
  else mydt = (double)dt;

#ifdef MPI
  int myrank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  struct { double val; int rank; } in, out;
  in.val  = mydt;
  in.rank = myrank;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  if (out.val > (double)CFL_ABORT_THRESHOLD) return;

  if (myrank == out.rank && myrank != 0) {
    MPI_Send(&reason, 1, MPI_INT, 0, 9100, MPI_COMM_WORLD);
    MPI_Send(ijk,     3, MPI_INT, 0, 9101, MPI_COMM_WORLD);
    MPI_Send(&termid, 1, MPI_INT, 0, 9102, MPI_COMM_WORLD);
    MPI_Send(&cfl5dir,1, MPI_INT, 0, 9107, MPI_COMM_WORLD);

    if (sizeof(real) == sizeof(double)) {
      MPI_Send(&dt,      1,  MPI_DOUBLE, 0, 9103, MPI_COMM_WORLD);
      MPI_Send(terms2,   10, MPI_DOUBLE, 0, 9104, MPI_COMM_WORLD);
      MPI_Send(phys,     4,  MPI_DOUBLE, 0, 9105, MPI_COMM_WORLD);
      MPI_Send(cfl5cand2,3,  MPI_DOUBLE, 0, 9106, MPI_COMM_WORLD);
      MPI_Send(geom,     8,  MPI_DOUBLE, 0, 9108, MPI_COMM_WORLD);
      MPI_Send(thermo,   10, MPI_DOUBLE, 0, 9109, MPI_COMM_WORLD);
      MPI_Send(face,     10, MPI_DOUBLE, 0, 9110, MPI_COMM_WORLD);
    } else {
      MPI_Send(&dt,      1,  MPI_FLOAT, 0, 9103, MPI_COMM_WORLD);
      MPI_Send(terms2,   10, MPI_FLOAT, 0, 9104, MPI_COMM_WORLD);
      MPI_Send(phys,     4,  MPI_FLOAT, 0, 9105, MPI_COMM_WORLD);
      MPI_Send(cfl5cand2,3,  MPI_FLOAT, 0, 9106, MPI_COMM_WORLD);
      MPI_Send(geom,     8,  MPI_FLOAT, 0, 9108, MPI_COMM_WORLD);
      MPI_Send(thermo,   10, MPI_FLOAT, 0, 9109, MPI_COMM_WORLD);
      MPI_Send(face,     10, MPI_FLOAT, 0, 9110, MPI_COMM_WORLD);
    }
  }

  if (myrank == 0) {
    if (out.rank != 0) {
      MPI_Recv(&reason, 1, MPI_INT, out.rank, 9100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(ijk,     3, MPI_INT, out.rank, 9101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&termid, 1, MPI_INT, out.rank, 9102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&cfl5dir,1, MPI_INT, out.rank, 9107, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (sizeof(real) == sizeof(double)) {
        MPI_Recv(&dt,      1,  MPI_DOUBLE, out.rank, 9103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(terms2,   10, MPI_DOUBLE, out.rank, 9104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(phys,     4,  MPI_DOUBLE, out.rank, 9105, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cfl5cand2,3,  MPI_DOUBLE, out.rank, 9106, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(geom,     8,  MPI_DOUBLE, out.rank, 9108, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(thermo,   10, MPI_DOUBLE, out.rank, 9109, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(face,     10, MPI_DOUBLE, out.rank, 9110, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&dt,      1,  MPI_FLOAT, out.rank, 9103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(terms2,   10, MPI_FLOAT, out.rank, 9104, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(phys,     4,  MPI_FLOAT, out.rank, 9105, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(cfl5cand2,3,  MPI_FLOAT, out.rank, 9106, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(geom,     8,  MPI_FLOAT, out.rank, 9108, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(thermo,   10, MPI_FLOAT, out.rank, 9109, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(face,     10, MPI_FLOAT, out.rank, 9110, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    const char *dname =
      (cfl5dir==1) ? "X(phi)" :
      (cfl5dir==2) ? "Y(r)"   :
      (cfl5dir==3) ? "Z(theta)" : "unknown";

    printf("\n[CFL-DEBUG-ABORT]\n");
    printf("  reason=%d (%s)\n", reason, CFLReasonName(reason));
    printf("  dt=%+.6e at i=%d j=%d k=%d (rank=%d)\n", (double)dt, ijk[0], ijk[1], ijk[2], out.rank);
    printf("  dominant term: cfl%d\n", termid);
    printf("  terms^2: c1=%e c2=%e c3=%e c4=%e c5=%e c6=%e c7=%e c8=%e c9=%e c10=%e\n",
           (double)terms2[0], (double)terms2[1], (double)terms2[2], (double)terms2[3],
           (double)terms2[4], (double)terms2[5], (double)terms2[6], (double)terms2[7],
           (double)terms2[8], (double)terms2[9]);
    printf("  cfl5 cand^2: X=%e Y=%e Z=%e   -> dominant %s\n",
           (double)cfl5cand2[0], (double)cfl5cand2[1], (double)cfl5cand2[2], dname);

    printf("  phys: cs=%g  vphi=%g  vr=%g  vtheta=%g\n",
           (double)phys[0], (double)phys[1], (double)phys[2], (double)phys[3]);

    printf("  geom: r=%g theta=%g R=%g z=%g H=%g z/H=%g dx=%g dy=%g dz=%g\n",
           (double)geom[0], (double)geom[1], (double)geom[2], (double)geom[3],
           (double)geom[4], (double)((fabs(geom[4])>0.0)?geom[3]/geom[4]:(real)0.0),
           (double)geom[5], (double)geom[6], (double)geom[7]);

    printf("  thermo: rho=%g e=%g p=%g cs=%g cs_bg=%g cs_eq_ad=%g e_eq=%g nu=%g tau_c=%g dt/tau=%g\n",
           (double)thermo[0], (double)thermo[1], (double)thermo[2], (double)thermo[3],
           (double)thermo[4], (double)thermo[5], (double)thermo[6],
           (double)thermo[7], (double)thermo[8], (double)thermo[9]);

    printf("  face(%s): vL=%g vR=%g dv=%g dl=%g | rhoL=%g rhoR=%g | eL=%g eR=%g | csL=%g csR=%g\n",
           dname,
           (double)face[0], (double)face[1], (double)face[8], (double)face[9],
           (double)face[2], (double)face[3], (double)face[4], (double)face[5],
           (double)face[6], (double)face[7]);
    fflush(stdout);
  }

  MPI_Abort(MPI_COMM_WORLD, 1);
#else
  if (CPU_Master) {
    const char *dname =
      (cfl5dir==1) ? "X(phi)" :
      (cfl5dir==2) ? "Y(r)"   :
      (cfl5dir==3) ? "Z(theta)" : "unknown";

    printf("\n[CFL-DEBUG-ABORT]\n");
    printf("  reason=%d (%s)\n", reason, CFLReasonName(reason));
    printf("  dt=%+.6e at i=%d j=%d k=%d\n", (double)dt, ijk[0], ijk[1], ijk[2]);
    printf("  dominant term: cfl%d\n", termid);
    printf("  cfl5 cand^2: X=%e Y=%e Z=%e   -> dominant %s\n",
           (double)cfl5cand2[0], (double)cfl5cand2[1], (double)cfl5cand2[2], dname);
    printf("  phys: cs=%g  vphi=%g  vr=%g  vtheta=%g\n",
           (double)phys[0], (double)phys[1], (double)phys[2], (double)phys[3]);
    printf("  geom: r=%g theta=%g R=%g z=%g H=%g z/H=%g dx=%g dy=%g dz=%g\n",
           (double)geom[0], (double)geom[1], (double)geom[2], (double)geom[3],
           (double)geom[4], (double)((fabs(geom[4])>0.0)?geom[3]/geom[4]:(real)0.0),
           (double)geom[5], (double)geom[6], (double)geom[7]);
    printf("  thermo: rho=%g e=%g p=%g cs=%g cs_bg=%g cs_eq_ad=%g e_eq=%g nu=%g tau_c=%g dt/tau=%g\n",
           (double)thermo[0], (double)thermo[1], (double)thermo[2], (double)thermo[3],
           (double)thermo[4], (double)thermo[5], (double)thermo[6],
           (double)thermo[7], (double)thermo[8], (double)thermo[9]);
    printf("  face(%s): vL=%g vR=%g dv=%g dl=%g | rhoL=%g rhoR=%g | eL=%g eR=%g | csL=%g csR=%g\n",
           dname,
           (double)face[0], (double)face[1], (double)face[8], (double)face[9],
           (double)face[2], (double)face[3], (double)face[4], (double)face[5],
           (double)face[6], (double)face[7]);
    fflush(stdout);
  }
  exit(1);
#endif
}
#endif

//<\INCLUDES>

void cfl_cpu() {

//<USER_DEFINED>
  INPUT(Energy);
  INPUT(Density);
  OUTPUT(DensStar);
#ifdef X
  INPUT(Vx);
  INPUT2D(VxMed);
#endif
#ifdef Y
  INPUT(Vy);
#endif
#ifdef Z
  INPUT(Vz);
#endif
#ifdef MHD
  INPUT(Bx);
  INPUT(By);
  INPUT(Bz);
#endif
#ifdef HALLEFFECT
  INPUT(EtaHall);
#endif
#ifdef AMBIPOLARDIFFUSION
  INPUT(EtaAD);
#endif
#ifdef OHMICDIFFUSION
  INPUT(EtaOhm);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* cs  = Energy->field_cpu;
  real* rho = Density->field_cpu;
  real* dtime = DensStar->field_cpu;
#ifdef X
  real* vx = Vx->field_cpu;
  real* vxmed = VxMed->field_cpu;
#endif
#ifdef Y
  real* vy = Vy->field_cpu;
#endif
#ifdef Z
  real* vz = Vz->field_cpu;
#endif
#ifdef MHD
  real* bx = Bx->field_cpu;
  real* by = By->field_cpu;
  real* bz = Bz->field_cpu;
#endif
#ifdef MHD
#ifdef OHMICDIFFUSION
  real* etao = EtaOhm->field_cpu;
#endif
#ifdef HALLEFFECT
  real* etahall = EtaHall->field_cpu;
#endif
#ifdef AMBIPOLARDIFFUSION
  real* etaad   = EtaAD->field_cpu;
#endif
#endif
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+NGHX;
  int size_y = Ny+NGHY;
  int size_z = Nz+NGHZ;
  int pitch2d = Pitch2D;
  int fluidtype = Fluidtype;
  real beta = BETA;
  real r0   = R0;
  real asp  = ASPECTRATIO;
  real flar = FLARINGINDEX;
  real bigg = G;
  real mstar= MSTAR;
//<\EXTERNAL>

//<INTERNAL>
  real __attribute__((unused))dtmin = 1e30;
  int i, j, k;
  int ll, llxp, llyp, llzp;

  real cfl1_a=0.0, cfl1_b=0.0, cfl1_c=0.0, cfl1=0.0;
  real cfl2=0.0, cfl3=0.0, cfl4=0.0;
  real cfl5_a=0.0, cfl5_b=0.0, cfl5_c=0.0, cfl5=0.0;
  real cfl6=0.0;
  real cfl7_a=0.0, cfl7_b=0.0, cfl7_c=0.0, cfl7=0.0;
  real cfl8=0.0, cfl9=0.0, cfl10=0.0;

  real vxx=0.0, vxxp=0.0;
  real soundspeed=0.0, soundspeed2=0.0;
  real viscosity=0.0;

  real rloc=0.0, thetaloc=0.0, Rloc=0.0, zloc=0.0, Hloc=0.0;
  real cs_bg=0.0, cs_eq_ad=0.0, e_eq=0.0, tau_c=0.0, pgas=0.0, dt_over_tau=0.0;
  real dxloc=0.0, dyloc=0.0, dzloc=0.0;
  real denom2 = 0.0;
  int bad_code = 0;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+2*NGHX+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real GAMMA(1);
// real CFL(1);
// real ALPHA(1);
// real NU(1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef __GPU
  CFLAbortResetDevice();
#endif

#ifdef Z
  for (k=NGHZ; k<size_z; k++) {
#endif
#ifdef Y
    for (j=NGHY; j<size_y; j++) {
#endif
#ifdef X
      for (i=NGHX; i<size_x; i++) {
#endif
//<#>
        ll   = l;
        llxp = lxp;
        llyp = lyp;
        llzp = lzp;

#ifdef X
#ifdef STANDARD
        vxx  = vx[ll];
        vxxp = vx[llxp];
#else
        vxx  = vx[ll]  - vxmed[l2D];
        vxxp = vx[llxp]- vxmed[l2D];
#endif
#endif

        soundspeed2 = (real)0.0;
        soundspeed  = (real)0.0;
        viscosity   = (real)0.0;

        if (fluidtype == GAS) {
#ifdef ISOTHERMAL
          soundspeed2 = cs[ll]*cs[ll];
#endif

#ifdef ADIABATIC
          soundspeed2 = GAMMA*(GAMMA-1.0)*cs[ll]/rho[ll];
#endif

#ifdef POLYTROPIC
          soundspeed2 = GAMMA*cs[ll]*pow(rho[ll], GAMMA-1.0);
#endif

#ifdef ALPHAVISCOSITY
#ifdef ISOTHERMAL
          viscosity = ALPHA*cs[ll]*cs[ll]*sqrt(ymed(j)*ymed(j)*ymed(j)/(G*MSTAR));
#else
          viscosity = ALPHA*GAMMA*(GAMMA-1.0)*cs[ll]/rho[ll]*sqrt(ymed(j)*ymed(j)*ymed(j)/(G*MSTAR));
#endif
#else
          viscosity = NU;
#endif

#ifdef MHD
          soundspeed2 += ((bx[ll]*bx[ll]+by[ll]*by[ll]+bz[ll]*bz[ll])/(MU0*rho[ll]));
#endif
        }

        if (!isfinite(soundspeed2) || soundspeed2 < (real)0.0) {
          soundspeed = (real)(-1.0);
        } else {
          soundspeed = sqrt(soundspeed2);
        }
        dxloc = (real)0.0;
        dyloc = (real)0.0;
        dzloc = (real)0.0;
#ifdef X
        dxloc = zone_size_x(i,j,k);
#endif
#ifdef Y
        dyloc = zone_size_y(j,k);
#endif
#ifdef Z
        dzloc = zone_size_z(j,k);
#endif

        rloc = (real)0.0;
        thetaloc = (real)0.0;
        Rloc = (real)0.0;
        zloc = (real)0.0;
        Hloc = (real)0.0;
        cs_bg = (real)0.0;
        cs_eq_ad = (real)0.0;
        e_eq = (real)0.0;
        tau_c = (real)0.0;
        pgas = (real)0.0;
        dt_over_tau = (real)0.0;

        if (fluidtype == GAS) {
#ifdef SPHERICAL
          rloc     = ymed(j);
          thetaloc = zmed(k);
          Rloc     = rloc*sin(thetaloc);
          zloc     = rloc*cos(thetaloc);
#endif
#ifdef CYLINDRICAL
          Rloc     = ymed(j);
          rloc     = Rloc;
          thetaloc = (real)(M_PI*0.5);
          zloc     = (real)0.0;
#endif
          if (Rloc > (real)0.0) {
            Hloc = asp * pow(Rloc/r0, flar) * Rloc;
            cs_bg = sqrt(bigg*mstar/(Rloc*Rloc*Rloc)) * Hloc;
            cs_eq_ad = sqrt((real)GAMMA) * cs_bg;
#ifdef ADIABATIC
            e_eq = rho[ll]*cs_bg*cs_bg/(GAMMA-(real)1.0);
            pgas = (GAMMA-(real)1.0)*cs[ll];
#endif
#ifdef BETACOOLING
            tau_c = beta / sqrt(bigg*mstar/(Rloc*Rloc*Rloc));
            // dt_over_tau is only meaningful after dtime[ll] is computed.
#endif
          }
        }

#ifdef X
        cfl1_a = soundspeed/dxloc;
#endif
#ifdef Y
        cfl1_b = soundspeed/dyloc;
#endif
#ifdef Z
        cfl1_c = soundspeed/dzloc;
#endif
        cfl1 = max3(cfl1_a, cfl1_b, cfl1_c);

#ifdef X
        cfl2 = (max2(fabs(vxx),fabs(vxxp)))/dxloc;
#endif
#ifdef Y
        cfl3 = (max2(fabs(vy[ll]),fabs(vy[llyp])))/dyloc;
#endif
#ifdef Z
        cfl4 = (max2(fabs(vz[ll]),fabs(vz[llzp])))/dzloc;
#endif

#ifndef NOSUBSTEP2
#ifdef X
        cfl5_a = fabs(vx[llxp]-vx[ll])/dxloc;
#endif
#ifdef Y
        cfl5_b = fabs(vy[llyp]-vy[ll])/dyloc;
#endif
#ifdef Z
        cfl5_c = fabs(vz[llzp]-vz[ll])/dzloc;
#endif
        cfl5 = max3(cfl5_a, cfl5_b, cfl5_c)*4.0*CVNR;
#endif

#ifdef STRONG_SHOCK
        cfl6 = cfl5/CVNR*CVNL;
#endif

#ifdef X
        cfl7_a = 1.0/dxloc;
#endif
#ifdef Y
        cfl7_b = 1.0/dyloc;
#endif
#ifdef Z
        cfl7_c = 1.0/dzloc;
#endif
        cfl7 = 4.0*viscosity*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);

#ifdef MHD
#ifdef OHMICDIFFUSION
        cfl8 = 4.0*etao[ll]*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#ifdef HALLEFFECT
        cfl9 = 6.0*fabs(etahall[ll])*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#ifdef AMBIPOLARDIFFUSION
        cfl10 = 4.0*etaad[ll]*pow(max3(cfl7_a,cfl7_b,cfl7_c),2);
#endif
#endif

        denom2 = cfl1*cfl1 + cfl2*cfl2 +
                 cfl3*cfl3 + cfl4*cfl4 +
                 cfl5*cfl5 + cfl6*cfl6 +
                 cfl7*cfl7 + cfl8*cfl8 +
                 cfl9*cfl9 + cfl10*cfl10;

        bad_code = 0;

#ifdef X
        if ((!isfinite(dxloc)) || (dxloc <= (real)0.0)) bad_code = CFL_REASON_BAD_DX;
#endif
#ifdef Y
        if ((bad_code == 0) && ((!isfinite(dyloc)) || (dyloc <= (real)0.0))) bad_code = CFL_REASON_BAD_DY;
#endif
#ifdef Z
        if ((bad_code == 0) && ((!isfinite(dzloc)) || (dzloc <= (real)0.0))) bad_code = CFL_REASON_BAD_DZ;
#endif

        if ((bad_code == 0) && ((!isfinite(rho[ll])) || (rho[ll] <= (real)0.0))) bad_code = CFL_REASON_BAD_RHO;
        if ((bad_code == 0) && (!isfinite(cs[ll]))) bad_code = CFL_REASON_BAD_E;

#ifdef X
        if ((bad_code == 0) && ((!isfinite(vxx)) || (!isfinite(vxxp)))) bad_code = CFL_REASON_BAD_VX;
#endif
#ifdef Y
        if ((bad_code == 0) && ((!isfinite(vy[ll])) || (!isfinite(vy[llyp])))) bad_code = CFL_REASON_BAD_VY;
#endif
#ifdef Z
        if ((bad_code == 0) && ((!isfinite(vz[ll])) || (!isfinite(vz[llzp])))) bad_code = CFL_REASON_BAD_VZ;
#endif

        if ((bad_code == 0) && ((!isfinite(soundspeed2)) || (soundspeed2 < (real)0.0))) bad_code = CFL_REASON_BAD_CS2;
        if ((bad_code == 0) && (!isfinite(soundspeed))) bad_code = CFL_REASON_BAD_CS;
        if ((bad_code == 0) && ((!isfinite(viscosity)) || (viscosity < (real)0.0))) bad_code = CFL_REASON_BAD_VISC;

        if ((bad_code == 0) && (!isfinite(cfl1))) bad_code = CFL_REASON_BAD_CFL1;
        if ((bad_code == 0) && (!isfinite(cfl2))) bad_code = CFL_REASON_BAD_CFL2;
        if ((bad_code == 0) && (!isfinite(cfl3))) bad_code = CFL_REASON_BAD_CFL3;
        if ((bad_code == 0) && (!isfinite(cfl4))) bad_code = CFL_REASON_BAD_CFL4;
        if ((bad_code == 0) && (!isfinite(cfl5))) bad_code = CFL_REASON_BAD_CFL5;
        if ((bad_code == 0) && (!isfinite(cfl6))) bad_code = CFL_REASON_BAD_CFL6;
        if ((bad_code == 0) && (!isfinite(cfl7))) bad_code = CFL_REASON_BAD_CFL7;
        if ((bad_code == 0) && (!isfinite(cfl8))) bad_code = CFL_REASON_BAD_CFL8;
        if ((bad_code == 0) && (!isfinite(cfl9))) bad_code = CFL_REASON_BAD_CFL9;
        if ((bad_code == 0) && (!isfinite(cfl10))) bad_code = CFL_REASON_BAD_CFL10;
        if ((bad_code == 0) && ((!isfinite(denom2)) || (denom2 <= (real)0.0))) bad_code = CFL_REASON_BAD_DENOM;

        if ((bad_code == 0) && isfinite(denom2) && (denom2 > (real)0.0)) {
          dtime[ll] = CFL/sqrt(denom2);
        } else {
          dtime[ll] = (real)(-1.0);
        }

#ifdef __CUDA_ARCH__
        if (bad_code != 0 ||
            (!isfinite(dtime[ll])) ||
            (dtime[ll] <= (real)0.0) ||
            (dtime[ll] < (real)CFL_ABORT_THRESHOLD)) {

          int reason = bad_code;
          if (reason == 0) {
            if (!isfinite(dtime[ll])) reason = CFL_REASON_DT_NAN;
            else if (dtime[ll] <= (real)0.0) reason = CFL_REASON_DT_NONPOS;
            else reason = CFL_REASON_SMALL_DT;
          }

          if (atomicCAS(&cfl_abort_flag, 0, 1) == 0) {
            cfl_abort_reason = reason;
            cfl_abort_ijk[0] = i;
            cfl_abort_ijk[1] = j;
            cfl_abort_ijk[2] = k;
            cfl_abort_dt     = dtime[ll];

            real t1  = cfl1*cfl1;
            real t2  = cfl2*cfl2;
            real t3  = cfl3*cfl3;
            real t4  = cfl4*cfl4;
            real t5  = cfl5*cfl5;
            real t6  = cfl6*cfl6;
            real t7  = cfl7*cfl7;
            real t8  = cfl8*cfl8;
            real t9  = cfl9*cfl9;
            real t10 = cfl10*cfl10;

            cfl_abort_terms2[0]=t1;  cfl_abort_terms2[1]=t2;  cfl_abort_terms2[2]=t3;  cfl_abort_terms2[3]=t4;
            cfl_abort_terms2[4]=t5;  cfl_abort_terms2[5]=t6;  cfl_abort_terms2[6]=t7;  cfl_abort_terms2[7]=t8;
            cfl_abort_terms2[8]=t9;  cfl_abort_terms2[9]=t10;

            real cand5x = (real)0.0, cand5y = (real)0.0, cand5z = (real)0.0;
#ifdef X
            cand5x = cfl5_a * (real)(4.0*CVNR);
#endif
#ifdef Y
            cand5y = cfl5_b * (real)(4.0*CVNR);
#endif
#ifdef Z
            cand5z = cfl5_c * (real)(4.0*CVNR);
#endif
            cfl_abort_cfl5cand2[0] = cand5x*cand5x;
            cfl_abort_cfl5cand2[1] = cand5y*cand5y;
            cfl_abort_cfl5cand2[2] = cand5z*cand5z;

            int dir = 1;
            real mx5 = cfl_abort_cfl5cand2[0];
            if (cfl_abort_cfl5cand2[1] > mx5) { mx5 = cfl_abort_cfl5cand2[1]; dir = 2; }
            if (cfl_abort_cfl5cand2[2] > mx5) { mx5 = cfl_abort_cfl5cand2[2]; dir = 3; }
            cfl_abort_cfl5dir = dir;

            int id = 1; real mx = t1;
            if (t2>mx){mx=t2; id=2;}
            if (t3>mx){mx=t3; id=3;}
            if (t4>mx){mx=t4; id=4;}
            if (t5>mx){mx=t5; id=5;}
            if (t6>mx){mx=t6; id=6;}
            if (t7>mx){mx=t7; id=7;}
            if (t8>mx){mx=t8; id=8;}
            if (t9>mx){mx=t9; id=9;}
            if (t10>mx){mx=t10; id=10;}
            cfl_abort_termid = id;

            cfl_abort_phys[0] = soundspeed;
            cfl_abort_phys[1] = vxx;
#ifdef Y
            cfl_abort_phys[2] = vy[ll];
#else
            cfl_abort_phys[2] = (real)0.0;
#endif
#ifdef Z
            cfl_abort_phys[3] = vz[ll];
#else
            cfl_abort_phys[3] = (real)0.0;
#endif

            cfl_abort_geom[0] = rloc;
            cfl_abort_geom[1] = thetaloc;
            cfl_abort_geom[2] = Rloc;
            cfl_abort_geom[3] = zloc;
            cfl_abort_geom[4] = Hloc;
            cfl_abort_geom[5] = dxloc;
            cfl_abort_geom[6] = dyloc;
            cfl_abort_geom[7] = dzloc;

            cfl_abort_thermo[0] = rho[ll];
            cfl_abort_thermo[1] = cs[ll];
            cfl_abort_thermo[2] = pgas;
            cfl_abort_thermo[3] = soundspeed;
            cfl_abort_thermo[4] = cs_bg;
            cfl_abort_thermo[5] = cs_eq_ad;
            cfl_abort_thermo[6] = e_eq;
            cfl_abort_thermo[7] = viscosity;
            cfl_abort_thermo[8] = tau_c;
            cfl_abort_thermo[9] = (tau_c > (real)0.0) ? dtime[ll]/tau_c : (real)0.0;

            int llnb = ll;
            real vL = (real)0.0, vR = (real)0.0, dl = (real)1.0;

            if (dir == 1) {
#ifdef X
              llnb = llxp;
              vL = vx[ll];
              vR = vx[llxp];
              dl = dxloc;
#endif
            } else if (dir == 2) {
#ifdef Y
              llnb = llyp;
              vL = vy[ll];
              vR = vy[llyp];
              dl = dyloc;
#endif
            } else {
#ifdef Z
              llnb = llzp;
              vL = vz[ll];
              vR = vz[llzp];
              dl = dzloc;
#endif
            }

            cfl_abort_face[0] = vL;
            cfl_abort_face[1] = vR;
            cfl_abort_face[2] = rho[ll];
            cfl_abort_face[3] = rho[llnb];
            cfl_abort_face[4] = cs[ll];
            cfl_abort_face[5] = cs[llnb];
            if (fluidtype == GAS) {
              cfl_abort_face[6] = sqrt(max((real)0.0, GAMMA*(GAMMA-1.0)*cs[ll]/max(rho[ll],(real)1e-30)));
              cfl_abort_face[7] = sqrt(max((real)0.0, GAMMA*(GAMMA-1.0)*cs[llnb]/max(rho[llnb],(real)1e-30)));
            } else {
              cfl_abort_face[6] = (real)0.0;
              cfl_abort_face[7] = (real)0.0;
            }
            cfl_abort_face[8] = vR - vL;
            cfl_abort_face[9] = dl;
          }
        }
#else
        if (bad_code != 0 ||
            (!isfinite(dtime[ll])) ||
            (dtime[ll] <= (real)0.0) ||
            (dtime[ll] < (real)CFL_ABORT_THRESHOLD)) {

          int reason = bad_code;
          if (reason == 0) {
            if (!isfinite(dtime[ll])) reason = CFL_REASON_DT_NAN;
            else if (dtime[ll] <= (real)0.0) reason = CFL_REASON_DT_NONPOS;
            else reason = CFL_REASON_SMALL_DT;
          }

          printf("\n[CFL-DEBUG-ABORT]\n");
          printf("  reason=%d (%s)\n", reason, CFLReasonName(reason));
          printf("  dt=%+.6e at i=%d j=%d k=%d\n", (double)dtime[ll], i, j, k);
          printf("  rho=%g e=%g soundspeed2=%g soundspeed=%g viscosity=%g\n",
                 (double)rho[ll], (double)cs[ll], (double)soundspeed2,
                 (double)soundspeed, (double)viscosity);
          printf("  geom: r=%g theta=%g R=%g z=%g dx=%g dy=%g dz=%g\n",
                 (double)rloc, (double)thetaloc, (double)Rloc, (double)zloc,
                 (double)dxloc, (double)dyloc, (double)dzloc);
          fflush(stdout);
          exit(1);
        }
#endif
// ====================================================

//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>

//<LAST_BLOCK>
#ifdef __GPU
  CFLAbortCheckAndExit();
#endif

  cfl_b();

#ifdef __GPU
  CFLAbortCheckAndExit();
#endif
//<\LAST_BLOCK>

}
