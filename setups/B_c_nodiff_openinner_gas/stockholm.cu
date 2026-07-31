#define __GPU
#define __NOPROTO

#include "fargo3d.h"
#include <math.h>

// Athena-style meridional damping parameters (Athena default: damping_rate=1.0, width=1.5*H)
#ifndef DAMPING_RATE
#define DAMPING_RATE 1.0
#endif
#ifndef DAMPING_WIDTH_Z
#define DAMPING_WIDTH_Z 1.5
#endif

// Athena-style radial damping parameters
// Inner/outer damping zones (Athena default: ratio=1.2)
#ifndef INNER_DAMPING_RATIO
#define INNER_DAMPING_RATIO 1.2
#endif
#ifndef OUTER_DAMPING_RATIO
#define OUTER_DAMPING_RATIO 1.2
#endif

/* Default dust vertical scale-height ratio used by Stockholm damping.
   This matches the current Athena input in practice (Hratio_1 = 1.0)
   without forcing a broader parameter-plumbing refactor through src/. */
#ifndef DUST_HRATIO
#define DUST_HRATIO 1.0
#endif

/* Keep Stockholm damping available for gas while allowing a control run to
   turn off only the dust relaxation.  The production default remains on. */
#ifndef DUST_STOCKHOLM_DAMPING
#define DUST_STOCKHOLM_DAMPING 1
#endif

#define xmin(i) xmin_s[(i)]
#define ymin(i) ymin_s[(i)]
#define zmin(i) zmin_s[(i)]

CONSTANT(real, xmin_s, 2564);
CONSTANT(real, ymin_s, 2564);
CONSTANT(real, zmin_s, 2564);

__global__ void StockholmBoundary_kernel(real dt,real* rho,
real* rho0,
#ifdef X
real* vx,
real* vx0,
#endif
#ifdef Y
real* vy,
real* vy0,
#endif
#ifdef Z
real* vz,
real* vz0,
#endif
#ifdef ADIABATIC
real* e,
real* e0,
#endif
int pitch,
int stride,
int size_x,
int size_y,
int size_z,
int pitch2d,
real y_min,
real y_max,
real z_min,
real z_max,
real dampingzone,
real kbcol,
real of,
real of0,
real r0,
real ds,
int periodic_z,
int fluidtype,
real aspect_ratio,
real flaring_index,
real damping_rate_val,
real damping_width_z_val,
real epsilon,
real invstokes1,
real dust_hratio) {

  int i;
  int j;
  int k;
  //  Similar to Benitez-Llambay et al. (2016), Eq. 7.
  real Y_inf = y_min*pow(dampingzone, 2.0/3.0);
  real Y_sup = y_max*pow(dampingzone,-2.0/3.0);
  real Z_inf = z_min - (z_max-z_min); // Here we push Z_inf & Z_sup
  real Z_sup = z_max + (z_max-z_min); // out of the mesh
#ifdef CYLINDRICAL
  Z_inf = z_min + (z_max-z_min)*0.1;
  Z_sup = z_max - (z_max-z_min)*0.1;
  if (periodic_z) { // Push Z_inf & Z_sup out of mesh if periodic in Z
    Z_inf = z_min-r0;
    Z_sup = z_max+r0;
  }
#endif

  // Use kbcol for vertical damping (original FARGO behavior)
  // When kbcol > 0, Z_inf and Z_sup are inside the domain
  // This provides meridional damping in the theta direction
  if (kbcol > 0.0) {
    Z_inf = M_PI/2.0 - (M_PI/2.0 - z_min) * kbcol;
    Z_sup = M_PI/2.0 + (M_PI/2.0 - z_min) * kbcol;
  } else {
    // Default: push outside domain (no damping unless kbcol is set)
    Z_inf = z_min - (z_max-z_min);
    Z_sup = z_max + (z_max-z_min);
  }
  real radius;
  real vx0_target;
  real rampy;
  real rampz;
  real rampzz;
  real rampi;
  real ramp;
  real tau;
  real taud;

#ifdef X
i = threadIdx.x + blockIdx.x * blockDim.x;
#else
i = 0;
#endif
#ifdef Y
j = threadIdx.y + blockIdx.y * blockDim.y;
#else
j = 0;
#endif
#ifdef Z
k = threadIdx.z + blockIdx.z * blockDim.z;
#else
k = 0;
#endif

#ifdef Z
if(k>=0 && k<size_z) {
#endif
#ifdef Y
if(j>=0 && j<size_y) {
#endif
#ifdef X
if(i<size_x) {
#endif
	rampy = 0.0;
	rampz = 0.0;
	rampzz = 0.0;
#ifdef Y
	// Athena-style radial damping (de Val-Borro et al. 2006)
	// Damping zone: r < Y_inf (inner) or r > Y_sup (outer)
	real R_cyl_r = ymed(j);
	if (R_cyl_r < 1e-12) R_cyl_r = 1e-12;
	real omega_dyn_y = sqrt(G * MSTAR / (R_cyl_r * R_cyl_r * R_cyl_r));

	if(ymed(j) > Y_sup) {
	  // Outer damping zone
	  real R_func = pow((ymed(j) - Y_sup) / (y_max - Y_sup), 2.0);
	  real inv_damping_tau = damping_rate_val * omega_dyn_y;
	  real alpha = R_func * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampy = (real)1.0 - exp(-alpha);
	}
	if(ymed(j) < Y_inf) {
	  // Inner damping zone
	  real R_func = pow((Y_inf - ymed(j)) / (Y_inf - y_min), 2.0);
	  real inv_damping_tau = damping_rate_val * omega_dyn_y;
	  real alpha = R_func * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampy = (real)1.0 - exp(-alpha);
	}
#endif
#ifdef Z
	// 1. Athena-style meridional damping (Active if damping zone configures)
	real theta_cell = zmed(k);
	real R_cyl = ymed(j) * sin(theta_cell);  // cylindrical radius
	if (R_cyl < 1e-12) R_cyl = 1e-12;
	real omega_dyn = sqrt(G * MSTAR / (R_cyl * R_cyl * R_cyl));

	real h_over_r = aspect_ratio * pow((real)1.0, flaring_index);
	real damping_width = damping_width_z_val * h_over_r;
	real theta_upper = z_min + damping_width;
	real theta_lower = z_max - damping_width;

	// Upper damping zone (near top of disk)
	if (theta_cell <= theta_upper) {
	  real theta_diff = theta_cell - theta_upper;
	  real Theta = (theta_diff * theta_diff) / (damping_width * damping_width);
	  real inv_damping_tau = damping_rate_val * omega_dyn;
	  real alpha = Theta * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampz = (real)1.0 - exp(-alpha);
	}
	// Lower damping zone (near bottom of disk)
	else if (theta_cell >= theta_lower) {
	  real theta_diff = theta_cell - theta_lower;
	  real Theta = (theta_diff * theta_diff) / (damping_width * damping_width);
	  real inv_damping_tau = damping_rate_val * omega_dyn;
	  real alpha = Theta * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampz = (real)1.0 - exp(-alpha);
	}

	// 1b. Apply identical damping to vz using zmin
	real theta_cell_z = zmin(k);
	real R_cyl_z = ymed(j) * sin(theta_cell_z);
	if (R_cyl_z < 1e-12) R_cyl_z = 1e-12;
	real omega_dyn_z = sqrt(G * MSTAR / (R_cyl_z * R_cyl_z * R_cyl_z));

	if (theta_cell_z <= theta_upper) {
	  real theta_diff = theta_cell_z - theta_upper;
	  real Theta = (theta_diff * theta_diff) / (damping_width * damping_width);
	  real inv_damping_tau = damping_rate_val * omega_dyn_z;
	  real alpha = Theta * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampzz = (real)1.0 - exp(-alpha);
	} else if (theta_cell_z >= theta_lower) {
	  real theta_diff = theta_cell_z - theta_lower;
	  real Theta = (theta_diff * theta_diff) / (damping_width * damping_width);
	  real inv_damping_tau = damping_rate_val * omega_dyn_z;
	  real alpha = Theta * inv_damping_tau * dt;
	  if (alpha > (real)0.0) rampzz = (real)1.0 - exp(-alpha);
	}

	// 2. Legacy Z damping (using kbcol) - fallback / additive
	if (theta_cell > Z_sup || theta_cell < Z_inf) {
	  if (zmed(k) > Z_sup) {
	    real rampz_legacy = (zmed(k)-Z_sup)/(z_max-Z_sup);
	    rampz_legacy = rampz_legacy * rampz_legacy;
	    if (rampz < rampz_legacy) rampz = rampz_legacy;
	  }
	  if (zmed(k) < Z_inf) {
	    real rampz_legacy = (Z_inf-zmed(k))/(Z_inf-z_min);
	    rampz_legacy = rampz_legacy * rampz_legacy;
	    if (rampz < rampz_legacy) rampz = rampz_legacy;
	  }
	}
#endif
	if (periodic_z) {
	  rampz = 0.0;
	  rampzz = 0.0;
	}
	ramp = rampy+rampz;
	rampi= rampy+rampzz;
	tau = ds*sqrt(ymed(j)*ymed(j)*ymed(j)/G/MSTAR);
// 	if(ramp>0.0) {
// 	  taud = tau/ramp;
// 	  rho[l] = (rho[l]*taud+rho0[l2D]*dt)/(dt+taud);
// #ifdef X
// 	  vx0_target = vx0[l2D];
// 	  radius = ymed(j);
// #ifdef SPHERICAL
// 	  radius *= sin(zmed(k));
// #endif
// 	  vx0_target -= (of-of0)*radius;
// 	  vx[l] = (vx[l]*taud+vx0_target*dt)/(dt+taud);
// #endif
// #ifdef Y
// 	  vy[l] = (vy[l]*taud+vy0[l2D]*dt)/(dt+taud);
// #endif
// 	}
  if (ramp > 0.0) {
    real alpha = dt * ramp / tau;          // alpha = dt/taud
    real w     = 1.0 - exp(-alpha);        // w = 1 - e^{-alpha}

    if (fluidtype == GAS) {
      rho[l] += (rho0[l2D] - rho[l]) * w;

#ifdef ADIABATIC
      e[l]   += (e0[l2D]   - e[l])   * w;
#endif

#ifdef X
      vx0_target = vx0[l2D];
      radius = ymed(j);
#ifdef SPHERICAL
      radius *= sin(zmed(k));
#endif
      vx0_target -= (of - of0) * radius;
      vx[l] += (vx0_target - vx[l]) * w;
#endif
#ifdef Y
      vy[l] += (vy0[l2D] - vy[l]) * w;
#endif
    }
#if DUST_STOCKHOLM_DAMPING
    else {
      /* Athena-style dust wave damping: in radial damping zones,
         relax dust to analytic NSH dust profiles instead of generic
         background fields. */
      real r_sph = ymed(j);
      real theta = 0.5*M_PI;
      real R_cyl = r_sph;
#ifdef SPHERICAL
      theta = zmed(k);
      R_cyl = r_sph * sin(theta);
#endif
      if (R_cyl < 1.0e-12) R_cyl = 1.0e-12;

      {
        const real dlnP_dlnR = -11.0/4.0;
        real omegaK = sqrt(G * MSTAR / (R_cyl * R_cyl * R_cyl));
        real vK = omegaK * R_cyl;
        real H_over_R = aspect_ratio * pow(R_cyl / r0, flaring_index);
        real cs2 = aspect_ratio * aspect_ratio *
                   pow(R_cyl / r0, 2.0 * flaring_index) *
                   (G * MSTAR / R_cyl);
        real dust_rho_target = (epsilon * pow(R_cyl / r0, -2.25) / dust_hratio) *
                               exp((G * MSTAR) / (dust_hratio * dust_hratio * cs2) *
                                   (1.0 / r_sph - 1.0 / R_cyl));
        if (dust_rho_target < 1.0e-20) dust_rho_target = 1.0e-20;

        {
          real eps_tot = 1.0 + epsilon;
          real eta = 0.5 * H_over_R * H_over_R * dlnP_dlnR;
          real St = 1.0 / invstokes1;
          real vR_d = (2.0 / (St + eps_tot * eps_tot / St)) * eta * vK;
          real vphi_d = (1.0 + (eps_tot * eta) / (eps_tot * eps_tot + St * St)) * vK;

          rho[l] += (dust_rho_target - rho[l]) * w;
#ifdef X
          vx[l] += ((vphi_d - of * R_cyl) - vx[l]) * w;
#endif
#ifdef Y
#ifdef SPHERICAL
          vy[l] += (vR_d * sin(theta) - vy[l]) * w;
#else
          vy[l] += (vR_d - vy[l]) * w;
#endif
#endif
#ifdef Z
          vz[l] += (vR_d * cos(theta) - vz[l]) * w;
#endif
        }
      }
    }
#endif
  }
#ifdef Z
  if (rampi > 0.0 && fluidtype == GAS) {
    real alpha = dt * rampi / tau;
    real w     = 1.0 - exp(-alpha);
    vz[l] += (vz0[l2D] - vz[l]) * w;
  }
#endif
#ifdef X
 }
 #endif
#ifdef Y
 }
 #endif
#ifdef Z
 }
 #endif
}

extern "C" void StockholmBoundary_gpu(real dt) {


  INPUT(Density);
  INPUT2D(Density0);
  OUTPUT(Density);
#ifdef ADIABATIC
  INPUT(Energy);
  INPUT2D(Energy0);
  OUTPUT(Energy);
#endif
#ifdef X
  INPUT(Vx);
  INPUT2D(Vx0);
  OUTPUT(Vx);
#endif
#ifdef Y
  INPUT(Vy);
  INPUT2D(Vy0);
  OUTPUT(Vy);
#endif
#ifdef Z
  INPUT(Vz);
  INPUT2D(Vz0);
  OUTPUT(Vz);
#endif

dim3 block (BLOCK_X, BLOCK_Y, BLOCK_Z);
dim3 grid ((Nx+2*NGHX+block.x-1)/block.x,
((Ny+2*NGHY)+block.y-1)/block.y,
((Nz+2*NGHZ)+block.z-1)/block.z);

#ifdef BIGMEM
#define xmin_d &Xmin_d
#define ymin_d &Ymin_d
#define zmin_d &Zmin_d
#define Sxj_d &Sxj_d
#define Syj_d &Syj_d
#define Szj_d &Szj_d
#define Sxk_d &Sxk_d
#define Syk_d &Syk_d
#define Szk_d &Szk_d
#define Sxi_d &Sxi_d
#define InvVj_d &InvVj_d
#define InvDiffXmed_d &InvDiffXmed_d
#endif

CUDAMEMCPY(xmin_s, xmin_d, sizeof(real)*(Nx+1), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(ymin_s, ymin_d, sizeof(real)*(Ny+2*NGHY+1), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(zmin_s, zmin_d, sizeof(real)*(Nz+2*NGHZ+1), 0, cudaMemcpyDeviceToDevice);


cudaFuncSetCacheConfig(StockholmBoundary_kernel, cudaFuncCachePreferL1 );
StockholmBoundary_kernel<<<grid,block>>>(dt,
Density->field_gpu,
Density0->field_gpu,
#ifdef X
Vx->field_gpu,
Vx0->field_gpu,
#endif
#ifdef Y
Vy->field_gpu,
Vy0->field_gpu,
#endif
#ifdef Z
Vz->field_gpu,
Vz0->field_gpu,
#endif
#ifdef ADIABATIC
Energy->field_gpu,
Energy0->field_gpu,
#endif
Pitch_gpu,
Stride_gpu,
Nx+2*NGHX,
Ny+2*NGHY,
Nz+2*NGHZ,
Pitch2D,
YMIN,
YMAX,
ZMIN,
ZMAX,
DAMPINGZONE,
KILLINGBCCOLATITUDE,
OMEGAFRAME,
OMEGAFRAME0,
R0,
TAUDAMP,
PERIODICZ,
Fluidtype,
ASPECTRATIO,
FLARINGINDEX,
DAMPING_RATE,
DAMPING_WIDTH_Z,
EPSILON,
INVSTOKES1,
DUST_HRATIO);

check_errors("StockholmBoundary_kernel");

}
