//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

#define __NOPROTO
#define __GPU

//<INCLUDES>
#include "fargo3d.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//<\INCLUDES>

extern int Pitch_gpu;

__device__ __forceinline__ real clamp_real(real x, real xmin, real xmax) {
  return (x < xmin) ? xmin : ((x > xmax) ? xmax : x);
}

__device__ __forceinline__ real abs_real(real x) {
  return (x < (real)0.0) ? -x : x;
}

__device__ __forceinline__ real max_real(real a, real b) {
  return (a > b) ? a : b;
}

__device__ __forceinline__ real sqrt_real(real x) {
  return (x > (real)0.0) ? sqrt(x) : (real)0.0;
}

// ----------------------------------------------------------------------
// Kernel: Apply minimally invasive floors / limiters.
// Philosophy:
//  - Gas: only protect very low-density high-altitude cells.
//  - Dust: only touch clearly bad / near-floor cells; do not globally reset.
// ----------------------------------------------------------------------
__global__ void Apply_Floor_Kernel(real *dens,
#ifdef ADIABATIC
                                  real *energy,
#else
                                  real * /*energy*/,
#endif
                                  real *vx, real *vy, real *vz,
                                  const real *ymin, const real *zmin,
                                  real *gas_dens, real *gas_vx, real *gas_vy, real *gas_vz,
                                  int size_x, int size_y, int size_z, int pitch,
                                  int fluidtype,
                                  real gas_abs_floor, real dust_abs_floor, real energy_floor,
                                  real dust_percent_floor, int tv_cap_flag,
                                  real tv_cap_factor, real tv_cap_d2g_max,
                                  real tv_cap_rho_floor_factor, real tv_cap_low_gas_floor_factor,
                                  real tv_cap_vratio_override, real tv_cap_mach_override,
                                  real tv_cap_z_hg_ratio,
                                  real aspectratio, real flaringindex, real r0,
                                  real gm, real invstokes, real epsilon) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = size_x * size_y * size_z;
  if (idx >= total) return;

  int i = idx % size_x;
  int t = idx / size_x;
  int j = t % size_y;
  int k = t / size_y;
  int mem = i + pitch * (j + k * size_y);

  real y_val = ymin[j];
#ifdef Z
  real z_val = (real)0.5 * (zmin[k] + zmin[k+1]);
#else
  real z_val = (real)0.0;
#endif

  // ------------------------------------------------------------------
  // 1. Density floor.
  // ------------------------------------------------------------------
  real limit_floor = (fluidtype == GAS) ? gas_abs_floor : dust_abs_floor;
  real d = dens[mem];
  if (!(d >= limit_floor)) dens[mem] = limit_floor;

  // ------------------------------------------------------------------
  // 2. NaN / Inf intercept for velocities.
  // ------------------------------------------------------------------
  real v_x = vx[mem];
  if (!(v_x >= (real)-1.0e30 && v_x <= (real)1.0e30)) v_x = (real)0.0;

  real v_y = vy[mem];
  if (!(v_y >= (real)-1.0e30 && v_y <= (real)1.0e30)) v_y = (real)0.0;

#ifdef Z
  real v_z = (real)0.0;
  if (vz != NULL) {
    v_z = vz[mem];
    if (!(v_z >= (real)-1.0e30 && v_z <= (real)1.0e30)) v_z = (real)0.0;
  }
#endif

  // Geometry helpers.
  real R_cyl = y_val;
#ifdef Z
  real r_sph = y_val;
  real theta = z_val;
  R_cyl = r_sph * sin(z_val);
#endif
  if (R_cyl < (real)1.0e-6) R_cyl = (real)1.0e-6;

  real vK = sqrt(gm / R_cyl);
  real h  = aspectratio * pow(R_cyl / r0, flaringindex);
  real cs_ana = h * vK;

  // ------------------------------------------------------------------
  // 3A. Gas: only limit very low-density, high-altitude v_theta.
  //     This does NOT touch the midplane flow that sets dust leakage.
  // ------------------------------------------------------------------
#ifdef Z
#ifdef SPHERICAL
  if (fluidtype == GAS && vz != NULL) {
    const real NH_CEIL       = (real)3.5;
    const real MACH_CEIL     = (real)1.0;
    const real LOW_DENS_MULT = (real)150.0;

    if (dens[mem] < LOW_DENS_MULT * gas_abs_floor) {
      real dth = abs_real(z_val - (real)(0.5 * M_PI));
      if (dth > NH_CEIL * h) {
        real vlim = MACH_CEIL * cs_ana;
        v_z = clamp_real(v_z, -vlim, vlim);
      }
    }
  }
#endif
#endif

  // ------------------------------------------------------------------
  // 3B. Dust: only repair clearly bad / near-floor cells.
  //     No global reset. Preserve 3D leakage physics in healthy cells.
  // ------------------------------------------------------------------
  if (fluidtype != GAS) {
    real rhod = dens[mem];
    real rhog = gas_dens[mem];
    if (!(rhog > (real)0.0)) rhog = gas_abs_floor;

    real vg_x = gas_vx[mem];
    real vg_y = gas_vy[mem];
#ifdef Z
    real vg_z = (gas_vz != NULL) ? gas_vz[mem] : (real)0.0;
#else
    real vg_z = (real)0.0;
#endif

    real dust_to_gas = rhod / rhog;

    // ------------------------------------------------------------------
    // 3B1. Dust percent floor: bump up very low dust relative to initial profile
    // ------------------------------------------------------------------
    if (dust_percent_floor > (real)0.0) {
      // Compute analytic dust density at this cell (same as condinit.c)
#ifdef Z
      real z_cyl = r_sph * cos(theta);
#else
      real z_cyl = (real)0.0;
#endif
      if (R_cyl < (real)1.0e-6) R_cyl = (real)1.0e-6;

      real H_over_R = aspectratio * pow(R_cyl / r0, flaringindex);
      real rho_g_init = pow(R_cyl / r0, -2.25) *
                        exp((R_cyl / (aspectratio * aspectratio * pow(R_cyl / r0, 2.0 * flaringindex))) *
                            (1.0 / r_sph - 1.0 / R_cyl));
      real rho_d_init = epsilon * rho_g_init;

      real min_dust = dust_percent_floor * rho_d_init;
      if (rhod < min_dust) {
        // Bump dust density up (Athena adds 0.25 dex = 1.778x extra)
        real old_rhod = rhod;
        rhod = (real)1.7782794100389227 * min_dust;
        dens[mem] = rhod;

        // Scale dust velocity to conserve momentum: v_new = (rho_old/rho_new)*v_old
        real scale = old_rhod / rhod;
        v_x = v_x * scale;
        v_y = v_y * scale;
#ifdef Z
        if (vz != NULL) v_z = v_z * scale;
#endif
      }
    }

    // ------------------------------------------------------------------
    // 3B2. Gated terminal-velocity cap (Athena++ style)
    // ------------------------------------------------------------------
    if (tv_cap_flag == 1) {
      // Compute dust and gas velocities
      real dv_x = v_x - vg_x;
      real dv_y = v_y - vg_y;
#ifdef Z
      real dv_z = v_z - vg_z;
#else
      real dv_z = (real)0.0;
#endif

      // Dust and gas speed squared
      real dust_vel_abs2 = dv_x*dv_x + dv_y*dv_y + dv_z*dv_z;
      real gas_vel_abs2 = vg_x*vg_x + vg_y*vg_y + vg_z*vg_z;
      real gas_vel_abs2_safe = (gas_vel_abs2 > (real)1.0e-24) ? gas_vel_abs2 : (real)1.0e-24;
      real gas_cs2 = cs_ana * cs_ana;

      // Determine if we should apply the cap
      int apply_cap = 0;
      int force_override = 0;
      int force_low_gas_cap = 0;

      // Override 1: extreme |v_d|/|v_g| ratio
      if (dust_vel_abs2 > tv_cap_vratio_override * tv_cap_vratio_override * gas_vel_abs2_safe) {
        force_override = 1;
      }
      // Override 2: extreme |v_d|/cs (Mach-like)
      if (tv_cap_mach_override > (real)0.0 &&
          dust_vel_abs2 > tv_cap_mach_override * tv_cap_mach_override * gas_cs2) {
        force_override = 1;
      }
      // Override 3: high altitude (|z| > z_hg_ratio * H)
      // Athena only checks this when dust significantly exceeds gas: > 20x
#ifdef Z
      if (dust_vel_abs2 > (real)20.0 * gas_vel_abs2_safe) {
        real z_cyl = r_sph * cos(theta);
        real omega_dyn = sqrt(gm / (R_cyl * R_cyl * R_cyl));
        real hg = cs_ana / omega_dyn;
        if (abs_real(z_cyl) > tv_cap_z_hg_ratio * hg) {
          force_override = 1;
          force_low_gas_cap = 1;
        }
      }
#endif

      // D2G gating: Athena uses both rho_d < rho_floor_factor*dust_rho_floor AND d2g < d2g_max
      // Without override, both conditions must be true
      bool d2g_gated = (rhod <= tv_cap_rho_floor_factor * dust_abs_floor && dust_to_gas < tv_cap_d2g_max);
      if (!d2g_gated && force_override == 0) {
        // Not gated and no override - skip
      } else {
        // Additional fast skip: dust must significantly exceed gas (Athena uses 4x)
        if (dust_vel_abs2 > (real)4.0 * gas_vel_abs2_safe) {
          apply_cap = 1;
        }
      }

      // Apply cap using Athena++ NSH terminal velocity formula
      if (apply_cap) {
        // Determine if we're in low-gas regime
        bool use_low_gas_cap = (rhog < tv_cap_low_gas_floor_factor * gas_abs_floor || force_low_gas_cap == 1);

        real vrel_cap1, vrel_cap2, vrel_cap3;

#ifdef Z
        const real dlnP_dlnR_tv = (real)-2.75;
#else
        const real dlnP_dlnR_tv = (real)-1.5;
#endif
        real St  = (invstokes > (real)0.0) ? ((real)1.0 / invstokes) : (real)0.0;
        real eta = (real)0.5 * h * h * dlnP_dlnR_tv;
        real St2 = St * St;

        if (use_low_gas_cap) {
          // Low-gas regime: use profile-based terminal velocity estimate
          // Athena: vrel_cap1 = factor * |2*St/(1+St²) * eta * vK|, vrel_cap3 = factor * |St²/(1+St²) * eta * vK|
          real vrel_r = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
          real vrel_phi = (real)0.0;
          real vrel_theta = tv_cap_factor * abs_real(St2 / ((real)1.0 + St2)) * eta * vK;

          // Map to spherical components (simplified: v_r and v_theta use same magnitude in this approximation)
          vrel_cap1 = vrel_r;   // radial
          vrel_cap2 = vrel_phi; // phi (azimuthal) - no radial drift in phi direction
          vrel_cap3 = vrel_theta; // theta
        } else {
          // Regular regime: use NSH terminal velocity formula (simplified version)
          // For now, use the same profile-based cap as fallback since we don't have pressure gradient access
          real vrel_cap = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;

          // Fallback: use cs-based cap if St is very small/large
          if (vrel_cap < (real)0.1 * cs_ana) {
            vrel_cap = tv_cap_factor * cs_ana;
          }

          vrel_cap1 = vrel_cap;
          vrel_cap2 = vrel_cap;
          vrel_cap3 = vrel_cap;
        }

        // Apply vector cap: compare |dv|² vs |vrel_cap|², use uniform damping
        real rel_vel2_tot = dv_x*dv_x + dv_y*dv_y + dv_z*dv_z;
        real vrel_cap2_tot = vrel_cap1*vrel_cap1 + vrel_cap2*vrel_cap2 + vrel_cap3*vrel_cap3;

        if (rel_vel2_tot > vrel_cap2_tot) {
          // Athena-style: uniform damping that preserves direction
          real damping = sqrt_real(vrel_cap2_tot / rel_vel2_tot);
          v_x = vg_x + damping * dv_x;
          v_y = vg_y + damping * dv_y;
#ifdef Z
          if (vz != NULL) v_z = vg_z + damping * dv_z;
#endif
        }
        // Otherwise, keep original velocities (no change needed)
      }
    }

    // ------------------------------------------------------------------
    // 3B3. Existing near-floor repair logic (keep for compatibility)
    // ------------------------------------------------------------------
    real near_floor  = (rhod < (real)30.0 * dust_abs_floor) ? (real)1.0 : (real)0.0;

    // Small-St terminal drift relative to gas.
#ifdef Z
    const real dlnP_dlnR = (real)-2.75;
#else
    const real dlnP_dlnR = (real)-1.5;
#endif
    real St  = (invstokes > (real)0.0) ? ((real)1.0 / invstokes) : (real)0.0;
    real eta = (real)0.5 * h * h * dlnP_dlnR;
    real dv_r_theo   = ((real)2.0 * St / ((real)1.0 + St * St)) * eta * vK;
    real dv_phi_theo = (-(St * St) / ((real)1.0 + St * St)) * eta * vK;

    real dv_x = v_x - vg_x;
    real dv_y = v_y - vg_y;
#ifdef Z
    real dv_z = v_z - vg_z;
#else
    real dv_z = (real)0.0;
#endif

    real drift_cap = (real)8.0 * cs_ana;
    real bad_drift =
      (abs_real(dv_x) > drift_cap) ||
      (abs_real(dv_y) > drift_cap) ||
#ifdef Z
      (abs_real(dv_z) > drift_cap) ||
#endif
      (!(v_x >= (real)-1.0e30 && v_x <= (real)1.0e30)) ||
      (!(v_y >= (real)-1.0e30 && v_y <= (real)1.0e30));

#ifdef Z
    if (vz != NULL) {
      bad_drift = bad_drift || (!(v_z >= (real)-1.0e30 && v_z <= (real)1.0e30));
    }
#endif

    // Worst cells: dust-poor + near floor -> reset only the planar drift,
    // but do NOT force theta difference to zero.
    if ((near_floor > (real)0.5) && (dust_to_gas < (real)1.0e-4)) {
      v_y = vg_y + dv_r_theo;
      v_x = vg_x + dv_phi_theo;
#ifdef Z
      if (vz != NULL) {
        real dvz_cap = (real)2.0 * cs_ana;
        v_z = vg_z + clamp_real(dv_z, -dvz_cap, dvz_cap);
      }
#endif
    }
    // Otherwise only clamp clearly excessive drift; do not hard reset.
    else if (bad_drift) {
      real cap_r   = max_real((real)6.0 * abs_real(dv_r_theo),   (real)8.0 * cs_ana);
      real cap_phi = max_real((real)6.0 * abs_real(dv_phi_theo), (real)8.0 * cs_ana);
      real cap_th  = (real)8.0 * cs_ana;

      dv_y = clamp_real(dv_y, -cap_r,   cap_r);
      dv_x = clamp_real(dv_x, -cap_phi, cap_phi);
#ifdef Z
      dv_z = clamp_real(dv_z, -cap_th,  cap_th);
#endif

      v_y = vg_y + dv_y;
      v_x = vg_x + dv_x;
#ifdef Z
      if (vz != NULL) v_z = vg_z + dv_z;
#endif
    }
  }

  // Write healthy velocities back.
  vx[mem] = v_x;
  vy[mem] = v_y;
#ifdef Z
  if (vz != NULL) vz[mem] = v_z;
#endif

  // ------------------------------------------------------------------
  // 4. Energy floor. Keep collaborator's relaxed gas pressure floor.
  //    Dust energy is not physical in this setup; keep it zero.
  // ------------------------------------------------------------------
#ifdef ADIABATIC
  if (energy != NULL) {
    if (fluidtype == GAS) {
      real e = energy[mem];
      if (!(e >= energy_floor)) energy[mem] = energy_floor;
    } else {
      energy[mem] = (real)0.0;
    }
  }
#endif
}

extern "C" void Floor_gpu() {
  int sx = Nx + 2 * NGHX;
  int sy = Ny + 2 * NGHY;
  int sz = Nz + 2 * NGHZ;
  int total_cells = sx * sy * sz;

  real *d_dens = Density->field_gpu;
  real *d_vx   = Vx->field_gpu;
  real *d_vy   = Vy->field_gpu;
#ifdef Z
  real *d_vz   = (Vz != NULL) ? Vz->field_gpu : NULL;
  real *d_zmin = Zmin_d;
#else
  real *d_vz   = NULL;
  real *d_zmin = NULL;
#endif

#ifdef ADIABATIC
  real *d_energy = (Energy != NULL) ? Energy->field_gpu : NULL;
#else
  real *d_energy = NULL;
#endif

  real *d_gas_dens = Fluids[0]->Density->field_gpu;
  real *d_gas_vx   = Fluids[0]->Vx->field_gpu;
  real *d_gas_vy   = Fluids[0]->Vy->field_gpu;
#ifdef Z
  real *d_gas_vz   = (Fluids[0]->Vz != NULL) ? Fluids[0]->Vz->field_gpu : NULL;
#else
  real *d_gas_vz   = NULL;
#endif

  // Use parameters from .par file (with safe defaults if not defined)
  // GASFLOOR, ENERGYFLOOR, DUST_PERCENT_FLOOR, DUST_TVCAP_*, etc.
  const real gas_abs_floor = (real)GASFLOOR;
  const real energy_floor  = (real)ENERGYFLOOR;
  const real dust_percent_floor = (real)DUST_PERCENT_FLOOR;
  const int  tv_cap_flag = (int)DUST_TVCAP_FLAG;
  const real tv_cap_factor = (real)DUST_TVCAP_FACTOR;
  const real tv_cap_d2g_max = (real)DUST_TVCAP_D2G_MAX;
  const real tv_cap_rho_floor_factor = (real)DUST_TVCAP_RHO_FLOOR_FACTOR;
  const real tv_cap_low_gas_floor_factor = (real)DUST_TVCAP_LOW_GAS_FLOOR_FACTOR;
  const real tv_cap_vratio_override = (real)DUST_TVCAP_VRATIO_OVERRIDE;
  const real tv_cap_mach_override = (real)DUST_TVCAP_MACH_OVERRIDE;
  const real tv_cap_z_hg_ratio = (real)DUST_TVCAP_Z_HG_RATIO;

  // Dust floor: use par value but cap at 1e-9 for high-res 3D to avoid
  // manufacturing dust in dust-poor regions
  const real par_dust_floor = (real)DUSTFLOOR;
  const real dust_abs_floor = (par_dust_floor < (real)1.0e-9) ? par_dust_floor : (real)1.0e-9;

  const real aspectratio  = ASPECTRATIO;
  const real flaringindex = FLARINGINDEX;
  const real r0           = R0;
  const real gm           = G * MSTAR;
  const real invstokes    = INVSTOKES1;
  const real epsilon      = EPSILON;

  int blockSize = 256;
  int gridSize  = (total_cells + blockSize - 1) / blockSize;

  Apply_Floor_Kernel<<<gridSize, blockSize>>>(
      d_dens,
      d_energy,
      d_vx, d_vy, d_vz,
      Ymin_d, d_zmin,
      d_gas_dens, d_gas_vx, d_gas_vy, d_gas_vz,
      sx, sy, sz, Pitch_gpu,
      Fluidtype,
      gas_abs_floor, dust_abs_floor, energy_floor,
      dust_percent_floor, tv_cap_flag, tv_cap_factor, tv_cap_d2g_max,
      tv_cap_rho_floor_factor, tv_cap_low_gas_floor_factor,
      tv_cap_vratio_override, tv_cap_mach_override, tv_cap_z_hg_ratio,
      aspectratio, flaringindex, r0,
      gm, invstokes, epsilon
  );

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Floor launch failed on rank %d: %s\n",
            CPU_Rank, cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Floor sync failed on rank %d: %s\n",
            CPU_Rank, cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }
}
