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

// External planetary system data (defined in FARGO3D global variables)
// Note: For GPU compatibility, we use fixed planet position in co-rotating frame
// Planet is at r=1, phi=0, z=0 in the co-rotating frame

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

__device__ __forceinline__ int round_to_int(real x) {
  return (int)floor(x + (real)0.5);
}

__device__ __forceinline__ real psi_x_local(real x, real x_min, real x_max,
                                            real x_ma, real x_mb, real x_mc) {
  real x_mesh_i = (x_max - x_min) + x_mc * (x_ma + x_mb);
  real ax = abs_real(x);
  if (ax <= x_ma) return (1.0 + x_mc) / x_mesh_i;
  if (ax < x_mb) {
    real arg = (real)M_PI * (ax - x_ma) / ((real)2.0 * (x_mb - x_ma));
    real c = cos(arg);
    return ((real)1.0 + x_mc * c * c) / x_mesh_i;
  }
  return (real)1.0 / x_mesh_i;
}

__device__ __forceinline__ real psi_y_local(real y, real y_min, real y_max, real y_my0,
                                            real y_ma, real y_mb, real y_mc) {
  real y_mesh_i = log(y_max / y_min) + y_mc * (y_ma + y_mb);
  real ay = abs_real(y - y_my0);
  if (ay <= y_ma) return ((real)1.0 / y + y_mc) / y_mesh_i;
  if (ay < y_mb) {
    real arg = (real)M_PI * (ay - y_ma) / ((real)2.0 * (y_mb - y_ma));
    real c = cos(arg);
    return ((real)1.0 / y + y_mc * c * c) / y_mesh_i;
  }
  return ((real)1.0 / y) / y_mesh_i;
}

__device__ __forceinline__ int tvcap_proxy_max_level(real x_mc, real y_mc) {
  real refine_x = (real)1.0 + x_mc;
  real refine_y = (real)1.0 + y_mc;
  real refine_max = (refine_x > refine_y) ? refine_x : refine_y;
  if (refine_max < (real)1.0) refine_max = (real)1.0;
  return round_to_int(log(refine_max) / log((real)2.0));
}

__device__ __forceinline__ int tvcap_proxy_level(real phi, real rad,
                                                 real x_min, real x_max, real x_ma, real x_mb, real x_mc,
                                                 real y_min, real y_max, real y_my0, real y_ma, real y_mb, real y_mc) {
  real refine_x = psi_x_local(phi, x_min, x_max, x_ma, x_mb, x_mc) * (x_max - x_min);
  real refine_y = psi_y_local(rad, y_min, y_max, y_my0, y_ma, y_mb, y_mc) * rad * log(y_max / y_min);
  real refine_local = (refine_x > refine_y) ? refine_x : refine_y;
  if (refine_local < (real)1.0) return 0;
  return round_to_int(log(refine_local) / log((real)2.0));
}

__device__ __forceinline__ real median6_real_device(real values[6]) {
  for (int i = 0; i < 5; ++i) {
    for (int j = i + 1; j < 6; ++j) {
      if (values[i] > values[j]) {
        real tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
      }
    }
  }
  return values[3];
}

__global__ void RelmedianSweepKernel(real *dens,
                                     real *vx,
                                     real *vy,
                                     real *vz,
                                     const real *src_dens,
                                     const real *src_vx,
                                     const real *src_vy,
                                     const real *src_vz,
                                     int size_x,
                                     int size_y,
                                     int size_z,
                                     int pitch,
                                     real cap_factor,
                                     real rho_floor_factor,
                                     int *changes) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = size_x * size_y * size_z;
  if (idx >= total) return;

  int i = idx % size_x;
  int t = idx / size_x;
  int j = t % size_y;
  int k = t / size_y;
  int mem = i + pitch * (j + k * size_y);
  int stride = pitch * size_y;

  if (i <= 0 || i >= size_x - 1) return;
  if (j <= 0 || j >= size_y - 1) return;
  if (k <= 0 || k >= size_z - 1) return;

  real neighbor_vel[6];
  real dust_vel2;
  real median_vel2;
  real median_cap_sq;

  neighbor_vel[0] = src_vx[mem + 1]      * src_vx[mem + 1]      + src_vy[mem + 1]      * src_vy[mem + 1]      + ((vz != NULL) ? src_vz[mem + 1]      * src_vz[mem + 1]      : (real)0.0);
  neighbor_vel[1] = src_vx[mem - 1]      * src_vx[mem - 1]      + src_vy[mem - 1]      * src_vy[mem - 1]      + ((vz != NULL) ? src_vz[mem - 1]      * src_vz[mem - 1]      : (real)0.0);
  neighbor_vel[2] = src_vx[mem + pitch]  * src_vx[mem + pitch]  + src_vy[mem + pitch]  * src_vy[mem + pitch]  + ((vz != NULL) ? src_vz[mem + pitch]  * src_vz[mem + pitch]  : (real)0.0);
  neighbor_vel[3] = src_vx[mem - pitch]  * src_vx[mem - pitch]  + src_vy[mem - pitch]  * src_vy[mem - pitch]  + ((vz != NULL) ? src_vz[mem - pitch]  * src_vz[mem - pitch]  : (real)0.0);
  neighbor_vel[4] = src_vx[mem + stride] * src_vx[mem + stride] + src_vy[mem + stride] * src_vy[mem + stride] + ((vz != NULL) ? src_vz[mem + stride] * src_vz[mem + stride] : (real)0.0);
  neighbor_vel[5] = src_vx[mem - stride] * src_vx[mem - stride] + src_vy[mem - stride] * src_vy[mem - stride] + ((vz != NULL) ? src_vz[mem - stride] * src_vz[mem - stride] : (real)0.0);

  median_vel2 = median6_real_device(neighbor_vel);
  if (median_vel2 <= (real)1.0e-20) return;

  dust_vel2 = src_vx[mem] * src_vx[mem] + src_vy[mem] * src_vy[mem] +
              ((vz != NULL) ? src_vz[mem] * src_vz[mem] : (real)0.0);
  median_cap_sq = cap_factor * cap_factor * median_vel2;
  if (dust_vel2 <= median_cap_sq) return;

  {
    real damping = sqrt_real(median_cap_sq / dust_vel2);
    vx[mem] = src_vx[mem] * damping;
    vy[mem] = src_vy[mem] * damping;
    if (vz != NULL) vz[mem] = src_vz[mem] * damping;
  }

  if (rho_floor_factor > (real)0.0) {
    real neighbor_rho[6];
    real rho_floor_local;

    neighbor_rho[0] = src_dens[mem + 1];
    neighbor_rho[1] = src_dens[mem - 1];
    neighbor_rho[2] = src_dens[mem + pitch];
    neighbor_rho[3] = src_dens[mem - pitch];
    neighbor_rho[4] = src_dens[mem + stride];
    neighbor_rho[5] = src_dens[mem - stride];
    rho_floor_local = rho_floor_factor * median6_real_device(neighbor_rho);
    if (dens[mem] < rho_floor_local) dens[mem] = rho_floor_local;
  }

  atomicAdd(changes, 1);
}

static void EnsureRelmedianGpuBuffers(size_t total,
                                      real **src_dens,
                                      real **src_vx,
                                      real **src_vy,
                                      real **src_vz,
                                      int **changes,
                                      size_t *capacity) {
  cudaError_t err;

  if (total <= *capacity) return;

  if (*src_dens != NULL) cudaFree(*src_dens);
  if (*src_vx   != NULL) cudaFree(*src_vx);
  if (*src_vy   != NULL) cudaFree(*src_vy);
  if (*src_vz   != NULL) cudaFree(*src_vz);
  if (*changes  != NULL) cudaFree(*changes);

  err = cudaMalloc((void **)src_dens, total * sizeof(real));
  if (err != cudaSuccess) goto alloc_fail;
  err = cudaMalloc((void **)src_vx, total * sizeof(real));
  if (err != cudaSuccess) goto alloc_fail;
  err = cudaMalloc((void **)src_vy, total * sizeof(real));
  if (err != cudaSuccess) goto alloc_fail;
  err = cudaMalloc((void **)src_vz, total * sizeof(real));
  if (err != cudaSuccess) goto alloc_fail;
  err = cudaMalloc((void **)changes, sizeof(int));
  if (err != cudaSuccess) goto alloc_fail;

  *capacity = total;
  return;

alloc_fail:
  fprintf(stderr, "Relmedian GPU buffer allocation failed on rank %d: %s\n",
          CPU_Rank, cudaGetErrorString(err));
  fflush(stderr);
  exit(1);
}

static void ApplyRelmedianIterativeGpu(real *dens,
                                       real *vx,
                                       real *vy,
                                       real *vz,
                                       int size_x,
                                       int size_y,
                                       int size_z,
                                       int pitch,
                                       int max_iters,
                                       real cap_factor,
                                       real rho_floor_factor) {
  static real *src_dens = NULL;
  static real *src_vx   = NULL;
  static real *src_vy   = NULL;
  static real *src_vz   = NULL;
  static int  *changes_d = NULL;
  static size_t capacity = 0;
  const size_t total_buf = (size_t)pitch * (size_t)size_y * (size_t)size_z;
  const size_t total_cells = (size_t)size_x * (size_t)size_y * (size_t)size_z;
  const size_t bytes = total_buf * sizeof(real);
  const int blockSize = 256;
  const int gridSize = (int)((total_cells + (size_t)blockSize - 1) / (size_t)blockSize);
  cudaError_t err;

  if (max_iters < 1 || cap_factor <= (real)0.0) return;

  EnsureRelmedianGpuBuffers(total_buf, &src_dens, &src_vx, &src_vy, &src_vz, &changes_d, &capacity);

  for (int iter = 0; iter < max_iters; ++iter) {
    int changes_h = 0;

    err = cudaMemcpy(src_dens, dens, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto copy_fail;
    err = cudaMemcpy(src_vx, vx, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto copy_fail;
    err = cudaMemcpy(src_vy, vy, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto copy_fail;
    if (vz != NULL) {
      err = cudaMemcpy(src_vz, vz, bytes, cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess) goto copy_fail;
    }
    err = cudaMemset(changes_d, 0, sizeof(int));
    if (err != cudaSuccess) goto copy_fail;

    RelmedianSweepKernel<<<gridSize, blockSize>>>(
        dens, vx, vy, vz,
        src_dens, src_vx, src_vy, src_vz,
        size_x, size_y, size_z, pitch,
        cap_factor, rho_floor_factor,
        changes_d);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Relmedian kernel launch failed on rank %d: %s\n",
              CPU_Rank, cudaGetErrorString(err));
      fflush(stderr);
      exit(1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr, "Relmedian kernel sync failed on rank %d: %s\n",
              CPU_Rank, cudaGetErrorString(err));
      fflush(stderr);
      exit(1);
    }

    err = cudaMemcpy(&changes_h, changes_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto copy_fail;
    if (changes_h == 0) break;
  }

  return;

copy_fail:
  fprintf(stderr, "Relmedian GPU copy failed on rank %d: %s\n",
          CPU_Rank, cudaGetErrorString(err));
  fflush(stderr);
  exit(1);
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
                                  const real *ymin, const real *zmin, const real *xmin,
                                  real *gas_dens, real *gas_vx, real *gas_vy, real *gas_vz,
#ifdef ADIABATIC
                                  real *gas_energy,
#endif
                                  int size_x, int size_y, int size_z, int pitch,
                                  int fluidtype,
                                  real gas_abs_floor, real dust_abs_floor, real energy_floor,
                                  real dust_percent_floor, int tv_cap_flag,
                                  real tv_cap_factor, real tv_cap_d2g_max,
                                  real tv_cap_rho_floor_factor, real tv_cap_low_gas_floor_factor,
                                  real tv_cap_vratio_override, real tv_cap_mach_override,
                                  real tv_cap_z_hg_ratio, int tv_cap_top_n_phys_levels,
                                  real mesh_xmin, real mesh_xmax, real mesh_xma, real mesh_xmb, real mesh_xmc,
                                  real mesh_ymin, real mesh_ymax, real mesh_ymy0, real mesh_yma, real mesh_ymb, real mesh_ymc,
                                  int v_cap_flag, real v_cap_factor, real v_cap_radius_ratio,
                                  real planet_x, real planet_y, real planet_z, real planet_mass, real planet_soft,
                                  real planet_radius,
                                  real aspectratio, real flaringindex, real r0,
                                  real gm, real invstokes, real epsilon,
#ifdef ADIABATIC
                                  real gamma,
#endif
                                  // Anomaly check parameters
                                  int anomaly_check_flag, real anomaly_check_r_ratio, real anomaly_check_hill_ratio,
                                  // Relative median cap parameters
                                  int relmedian_cap_flag, real relmedian_cap_factor) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = size_x * size_y * size_z;
  if (idx >= total) return;

  int i = idx % size_x;
  int t = idx / size_x;
  int j = t % size_y;
  int k = t / size_y;
  int mem = i + pitch * (j + k * size_y);

  real y_val = ymin[j];
  real x_val = (real)0.5 * (xmin[i] + xmin[i+1]);
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
      int max_proxy_level = tvcap_proxy_max_level(mesh_xmc, mesh_ymc);
      int min_proxy_level = max_proxy_level - (tv_cap_top_n_phys_levels - 1);
      int cell_proxy_level = tvcap_proxy_level(x_val, y_val,
                                               mesh_xmin, mesh_xmax, mesh_xma, mesh_xmb, mesh_xmc,
                                               mesh_ymin, mesh_ymax, mesh_ymy0, mesh_yma, mesh_ymb, mesh_ymc);
      if (min_proxy_level < 0) min_proxy_level = 0;
      if (cell_proxy_level < min_proxy_level) goto skip_terminal_velocity_cap;

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
#ifdef ADIABATIC
          // Regular regime: use local pressure gradient (Athena-style)
          // vrel_cap = factor * tau_s * |grad P| / rho_g
          // gamma is passed as kernel argument

          // Compute neighbor indices for pressure gradient
          int ll_ip = mem + 1;
          int ll_im = mem - 1;
          int ll_jp = mem + pitch;
          int ll_jm = mem - pitch;
#ifdef Z
          int ll_kp = mem + pitch * size_y;
          int ll_km = mem - pitch * size_y;
#endif

          int i_plus = (i < size_x - 1) ? 1 : 0;
          int i_minus = (i > 0) ? 1 : 0;
          int j_plus = (j < size_y - 1) ? 1 : 0;
          int j_minus = (j > 0) ? 1 : 0;
#ifdef Z
          int k_plus = (k < size_z - 1) ? 1 : 0;
          int k_minus = (k > 0) ? 1 : 0;
#endif

          real P_center = (gamma - (real)1.0) * gas_energy[mem];
          real P_ip = (i_plus) ? (gamma - (real)1.0) * gas_energy[ll_ip] : P_center;
          real P_im = (i_minus) ? (gamma - (real)1.0) * gas_energy[ll_im] : P_center;
          real P_jp = (j_plus) ? (gamma - (real)1.0) * gas_energy[ll_jp] : P_center;
          real P_jm = (j_minus) ? (gamma - (real)1.0) * gas_energy[ll_jm] : P_center;
#ifdef Z
          real P_kp = (k_plus) ? (gamma - (real)1.0) * gas_energy[ll_kp] : P_center;
          real P_km = (k_minus) ? (gamma - (real)1.0) * gas_energy[ll_km] : P_center;
#endif

          real dy = ymin[j+1] - ymin[j];
          real inv_dy = (dy > (real)1e-12) ? ((real)0.5 / dy) : (real)0.0;
#ifdef Z
          real dz = zmin[k+1] - zmin[k];
          real inv_dz_theta = (dz > (real)1e-12) ? ((real)0.5 / (r_sph * dz)) : (real)0.0;
#endif
          real dx = xmin[i+1] - xmin[i];
          real inv_dx = (dx > (real)1e-12) ? ((real)0.5 / (r_sph * sin(theta) * dx)) : (real)0.0;

          real grad_P_r = (P_jp - P_jm) * inv_dy;
          real grad_P_phi = (P_ip - P_im) * inv_dx;
#ifdef Z
          real grad_P_theta = (P_kp - P_km) * inv_dz_theta;
#endif

          real St_val = (invstokes > (real)0.0) ? ((real)1.0 / invstokes) : (real)0.0;
          real omega_K_local = sqrt_real(gm / (R_cyl * R_cyl * R_cyl));
          real tau_s = (omega_K_local > (real)0.0) ? (St_val / omega_K_local) : (real)0.0;
          real vrel_coef = tv_cap_factor * abs_real(tau_s) / rhog;

          vrel_cap2 = vrel_coef * abs_real(grad_P_r);
          vrel_cap1 = vrel_coef * abs_real(grad_P_phi);
#ifdef Z
          vrel_cap3 = vrel_coef * abs_real(grad_P_theta);
#else
          vrel_cap3 = (real)0.0;
#endif

          real vrel_cap_profile = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
          if (vrel_cap_profile < (real)0.1 * cs_ana) {
            vrel_cap_profile = tv_cap_factor * cs_ana;
          }

          real vrel_cap_max = vrel_cap1;
          if (vrel_cap2 > vrel_cap_max) vrel_cap_max = vrel_cap2;
          if (vrel_cap3 > vrel_cap_max) vrel_cap_max = vrel_cap3;

          if (vrel_cap_max > (real)10.0 * vrel_cap_profile || vrel_cap_max < (real)1e-10) {
            vrel_cap1 = vrel_cap_profile;
            vrel_cap2 = vrel_cap_profile;
            vrel_cap3 = vrel_cap_profile;
          }
#else
          real vrel_cap = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
          if (vrel_cap < (real)0.1 * cs_ana) {
            vrel_cap = tv_cap_factor * cs_ana;
          }
          vrel_cap1 = vrel_cap;
          vrel_cap2 = vrel_cap;
          vrel_cap3 = vrel_cap;
#endif
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
skip_terminal_velocity_cap:

    // ------------------------------------------------------------------
    // 3B3. Near-planet free-fall velocity cap (Athena++ style)
    // ------------------------------------------------------------------
    if (v_cap_flag == 1 && fluidtype != GAS && planet_mass > (real)0.0) {
      // Compute distance to planet in spherical coordinates
      // FARGO uses spherical: Y=radius, Z=colatitude
      real r_cell = y_val;  // r_sph
#ifdef Z
      real theta_cell = z_val;
      // Use actual phi from xmin array (cell-centered coordinate)
      real phi_cell = (real)0.5 * (xmin[i] + xmin[i+1]);  // xmed equivalent
      // Convert to Cartesian for distance calculation
      real x_cell = r_cell * sin(theta_cell) * cos(phi_cell);
      real y_cell = r_cell * sin(theta_cell) * sin(phi_cell);
      real z_cell = r_cell * cos(theta_cell);
#else
      real x_cell = r_cell;
      real y_cell = (real)0.0;
      real z_cell = (real)0.0;
#endif

      // Distance to planet
      real dx = x_cell - planet_x;
      real dy = y_cell - planet_y;
      real dz = z_cell - planet_z;
      real dist2 = dx*dx + dy*dy + dz*dz;

      // Cap radius: v_cap_radius_ratio * Hill radius
      // Hill radius: r_H = r_p * (m_p / 3*m_star)^(1/3)
      // Use planet_radius argument passed from Floor_gpu()
      real hill_radius = planet_radius * pow(planet_mass / ((real)3.0 * MSTAR), (real)(1.0/3.0));
      real cap_radius = v_cap_radius_ratio * hill_radius;
      real cap_radius2 = cap_radius * cap_radius;

      if (dist2 < cap_radius2) {
        // Softened distance for free-fall velocity
        real soft_dist = sqrt_real(dist2 + planet_soft * planet_soft);

        // Free-fall velocity: v_ff = sqrt(2 * G * m_planet / r)
        // Note: planet_mass is the actual mass, gm is stellar GM
        // Athena uses planet_gm directly; here we use G * planet_mass
        real planet_gm = (real)G * planet_mass;
        real v_ff = sqrt_real((real)2.0 * planet_gm / soft_dist);

        // Cap velocity
        real v_cap = v_cap_factor * v_ff;
        real v_cap2 = v_cap * v_cap;

        // Dust total velocity squared
        real dust_vel2 = v_x*v_x + v_y*v_y + v_z*v_z;

        if (dust_vel2 > v_cap2) {
          // Uniform damping to preserve direction (Athena-style)
          real damping = v_cap / sqrt_real(dust_vel2);
          v_x = v_x * damping;
          v_y = v_y * damping;
#ifdef Z
          if (vz != NULL) v_z = v_z * damping;
#endif
          // Note: Diffusion term scaling is not implemented in FARGO3D floor.cu
          // This would require access to dust diffusion momentum arrays
        }
      }
    }

    // ------------------------------------------------------------------
    // 3B4. Existing near-floor repair logic (keep for compatibility)
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

  // ------------------------------------------------------------------
  // 3B5. Anomaly check (Athena++ style) - only for dust, near planet
  // ------------------------------------------------------------------
  if (anomaly_check_flag == 1 && fluidtype != GAS && planet_mass > (real)0.0) {
    // Re-declare gas velocities for this block
    real vg_x = gas_vx[mem];
    real vg_y = gas_vy[mem];
#ifdef Z
    real vg_z = (gas_vz != NULL) ? gas_vz[mem] : (real)0.0;
#else
    real vg_z = (real)0.0;
#endif

    // Check if near planet (within r_ratio * r_s or hill_radius)
    real r_cell = y_val;
    // Compute Cartesian position for distance check
#ifdef Z
    real theta_cell = z_val;
    real phi_cell = (real)0.5 * (xmin[i] + xmin[i+1]);
    real x_cell = r_cell * sin(theta_cell) * cos(phi_cell);
    real y_cell = r_cell * sin(theta_cell) * sin(phi_cell);
    real z_cell = r_cell * cos(theta_cell);
#else
    real x_cell = r_cell;
    real y_cell = (real)0.0;
    real z_cell = (real)0.0;
#endif

    real dx = x_cell - planet_x;
    real dy = y_cell - planet_y;
    real dz = z_cell - planet_z;
    real dist2 = dx*dx + dy*dy + dz*dz;

    real cap_r_sq = anomaly_check_r_ratio * r_cell * r_cell;
    real hill_radius = planet_radius * pow(planet_mass / ((real)3.0 * MSTAR), (real)(1.0/3.0));
    real cap_hill_sq = anomaly_check_hill_ratio * hill_radius * hill_radius;
    if (cap_r_sq > cap_hill_sq) cap_r_sq = cap_hill_sq;

    if (dist2 < cap_r_sq) {
      // Re-declare gas velocities for this block
      real vg_x = gas_vx[mem];
      real vg_y = gas_vy[mem];
#ifdef Z
      real vg_z = (gas_vz != NULL) ? gas_vz[mem] : (real)0.0;
#else
      real vg_z = (real)0.0;
#endif

      // Compute |v_d|^2 and |v_g|^2
      real dust_vel2 = v_x*v_x + v_y*v_y + v_z*v_z;
      real gas_vel2 = vg_x*vg_x + vg_y*vg_y + vg_z*vg_z;
      real gas_vel2_safe = (gas_vel2 > (real)1e-24) ? gas_vel2 : (real)1e-24;

      // Check if |v_d|^2 / |v_g|^2 > 20
      if (dust_vel2 > (real)20.0 * gas_vel2_safe) {
        // 26-neighbor check
        real sum_dust = (real)0.0;
        real sum_gas = (real)0.0;
        int n_neighbors = 0;
        int pitch_y = pitch;
        int stride_z = pitch * size_y;

        // 3x3x3 neighborhood (26 cells, excluding center)
        for (int dk = -1; dk <= 1; dk++) {
          for (int dj = -1; dj <= 1; dj++) {
            for (int di = -1; di <= 1; di++) {
              if (di == 0 && dj == 0 && dk == 0) continue;

              int ni = i + di;
              int nj = j + dj;
              int nk = k + dk;

              if (ni < 0 || ni >= size_x || nj < 0 || nj >= size_y || nk < 1 || nk >= size_z-1) continue;

              int nll = ni + pitch_y * (nj + nk * size_y);
              real n_dust_vx = vx[nll];
              real n_dust_vy = vy[nll];
              real n_dust_vz = (vz != NULL) ? vz[nll] : (real)0.0;
              real n_dust_vel2 = n_dust_vx*n_dust_vx + n_dust_vy*n_dust_vy + n_dust_vz*n_dust_vz;

              real n_gas_vx = gas_vx[nll];
              real n_gas_vy = gas_vy[nll];
              real n_gas_vz = (gas_vz != NULL) ? gas_vz[nll] : (real)0.0;
              real n_gas_vel2 = n_gas_vx*n_gas_vx + n_gas_vy*n_gas_vy + n_gas_vz*n_gas_vz;

              sum_dust += n_dust_vel2;
              sum_gas += n_gas_vel2;
              n_neighbors++;
            }
          }
        }

        if (n_neighbors > 0) {
          real mean_dust = sum_dust / n_neighbors;
          real mean_gas = sum_gas / n_neighbors;

          // Compare with mean
          if (mean_dust > (real)1e-20 && dust_vel2 / mean_dust > (real)20.0) {
            // Damping factor = mean / current
            real damping = sqrt_real(mean_dust / dust_vel2);
            v_x = vg_x + damping * (v_x - vg_x);
            v_y = vg_y + damping * (v_y - vg_y);
#ifdef Z
            if (vz != NULL) v_z = vg_z + damping * (v_z - vg_z);
#endif
          }
          // Also check gas anomaly
          else if (mean_gas > (real)1e-20 && gas_vel2 / mean_gas > (real)20.0) {
            real damping = sqrt_real(mean_gas / gas_vel2);
            v_x = vg_x + damping * (v_x - vg_x);
            v_y = vg_y + damping * (v_y - vg_y);
#ifdef Z
            if (vz != NULL) v_z = vg_z + damping * (v_z - vg_z);
#endif
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // 3B6. Relative median cap (Athena++ style) - 6 neighbor check
  // ------------------------------------------------------------------
  if (0 && relmedian_cap_flag == 1 && fluidtype != GAS) {
    // Re-declare gas velocities for this block
    real vg_x = gas_vx[mem];
    real vg_y = gas_vy[mem];
#ifdef Z
    real vg_z = (gas_vz != NULL) ? gas_vz[mem] : (real)0.0;
#else
    real vg_z = (real)0.0;
#endif

    // 6-neighbor: +/- i, +/- j, +/- k
    real neighbor_vel[6];
    int n_idx = 0;
    int pitch_y = pitch;
    int stride_z = pitch * size_y;

    // Get neighbors in r, phi, theta directions
    int ll_ip = mem + 1;
    int ll_im = mem - 1;
    int ll_jp = mem + pitch_y;
    int ll_jm = mem - pitch_y;
#ifdef Z
    int ll_kp = mem + stride_z;
    int ll_km = mem - stride_z;
#endif

    if (i < size_x - 1) neighbor_vel[n_idx++] = vx[ll_ip]*vx[ll_ip] + vy[ll_ip]*vy[ll_ip] + ((vz != NULL) ? vz[ll_ip]*vz[ll_ip] : (real)0.0);
    if (i > 0) neighbor_vel[n_idx++] = vx[ll_im]*vx[ll_im] + vy[ll_im]*vy[ll_im] + ((vz != NULL) ? vz[ll_im]*vz[ll_im] : (real)0.0);
    if (j < size_y - 1) neighbor_vel[n_idx++] = vx[ll_jp]*vx[ll_jp] + vy[ll_jp]*vy[ll_jp] + ((vz != NULL) ? vz[ll_jp]*vz[ll_jp] : (real)0.0);
    if (j > 0) neighbor_vel[n_idx++] = vx[ll_jm]*vx[ll_jm] + vy[ll_jm]*vy[ll_jm] + ((vz != NULL) ? vz[ll_jm]*vz[ll_jm] : (real)0.0);
#ifdef Z
    if (k < size_z - 1 && vz != NULL) neighbor_vel[n_idx++] = vx[ll_kp]*vx[ll_kp] + vy[ll_kp]*vy[ll_kp] + vz[ll_kp]*vz[ll_kp];
    if (k > 1 && vz != NULL) neighbor_vel[n_idx++] = vx[ll_km]*vx[ll_km] + vy[ll_km]*vy[ll_km] + vz[ll_km]*vz[ll_km];
#endif

    // Simple median (sort 6 elements - bubble sort for small array)
    if (n_idx >= 3) {
      for (int mi = 0; mi < n_idx-1; mi++) {
        for (int mj = mi+1; mj < n_idx; mj++) {
          if (neighbor_vel[mi] > neighbor_vel[mj]) {
            real temp = neighbor_vel[mi];
            neighbor_vel[mi] = neighbor_vel[mj];
            neighbor_vel[mj] = temp;
          }
        }
      }
      real median_vel2 = neighbor_vel[n_idx/2];

      // If |v_d|^2 > factor^2 * median, damp
      real dust_vel2 = v_x*v_x + v_y*v_y + v_z*v_z;
      real median_cap_sq = relmedian_cap_factor * relmedian_cap_factor * median_vel2;

      if (dust_vel2 > median_cap_sq && median_vel2 > (real)1e-20) {
        real damping = sqrt_real(median_cap_sq / dust_vel2);
        v_x = vg_x + damping * (v_x - vg_x);
        v_y = vg_y + damping * (v_y - vg_y);
#ifdef Z
        if (vz != NULL) v_z = vg_z + damping * (v_z - vg_z);
#endif
      }
    }
  }

  // Write updated velocities after anomaly/median cap
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
#ifdef ADIABATIC
  real *d_gas_energy = Fluids[0]->Energy->field_gpu;
#else
  real *d_gas_energy = NULL;
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
  const int  tv_cap_top_n_phys_levels = (int)DUST_TVCAP_TOP_N_PHYS_LEVELS;
  const int  relmedian_cap_flag = (int)RELATIVE_MEDIAN_CAP_FLAG;
  const real relmedian_cap_factor = (real)RELATIVE_MEDIAN_CAP_FACTOR;
  const int  relmedian_cap_max_iters = (int)RELATIVE_MEDIAN_CAP_MAX_ITERS;
  const real relmedian_cap_rho_floor_factor = (real)RELATIVE_MEDIAN_CAP_RHO_FLOOR_FACTOR;
  const real mesh_xmin = (real)XMIN;
  const real mesh_xmax = (real)XMAX;
  const real mesh_xma  = (real)XMA;
  const real mesh_xmb  = (real)XMB;
  const real mesh_xmc  = (real)XMC;
  const real mesh_ymin = (real)YMIN;
  const real mesh_ymax = (real)YMAX;
  const real mesh_ymy0 = (real)YMY0;
  const real mesh_yma  = (real)YMA;
  const real mesh_ymb  = (real)YMB;
  const real mesh_ymc  = (real)YMC;

  // Near-planet cap parameters
  const int  v_cap_flag = (int)DUST_VCAP_FLAG;
  const real v_cap_factor = (real)DUST_VCAP_FACTOR;
  const real v_cap_radius_ratio = (real)DUST_VCAP_RADIUS_RATIO;

  // Planet parameters (from global FARGO3D variables)
  const real planet_x = (real)Xplanet;
  const real planet_y = (real)Yplanet;
  const real planet_z = (real)Zplanet;
  const real planet_mass = (real)MplanetVirtual;
  const real planet_soft = (real)ROCHESMOOTHING;
  // Planet orbital radius (distance from star)
  const real planet_radius = sqrt(planet_x*planet_x + planet_y*planet_y + planet_z*planet_z);

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
      Ymin_d, d_zmin, Xmin_d,
      d_gas_dens, d_gas_vx, d_gas_vy, d_gas_vz,
      d_gas_energy,
      sx, sy, sz, Pitch_gpu,
      Fluidtype,
      gas_abs_floor, dust_abs_floor, energy_floor,
      dust_percent_floor, tv_cap_flag, tv_cap_factor, tv_cap_d2g_max,
      tv_cap_rho_floor_factor, tv_cap_low_gas_floor_factor,
      tv_cap_vratio_override, tv_cap_mach_override, tv_cap_z_hg_ratio, tv_cap_top_n_phys_levels,
      mesh_xmin, mesh_xmax, mesh_xma, mesh_xmb, mesh_xmc,
      mesh_ymin, mesh_ymax, mesh_ymy0, mesh_yma, mesh_ymb, mesh_ymc,
      v_cap_flag, v_cap_factor, v_cap_radius_ratio,
      planet_x, planet_y, planet_z, planet_mass, planet_soft,
      planet_radius,
      aspectratio, flaringindex, r0,
      gm, invstokes, epsilon,
#ifdef ADIABATIC
      (real)GAMMA,
#endif
      // Anomaly check parameters
      (int)ANOMALY_CHECK_FLAG, (real)ANOMALY_CHECK_R_RATIO, (real)ANOMALY_CHECK_HILL_RATIO,
      // Relative median cap parameters
      relmedian_cap_flag, relmedian_cap_factor
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

  if (Fluidtype != GAS && relmedian_cap_flag == 1) {
    ApplyRelmedianIterativeGpu(d_dens, d_vx, d_vy, d_vz,
                               sx, sy, sz, Pitch_gpu,
                               relmedian_cap_max_iters,
                               relmedian_cap_factor,
                               relmedian_cap_rho_floor_factor);
  }
}
