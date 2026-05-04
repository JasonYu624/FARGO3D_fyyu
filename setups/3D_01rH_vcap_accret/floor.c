//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//<\INCLUDES>

static inline real clamp_real(real x, real xmin, real xmax) {
  return (x < xmin) ? xmin : ((x > xmax) ? xmax : x);
}

static inline real abs_real(real x) {
  return (x < (real)0.0) ? -x : x;
}

static inline real max_real(real a, real b) {
  return (a > b) ? a : b;
}

static inline real sqrt_real(real x) {
  return (x > (real)0.0) ? sqrt(x) : (real)0.0;
}

static inline int round_to_int(real x) {
  return (int)floor(x + (real)0.5);
}

static inline real psi_x_local(real x) {
  real x_mesh_i = (XMAX - XMIN) + XMC * (XMA + XMB);
  real ax = abs_real(x);
  if (ax <= (real)XMA) return ((real)1.0 + (real)XMC) / x_mesh_i;
  if (ax < (real)XMB) {
    real arg = (real)M_PI * (ax - (real)XMA) / ((real)2.0 * ((real)XMB - (real)XMA));
    real c = cos(arg);
    return ((real)1.0 + (real)XMC * c * c) / x_mesh_i;
  }
  return (real)1.0 / x_mesh_i;
}

static inline real psi_y_local(real y) {
  real y_mesh_i = log(YMAX/YMIN) + YMC * (YMA + YMB);
  real ay = abs_real(y - (real)YMY0);
  if (ay <= (real)YMA) return ((real)1.0 / y + (real)YMC) / y_mesh_i;
  if (ay < (real)YMB) {
    real arg = (real)M_PI * (ay - (real)YMA) / ((real)2.0 * ((real)YMB - (real)YMA));
    real c = cos(arg);
    return ((real)1.0 / y + (real)YMC * c * c) / y_mesh_i;
  }
  return ((real)1.0 / y) / y_mesh_i;
}

static inline int tvcap_proxy_max_level(void) {
  real refine_x = (real)1.0 + (real)XMC;
  real refine_y = (real)1.0 + (real)YMC;
  real refine_max = (refine_x > refine_y) ? refine_x : refine_y;
  if (refine_max < (real)1.0) refine_max = (real)1.0;
  return round_to_int(log(refine_max)/log((real)2.0));
}

static inline int tvcap_proxy_level(real phi, real rad) {
  real refine_x = psi_x_local(phi) * (XMAX - XMIN);
  real refine_y = psi_y_local(rad) * rad * log(YMAX/YMIN);
  real refine_local = (refine_x > refine_y) ? refine_x : refine_y;
  if (refine_local < (real)1.0) return 0;
  return round_to_int(log(refine_local)/log((real)2.0));
}

static inline real median6_real(real values[6]) {
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

static void EnsureRelmedianCpuBuffers(size_t total,
                                      real **src_dens,
                                      real **src_vx,
                                      real **src_vy,
                                      real **src_vz,
                                      size_t *capacity) {
  if (total <= *capacity) return;

  real *new_dens = (real *)realloc(*src_dens, total * sizeof(real));
  real *new_vx   = (real *)realloc(*src_vx,   total * sizeof(real));
  real *new_vy   = (real *)realloc(*src_vy,   total * sizeof(real));
  real *new_vz   = (real *)realloc(*src_vz,   total * sizeof(real));

  if (new_dens == NULL || new_vx == NULL || new_vy == NULL || new_vz == NULL) {
    mastererr("Out of memory in iterative relmedian CPU buffers.\n");
    prs_exit(EXIT_FAILURE);
  }

  *src_dens = new_dens;
  *src_vx   = new_vx;
  *src_vy   = new_vy;
  *src_vz   = new_vz;
  *capacity = total;
}

static void ApplyRelmedianIterativeCpu(real *dens,
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
  static size_t capacity = 0;
  const size_t total = (size_t)pitch * (size_t)size_y * (size_t)size_z;
  const int stride = pitch * size_y;

  if (max_iters < 1 || cap_factor <= (real)0.0) return;

  EnsureRelmedianCpuBuffers(total, &src_dens, &src_vx, &src_vy, &src_vz, &capacity);

  for (int iter = 0; iter < max_iters; ++iter) {
    int changed = 0;
    memcpy(src_dens, dens, total * sizeof(real));
    memcpy(src_vx, vx, total * sizeof(real));
    memcpy(src_vy, vy, total * sizeof(real));
    if (vz != NULL) memcpy(src_vz, vz, total * sizeof(real));

    for (int k = 1; k < size_z - 1; ++k) {
      for (int j = 1; j < size_y - 1; ++j) {
        for (int i = 1; i < size_x - 1; ++i) {
          const int ll = i + pitch * (j + k * size_y);
          real neighbor_vel[6];
          real neighbor_rho[6];
          real dust_vel2;
          real median_vel2;
          real median_cap_sq;

          neighbor_vel[0] = src_vx[ll + 1]      * src_vx[ll + 1]      + src_vy[ll + 1]      * src_vy[ll + 1]      + ((vz != NULL) ? src_vz[ll + 1]      * src_vz[ll + 1]      : (real)0.0);
          neighbor_vel[1] = src_vx[ll - 1]      * src_vx[ll - 1]      + src_vy[ll - 1]      * src_vy[ll - 1]      + ((vz != NULL) ? src_vz[ll - 1]      * src_vz[ll - 1]      : (real)0.0);
          neighbor_vel[2] = src_vx[ll + pitch]  * src_vx[ll + pitch]  + src_vy[ll + pitch]  * src_vy[ll + pitch]  + ((vz != NULL) ? src_vz[ll + pitch]  * src_vz[ll + pitch]  : (real)0.0);
          neighbor_vel[3] = src_vx[ll - pitch]  * src_vx[ll - pitch]  + src_vy[ll - pitch]  * src_vy[ll - pitch]  + ((vz != NULL) ? src_vz[ll - pitch]  * src_vz[ll - pitch]  : (real)0.0);
          neighbor_vel[4] = src_vx[ll + stride] * src_vx[ll + stride] + src_vy[ll + stride] * src_vy[ll + stride] + ((vz != NULL) ? src_vz[ll + stride] * src_vz[ll + stride] : (real)0.0);
          neighbor_vel[5] = src_vx[ll - stride] * src_vx[ll - stride] + src_vy[ll - stride] * src_vy[ll - stride] + ((vz != NULL) ? src_vz[ll - stride] * src_vz[ll - stride] : (real)0.0);

          median_vel2 = median6_real(neighbor_vel);
          dust_vel2 = src_vx[ll] * src_vx[ll] + src_vy[ll] * src_vy[ll] +
                      ((vz != NULL) ? src_vz[ll] * src_vz[ll] : (real)0.0);
          median_cap_sq = cap_factor * cap_factor * median_vel2;

          if (median_vel2 <= (real)1.0e-20 || dust_vel2 <= median_cap_sq) continue;

          {
            const real damping = sqrt_real(median_cap_sq / dust_vel2);
            vx[ll] = src_vx[ll] * damping;
            vy[ll] = src_vy[ll] * damping;
            if (vz != NULL) vz[ll] = src_vz[ll] * damping;
          }
          changed = 1;

          if (rho_floor_factor > (real)0.0) {
            neighbor_rho[0] = src_dens[ll + 1];
            neighbor_rho[1] = src_dens[ll - 1];
            neighbor_rho[2] = src_dens[ll + pitch];
            neighbor_rho[3] = src_dens[ll - pitch];
            neighbor_rho[4] = src_dens[ll + stride];
            neighbor_rho[5] = src_dens[ll - stride];
            {
              const real rho_floor_local = rho_floor_factor * median6_real(neighbor_rho);
              if (dens[ll] < rho_floor_local) dens[ll] = rho_floor_local;
            }
          }
        }
      }
    }

    if (!changed) break;
  }
}

// ----------------------------------------------------------------------
// CPU Kernel: Apply minimally invasive floors / limiters.
// Same logic as floor.cu GPU version, adapted for CPU.
// ----------------------------------------------------------------------
void Floor_cpu() {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Vx);
  INPUT(Vy);
#ifdef Z
  INPUT(Vz);
#endif
#ifdef ADIABATIC
  INPUT(Energy);
#endif
#ifdef ADIABATIC
  // Gas fields for TV-cap with local pressure gradient
  INPUT(Fluids[0]->Energy);
#endif
  // Gas fields (for dust percent floor and TV-cap)
  INPUT(Fluids[0]->Density);
  INPUT(Fluids[0]->Vx);
  INPUT(Fluids[0]->Vy);
#ifdef Z
  INPUT(Fluids[0]->Vz);
#endif
  OUTPUT(Density);
  OUTPUT(Vx);
  OUTPUT(Vy);
#ifdef Z
  OUTPUT(Vz);
#endif
#ifdef ADIABATIC
  OUTPUT(Energy);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* dens = Density->field_cpu;
  real* vx   = Vx->field_cpu;
  real* vy   = Vy->field_cpu;
#ifdef Z
  real* vz   = Vz->field_cpu;
#endif
#ifdef ADIABATIC
  real* energy = Energy->field_cpu;
  // Gas energy for local pressure gradient TV-cap
  real* gas_energy = Fluids[0]->Energy->field_cpu;
#endif
  real* gas_dens = Fluids[0]->Density->field_cpu;
  real* gas_vx   = Fluids[0]->Vx->field_cpu;
  real* gas_vy   = Fluids[0]->Vy->field_cpu;
#ifdef Z
  real* gas_vz   = Fluids[0]->Vz->field_cpu;
#endif

  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;

  // Get parameters from .par file
  const real gas_abs_floor = (real)GASFLOOR;
  const real energy_floor  = (real)ENERGYFLOOR;
  const real dust_abs_floor = (real)DUSTFLOOR;
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

  // Near-planet cap parameters
  const int  v_cap_flag = (int)DUST_VCAP_FLAG;
  const real v_cap_factor = (real)DUST_VCAP_FACTOR;
  const real v_cap_radius_ratio = (real)DUST_VCAP_RADIUS_RATIO;

  // Anomaly check parameters
  const int  anomaly_check_flag = (int)ANOMALY_CHECK_FLAG;
  const real anomaly_check_r_ratio = (real)ANOMALY_CHECK_R_RATIO;
  const real anomaly_check_hill_ratio = (real)ANOMALY_CHECK_HILL_RATIO;

  // Relative median cap parameters
  const int  relmedian_cap_flag = (int)RELATIVE_MEDIAN_CAP_FLAG;
  const real relmedian_cap_factor = (real)RELATIVE_MEDIAN_CAP_FACTOR;
  const int  relmedian_cap_max_iters = (int)RELATIVE_MEDIAN_CAP_MAX_ITERS;
  const real relmedian_cap_rho_floor_factor = (real)RELATIVE_MEDIAN_CAP_RHO_FLOOR_FACTOR;

  // Planet parameters
  const real planet_x = (real)Xplanet;
  const real planet_y = (real)Yplanet;
  const real planet_z = (real)Zplanet;
  const real planet_mass = (real)MplanetVirtual;
  const real planet_soft = (real)ROCHESMOOTHING;
  // Planet orbital radius (distance from star)
  const real planet_radius = sqrt(planet_x*planet_x + planet_y*planet_y + planet_z*planet_z);

  const real aspectratio  = ASPECTRATIO;
  const real flaringindex = FLARINGINDEX;
  const real r0           = R0;
  const real gm           = G * MSTAR;
  const real invstokes    = INVSTOKES1;
  const real epsilon      = EPSILON;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int l_ym, l_ym_p1;
#ifdef Z
  int l_zm, l_zm_p1;
#endif
//<\INTERNAL>

//<CONSTANT>
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=1; k<size_z-1; k++) {
#endif
#ifdef Y
    for (j=1; j<size_y-1; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
        ll = l;

        // Determine if this is gas or dust
        int is_gas = (Fluidtype == GAS);

        // ------------------------------------------------------------------
        // 1. Density floor.
        // ------------------------------------------------------------------
        real limit_floor = is_gas ? gas_abs_floor : dust_abs_floor;
        real d = dens[ll];
        if (!(d >= limit_floor)) dens[ll] = limit_floor;

        // ------------------------------------------------------------------
        // 2. NaN / Inf intercept for velocities.
        // ------------------------------------------------------------------
        real v_x = vx[ll];
        if (!(v_x >= (real)-1.0e30 && v_x <= (real)1.0e30)) v_x = (real)0.0;

        real v_y = vy[ll];
        if (!(v_y >= (real)-1.0e30 && v_y <= (real)1.0e30)) v_y = (real)0.0;

#ifdef Z
        real v_z = (real)0.0;
        if (vz != NULL) {
          v_z = vz[ll];
          if (!(v_z >= (real)-1.0e30 && v_z <= (real)1.0e30)) v_z = (real)0.0;
        }
#endif

        // Geometry helpers.
        real x_val = xmed(i);
        real y_val = ymed(j);
#ifdef Z
        real z_val = zmed(k);
        real r_sph = y_val;
        real theta = z_val;
        real R_cyl = r_sph * sin(z_val);
#else
        real R_cyl = y_val;
#endif
        if (R_cyl < (real)1.0e-6) R_cyl = (real)1.0e-6;

        real vK = sqrt(gm / R_cyl);
        real h  = aspectratio * pow(R_cyl / r0, flaringindex);
        real cs_ana = h * vK;

        // ------------------------------------------------------------------
        // 3A. Gas: only limit very low-density, high-altitude v_theta.
        // ------------------------------------------------------------------
#ifdef Z
#ifdef SPHERICAL
        if (is_gas && vz != NULL) {
          const real NH_CEIL       = (real)3.5;
          const real MACH_CEIL     = (real)1.0;
          const real LOW_DENS_MULT = (real)150.0;

          if (dens[ll] < LOW_DENS_MULT * gas_abs_floor) {
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
        // ------------------------------------------------------------------
        if (!is_gas) {
          real rhod = dens[ll];
          real rhog = gas_dens[ll];
          if (!(rhog > (real)0.0)) rhog = gas_abs_floor;

          real vg_x = gas_vx[ll];
          real vg_y = gas_vy[ll];
#ifdef Z
          real vg_z = (gas_vz != NULL) ? gas_vz[ll] : (real)0.0;
#else
          real vg_z = (real)0.0;
#endif

          real dust_to_gas = rhod / rhog;

          // ------------------------------------------------------------------
          // 3B1. Dust percent floor
          // ------------------------------------------------------------------
          if (dust_percent_floor > (real)0.0) {
#ifdef Z
            real z_cyl = r_sph * cos(theta);
#else
            real z_cyl = (real)0.0;
#endif

            real H_over_R = aspectratio * pow(R_cyl / r0, flaringindex);
            real rho_g_init = pow(R_cyl / r0, -2.25) *
                              exp((R_cyl / (aspectratio * aspectratio * pow(R_cyl / r0, 2.0 * flaringindex))) *
                                  (1.0 / r_sph - 1.0 / R_cyl));
            real rho_d_init = epsilon * rho_g_init;

            real min_dust = dust_percent_floor * rho_d_init;
            if (rhod < min_dust) {
              real old_rhod = rhod;
              rhod = (real)1.7782794100389227 * min_dust;
              dens[ll] = rhod;

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
            int max_proxy_level = tvcap_proxy_max_level();
            int min_proxy_level = max_proxy_level - (tv_cap_top_n_phys_levels - 1);
            int cell_proxy_level = tvcap_proxy_level(x_val, y_val);
            if (min_proxy_level < 0) min_proxy_level = 0;
            if (cell_proxy_level < min_proxy_level) goto skip_terminal_velocity_cap;

            real dv_x = v_x - vg_x;
            real dv_y = v_y - vg_y;
#ifdef Z
            real dv_z = v_z - vg_z;
            real z_cyl = r_sph * cos(theta);
#else
            real dv_z = (real)0.0;
            real z_cyl = (real)0.0;
#endif

            real dust_vel_abs2 = dv_x*dv_x + dv_y*dv_y + dv_z*dv_z;
            real gas_vel_abs2 = vg_x*vg_x + vg_y*vg_y + vg_z*vg_z;
            real gas_vel_abs2_safe = (gas_vel_abs2 > (real)1.0e-24) ? gas_vel_abs2 : (real)1.0e-24;
            real gas_cs2 = cs_ana * cs_ana;

            int apply_cap = 0;
            int force_override = 0;
            int force_low_gas_cap = 0;

            if (dust_vel_abs2 > tv_cap_vratio_override * tv_cap_vratio_override * gas_vel_abs2_safe) {
              force_override = 1;
            }
            if (tv_cap_mach_override > (real)0.0 &&
                dust_vel_abs2 > tv_cap_mach_override * tv_cap_mach_override * gas_cs2) {
              force_override = 1;
            }
#ifdef Z
            if (dust_vel_abs2 > (real)20.0 * gas_vel_abs2_safe) {
              real omega_dyn = sqrt(gm / (R_cyl * R_cyl * R_cyl));
              real hg = cs_ana / omega_dyn;
              if (abs_real(z_cyl) > tv_cap_z_hg_ratio * hg) {
                force_override = 1;
                force_low_gas_cap = 1;
              }
            }
#endif

            int d2g_gated = (rhod <= tv_cap_rho_floor_factor * dust_abs_floor && dust_to_gas < tv_cap_d2g_max);
            if (!d2g_gated && force_override == 0) {
              // skip
            } else {
              if (dust_vel_abs2 > (real)4.0 * gas_vel_abs2_safe) {
                apply_cap = 1;
              }
            }

            if (apply_cap) {
              int use_low_gas_cap = (rhog < tv_cap_low_gas_floor_factor * gas_abs_floor || force_low_gas_cap == 1);

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
                real vrel_r = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
                real vrel_phi = (real)0.0;
                real vrel_theta = tv_cap_factor * abs_real(St2 / ((real)1.0 + St2)) * eta * vK;

                vrel_cap1 = vrel_r;
                vrel_cap2 = vrel_phi;
                vrel_cap3 = vrel_theta;
              } else {
                // Athena-style local pressure gradient TV-cap
                // vrel_cap = factor * tau_s * |grad P| / rho_g
#ifdef ADIABATIC
                real gamma = GAMMA;

                // Get neighbor indices for pressure gradient
                int ll_ip = ll + 1;        // i+1
                int ll_im = ll - 1;        // i-1
                int ll_jp = ll + pitch;    // j+1
                int ll_jm = ll - pitch;    // j-1
                int ll_kp = ll + stride;   // k+1
                int ll_km = ll - stride;   // k-1

                // Compute pressure P = (gamma-1) * E
                real P_center = (gamma - (real)1.0) * gas_energy[ll];
                real P_ip = (ll_ip >= 0 && ll_ip < size_x*size_y*size_z) ?
                           (gamma - (real)1.0) * gas_energy[ll_ip] : P_center;
                real P_im = (ll_im >= 0) ?
                           (gamma - (real)1.0) * gas_energy[ll_im] : P_center;
                real P_jp = (ll_jp < size_x*size_y*size_z) ?
                           (gamma - (real)1.0) * gas_energy[ll_jp] : P_center;
                real P_jm = (ll_jm >= 0) ?
                           (gamma - (real)1.0) * gas_energy[ll_jm] : P_center;
#ifdef Z
                real P_kp = (ll_kp < size_x*size_y*size_z) ?
                            (gamma - (real)1.0) * gas_energy[ll_kp] : P_center;
                real P_km = (ll_km >= 0) ?
                            (gamma - (real)1.0) * gas_energy[ll_km] : P_center;
#endif

                // Grid spacing for gradient
                real dy = ymed(j+1) - ymed(j);
                real inv_dy = (dy > (real)1e-12) ? ((real)0.5 / dy) : (real)0.0;
#ifdef Z
                real dz = zmed(k+1) - zmed(k);
                real inv_dz = (dz > (real)1e-12) ? ((real)0.5 / dz) : (real)0.0;
                // dP/dz in spherical: need r*theta component
                real inv_dz_theta = (dz > (real)1e-12) ? ((real)0.5 / (r_sph * dz)) : (real)0.0;
#endif
                // dP/dphi: need r*sin(theta) component
                real dx = xmed(i+1) - xmed(i);
                real inv_dx = (dx > (real)1e-12) ? ((real)0.5 / (r_sph * sin(theta) * dx)) : (real)0.0;

                // Pressure gradients (using centered difference)
                real grad_P_r = (P_jp - P_jm) * inv_dy;           // radial
                real grad_P_phi = (P_ip - P_im) * inv_dx;          // phi
#ifdef Z
                real grad_P_theta = (P_kp - P_km) * inv_dz_theta; // theta
#endif

                // Terminal velocity coefficient: vrel = tau_s * |grad P| / rho_g
                real St_val = (invstokes > (real)0.0) ? ((real)1.0 / invstokes) : (real)0.0;
                real vrel_coef = tv_cap_factor * abs_real(St_val) / rhog;

                // Individual direction caps
                vrel_cap2 = vrel_coef * abs_real(grad_P_r);    // r (Vy)
                vrel_cap1 = vrel_coef * abs_real(grad_P_phi); // phi (Vx)
#ifdef Z
                vrel_cap3 = vrel_coef * abs_real(grad_P_theta); // theta (Vz)
#else
                vrel_cap3 = (real)0.0;
#endif

                // Fallback if cap unreasonably large or gradient zero
                real vrel_cap_profile = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
                if (vrel_cap_profile < (real)0.1 * cs_ana) {
                  vrel_cap_profile = tv_cap_factor * cs_ana;
                }

                // Check if local gradient gave unphysical large cap
                real vrel_cap_max = vrel_cap1;
                if (vrel_cap2 > vrel_cap_max) vrel_cap_max = vrel_cap2;
                if (vrel_cap3 > vrel_cap_max) vrel_cap_max = vrel_cap3;

                if (vrel_cap_max > (real)10.0 * vrel_cap_profile || vrel_cap_max < (real)1e-10) {
                  // Fallback to profile-based
                  vrel_cap1 = vrel_cap_profile;
                  vrel_cap2 = vrel_cap_profile;
                  vrel_cap3 = vrel_cap_profile;
                }
#else
                // ISOTHERMAL: use profile-based fallback
                real vrel_cap = tv_cap_factor * abs_real((real)2.0 * St / ((real)1.0 + St2)) * eta * vK;
                if (vrel_cap < (real)0.1 * cs_ana) {
                  vrel_cap = tv_cap_factor * cs_ana;
                }
                vrel_cap1 = vrel_cap;
                vrel_cap2 = vrel_cap;
                vrel_cap3 = vrel_cap;
#endif
              }

              real rel_vel2_tot = dv_x*dv_x + dv_y*dv_y + dv_z*dv_z;
              real vrel_cap2_tot = vrel_cap1*vrel_cap1 + vrel_cap2*vrel_cap2 + vrel_cap3*vrel_cap3;

              if (rel_vel2_tot > vrel_cap2_tot) {
                real damping = sqrt_real(vrel_cap2_tot / rel_vel2_tot);
                v_x = vg_x + damping * dv_x;
                v_y = vg_y + damping * dv_y;
#ifdef Z
                if (vz != NULL) v_z = vg_z + damping * dv_z;
#endif
              }
            }
          }
skip_terminal_velocity_cap:

    // ------------------------------------------------------------------
          // 3B3. Near-planet free-fall velocity cap
          // ------------------------------------------------------------------
          if (v_cap_flag == 1 && planet_mass > (real)0.0) {
            // Cell position in spherical coordinates (r, theta, phi)
            // FARGO3D: Y=radius, Z=colatitude, X=azimuth
            real r_cell = ymed(j);
#ifdef Z
            real theta_cell = zmed(k);
            real phi_cell = xmed(i);  // Use actual phi from mesh
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

            // Hill radius: r_H = a * (Mp/3M*)^(1/3)
            // planet_radius = orbital radius (a)
            // planet_mass = actual planet mass (Mp), NOT GMp
            // MSTAR = stellar mass
            real hill_radius = planet_radius * pow(planet_mass / ((real)3.0 * MSTAR), (real)(1.0/3.0));
            real cap_radius = v_cap_radius_ratio * hill_radius;
            real cap_radius2 = cap_radius * cap_radius;

            if (dist2 < cap_radius2) {
              real soft_dist = sqrt_real(dist2 + planet_soft * planet_soft);
              real planet_gm = (real)G * planet_mass;
              real v_ff = sqrt_real((real)2.0 * planet_gm / soft_dist);

              real v_cap = v_cap_factor * v_ff;
              real v_cap2 = v_cap * v_cap;

              real dust_vel2 = v_x*v_x + v_y*v_y + v_z*v_z;

              if (dust_vel2 > v_cap2) {
                real damping = v_cap / sqrt_real(dust_vel2);
                v_x = v_x * damping;
                v_y = v_y * damping;
#ifdef Z
                if (vz != NULL) v_z = v_z * damping;
#endif
              }
            }
          }

          // ------------------------------------------------------------------
          // 3B4. Existing near-floor repair logic
          // ------------------------------------------------------------------
          real near_floor  = (rhod < (real)30.0 * dust_abs_floor) ? (real)1.0 : (real)0.0;

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
        if (anomaly_check_flag == 1 && !is_gas && planet_mass > (real)0.0) {
          // Re-declare gas velocities for this block
          real vg_x = gas_vx[ll];
          real vg_y = gas_vy[ll];
#ifdef Z
          real vg_z = (gas_vz != NULL) ? gas_vz[ll] : (real)0.0;
#else
          real vg_z = (real)0.0;
#endif

          // Check if near planet (within r_ratio * r_s or hill_radius)
          real r_cell = ymed(j);
#ifdef Z
          real theta_cell = zmed(k);
          real phi_cell = xmed(i);
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

              // 3x3x3 neighborhood (26 cells, excluding center)
              for (int dk = -1; dk <= 1; dk++) {
                for (int dj = -1; dj <= 1; dj++) {
                  for (int di = -1; di <= 1; di++) {
                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int ni = i + di;
                    int nj = j + dj;
                    int nk = k + dk;

                    if (ni < 0 || ni >= size_x || nj < 0 || nj >= size_y || nk < 1 || nk >= size_z-1) continue;

                    int nll = ni + pitch * (nj + nk * size_y);
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
        if (0 && relmedian_cap_flag == 1 && !is_gas) {
          // Re-declare gas velocities for this block
          real vg_x = gas_vx[ll];
          real vg_y = gas_vy[ll];
#ifdef Z
          real vg_z = (gas_vz != NULL) ? gas_vz[ll] : (real)0.0;
#else
          real vg_z = (real)0.0;
#endif

          // 6-neighbor: +/- i, +/- j, +/- k
          real neighbor_vel[6];
          int n_idx = 0;

          // Get neighbors in r, phi, theta directions
          int ll_ip = ll + 1;
          int ll_im = ll - 1;
          int ll_jp = ll + pitch;
          int ll_jm = ll - pitch;
#ifdef Z
          int ll_kp = ll + stride;
          int ll_km = ll - stride;
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
        vx[ll] = v_x;
        vy[ll] = v_y;
#ifdef Z
        if (vz != NULL) vz[ll] = v_z;
#endif

        // ------------------------------------------------------------------
        // 4. Energy floor.
        // ------------------------------------------------------------------
#ifdef ADIABATIC
        if (energy != NULL) {
          if (is_gas) {
            real e = energy[ll];
            if (!(e >= energy_floor)) energy[ll] = energy_floor;
          } else {
            energy[ll] = (real)0.0;
          }
        }
#endif
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
  if (Fluidtype != GAS && relmedian_cap_flag == 1) {
    ApplyRelmedianIterativeCpu(dens, vx, vy,
#ifdef Z
                               vz,
#else
                               NULL,
#endif
                               size_x, size_y, size_z, pitch,
                               relmedian_cap_max_iters,
                               relmedian_cap_factor,
                               relmedian_cap_rho_floor_factor);
  }
}
