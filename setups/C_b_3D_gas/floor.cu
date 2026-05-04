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

// ----------------------------------------------------------------------
// Device Function: 计算理论初始密度
// ----------------------------------------------------------------------
__device__ real Get_Init_Density_Analytic(real y, real z, 
                                          real sigma0, real r0, 
                                          real aspectratio, real flaringindex, 
                                          real sigmaslope) {
    real R_cyl, r_sph, theta;

#ifdef Z
    r_sph = y;
    theta = z;
    R_cyl = r_sph * sin(theta);
    if (R_cyl < 1.0e-6) R_cyl = 1.0e-6;

    real dist_exp = -sigmaslope - 1.0 - flaringindex; 
    real H = aspectratio * pow(R_cyl / r0, flaringindex) * R_cyl;
    real val = sigma0 * pow(R_cyl / r0, dist_exp) * exp( (R_cyl / (H * H)) * (1.0/r_sph - 1.0/R_cyl) );
    return val;
#else
    R_cyl = y;
    if (R_cyl < 1.0e-6) R_cyl = 1.0e-6;
    real val = sigma0 * pow(R_cyl / r0, -sigmaslope);
    return val;
#endif
}

// ----------------------------------------------------------------------
// Kernel: 应用 Floor 和 Velocity Ceilings (Gas & Dust 完美分离)
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
                                  real sigma0, real r0, 
                                  real aspectratio, real flaringindex, real sigmaslope,
                                  real gm, real invstokes) {

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
  real z_val = 0.5f * (zmin[k] + zmin[k+1]); 
#else
  real z_val = 0.0f;
#endif

  // 1. Density Floor 限制
  real limit_floor = (fluidtype == GAS) ? gas_abs_floor : dust_abs_floor;
  real d = dens[mem];
  if (!(d >= limit_floor)) {
      dens[mem] = limit_floor;
  }

  // 2. 速度场 NaN 拦截器
  real v_x = vx[mem];
  if (!(v_x >= -1e30 && v_x <= 1e30)) v_x = 0.0;

  real v_y = vy[mem];
  if (!(v_y >= -1e30 && v_y <= 1e30)) v_y = 0.0;

#ifdef Z
  real v_z = 0.0;
  if (vz != NULL) {
      v_z = vz[mem];
      if (!(v_z >= -1e30 && v_z <= 1e30)) v_z = 0.0;
  }
#endif

  // ==============================================================
  // 3A. 气体的克制天花板：恢复你原本的高空保护逻辑！
  // ==============================================================
#ifdef Z
#ifdef SPHERICAL
  if (fluidtype == GAS && vz != NULL) {
      const real NH_CEIL       = 3.5;
      const real MACH_CEIL     = 1.0; 
      const real LOW_DENS_MULT = 50.0;

      if (dens[mem] < LOW_DENS_MULT * limit_floor) {
          real R_cyl = y_val * sin(z_val);
          if (R_cyl < 1e-12) R_cyl = 1e-12;
          
          real dth = fabs(z_val - 0.5f * M_PI);
          real h = aspectratio * pow(R_cyl / r0, flaringindex);

          // 仅在高空生效，保护 k=10 等极地网格
          if (dth > NH_CEIL * h) {
              real vK   = sqrt(gm / R_cyl);
              real cs_l = h * vK;
              real vlim = MACH_CEIL * cs_l;

              if (v_z >  vlim) v_z =  vlim;
              if (v_z < -vlim) v_z = -vlim;
          }
      }
  }
#endif
#endif

  // ==============================================================
  // 3B. 尘埃物理限速：严格执行终端速度近似 (Terminal Velocity Reset)
  // ==============================================================
  if (fluidtype != GAS) {
      real R_cyl = y_val;
#ifdef Z
      R_cyl = y_val * sin(z_val);
#endif
      if (R_cyl < 1e-6) R_cyl = 1e-6;

      real vK = sqrt(gm / R_cyl);
      real h = aspectratio * pow(R_cyl / r0, flaringindex);
      real cs_ana = h * vK;

      real rhog = gas_dens[mem];
      real rhod = dens[mem]; 

      // Collaborator 建议条件：rho_dust/rho_gas < 1
      if (rhod < rhog) {
          #ifdef Z
          const real dlnP_dlnR = -2.75;
          #else
          const real dlnP_dlnR = -1.5;
          #endif

          real eta = 0.5 * h * h * dlnP_dlnR;
          real St = 1.0 / invstokes;

          real dv_r_theo = (2.0 * St / (1.0 + St * St)) * eta * vK;
          real dv_phi_theo = -(St * St / (1.0 + St * St)) * eta * vK;

          v_y = gas_vy[mem] + dv_r_theo;
          v_x = gas_vx[mem] + dv_phi_theo;

          // zero theta difference
          #ifdef Z
          if (vz != NULL && gas_vz != NULL) {
              v_z = gas_vz[mem];
          }
          #endif
      } else {
          // Fallback Mach 2.0 ceiling
          real vlim = 2.0 * cs_ana; 
          if (v_y > vlim) v_y = vlim;
          else if (v_y < -vlim) v_y = -vlim;

          #ifdef Z
          if (vz != NULL) {
              if (v_z > vlim) v_z = vlim;
              else if (v_z < -vlim) v_z = -vlim;
          }
          #endif
      }
  }

  // 写入最终的健康速度
  vx[mem] = v_x;
  vy[mem] = v_y;
#ifdef Z
  if (vz != NULL) vz[mem] = v_z;
#endif

  // ==============================================================
  // 4. 能量 (Pressure) Floor (Relaxed to 1e-10 to free dt)
  // ==============================================================
#ifdef ADIABATIC
  if (energy != NULL) {
      if (fluidtype == GAS) {
           real e = energy[mem];
           if (!(e >= energy_floor)) energy[mem] = energy_floor;
      } else {
           energy[mem] = 0.0;
      }
  }
#endif
}

// -----------------------------
// Host Wrapper
// -----------------------------
extern "C" void Floor_gpu() {
  int sx = Nx + 2 * NGHX;
  int sy = Ny + 2 * NGHY;
  int sz = Nz + 2 * NGHZ;
  int total_cells = sx * sy * sz;

  real *d_dens   = Density->field_gpu;
  real *d_vx     = Vx->field_gpu;
  real *d_vy     = Vy->field_gpu;
#ifdef Z
  real *d_vz     = (Vz != NULL) ? Vz->field_gpu : NULL;
  real *d_zmin   = Zmin_d;
#else
  real *d_vz     = NULL;
  real *d_zmin   = NULL;
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

  const real gas_abs_floor  = 1.0e-7;   
  const real dust_abs_floor = 1.0e-9;   
  const real energy_floor   = 1.0e-10;  

  const real aspectratio  = ASPECTRATIO;
  const real flaringindex = FLARINGINDEX;
  const real r0           = R0;
  const real sigma0       = SIGMA0;
  const real sigmaslope   = SIGMASLOPE;
  const real gm           = G * MSTAR;
  const real invstokes    = INVSTOKES1; 

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
      sigma0, r0, aspectratio, flaringindex, sigmaslope, gm, invstokes
  );

  cudaDeviceSynchronize();
}