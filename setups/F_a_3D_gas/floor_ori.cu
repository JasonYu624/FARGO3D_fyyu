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
    // 严防极点处的除零错误
    if (R_cyl < 1.0e-6) R_cyl = 1.0e-6;

    real dist_exp = -sigmaslope - 1.0 - flaringindex;
    real H = aspectratio * pow(R_cyl / r0, flaringindex) * R_cyl;

    // 简化后的指数项，更稳定
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
// Kernel: 应用 Floor 和 统一动能天花板 (免疫 Fast-Math 优化版)
// ----------------------------------------------------------------------
__global__ void Apply_Floor_Kernel(real *dens,
#ifdef ADIABATIC
                                  real *energy,
#else
                                  real * /*energy*/,
#endif
                                  real *vx, real *vy, real *vz,
                                  const real *ymin, const real *zmin,
                                  int size_x, int size_y, int size_z, int pitch,
                                  int fluidtype,
                                  real gas_abs_floor, real gas_rel_floor,
                                  real dust_abs_floor, real dust_rel_floor,
                                  real energy_floor,
                                  real sigma0, real r0,
                                  real aspectratio, real flaringindex, real sigmaslope,
                                  real gm) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = size_x * size_y * size_z;
  if (idx >= total) return;

  int i = idx % size_x;
  int t = idx / size_x;
  int j = t % size_y;
  int k = t / size_y;
  int mem = i + pitch * (j + k * size_y);

  // 1. 获取几何坐标
  real y_val = ymin[j];
#ifdef Z
  real z_val = 0.5f * (zmin[k] + zmin[k+1]);
#else
  real z_val = 0.0f;
#endif

  real init_rho = Get_Init_Density_Analytic(y_val, z_val, sigma0, r0, aspectratio, flaringindex, sigmaslope);

  // Failsafe: 如果理论公式算出 NaN，强行给一个安全的底 (同样用反向逻辑捕获 NaN)
  if (!(init_rho >= 0.0 && init_rho <= 1e30)) init_rho = 1.0e-6;

  real limit_floor;
  if (fluidtype == GAS) {
    limit_floor = gas_rel_floor * init_rho;
    if (limit_floor < gas_abs_floor) limit_floor = gas_abs_floor;
  } else {
    limit_floor = dust_rel_floor * init_rho;
    if (limit_floor < dust_abs_floor) limit_floor = dust_abs_floor;
  }

  // ==============================================================
  // 2. Density Floor 与 致命 NaN 拦截器
  // ==============================================================
  real d = dens[mem];
  // 核心技巧：!(d >= limit) 完美捕捉 d < limit, NaN, -INF
  if (!(d >= limit_floor)) {
      dens[mem] = limit_floor;
  }

  // ==============================================================
  // 3. 速度场 NaN 和 INF 清理
  // ==============================================================
  real v_x = vx[mem];
  if (!(v_x >= -1e30 && v_x <= 1e30)) v_x = 0.0;
  vx[mem] = v_x;

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
  // 4. 统一动能天花板 (Velocity Ceiling for BOTH Gas & Dust)
  // ==============================================================
#ifdef Z
#ifdef SPHERICAL
  if (vz != NULL) {
      real R_cyl = y_val * sin(z_val);
      if (R_cyl < 1e-6) R_cyl = 1e-6;

      real vK = sqrt(gm / R_cyl);
      real cs_ana = aspectratio * pow(R_cyl / r0, flaringindex) * vK;

      // 全局物理安全网: 限制 v_r 和 v_theta 不超过 Mach 2.0
      real vlim = 2.0 * cs_ana;

      if (v_y > vlim) v_y = vlim;
      else if (v_y < -vlim) v_y = -vlim;

      if (v_z > vlim) v_z = vlim;
      else if (v_z < -vlim) v_z = -vlim;
  }
#endif
#endif

  // 将经过净化和限速的 v_y 和 v_z 写回显存
  vy[mem] = v_y;
#ifdef Z
  if (vz != NULL) vz[mem] = v_z;
#endif

#ifdef ADIABATIC
  // ==============================================================
  // 5. 能量 Floor (同样加入反向逻辑)
  // ==============================================================
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

  const real gas_abs_floor  = 1.0e-10;
  const real gas_rel_floor  = 1.0e-6;
  const real dust_abs_floor = 1.0e-12;
  const real dust_rel_floor = 1.0e-8;
  const real energy_floor   = 1.0e-20;

  const real aspectratio  = ASPECTRATIO;
  const real flaringindex = FLARINGINDEX;
  const real r0           = R0;
  const real sigma0       = SIGMA0;
  const real sigmaslope   = SIGMASLOPE;
  const real gm           = G * MSTAR;

  int blockSize = 256;
  int gridSize  = (total_cells + blockSize - 1) / blockSize;

  Apply_Floor_Kernel<<<gridSize, blockSize>>>(
      d_dens,
      d_energy,
      d_vx, d_vy, d_vz,
      Ymin_d, d_zmin,
      sx, sy, sz, Pitch_gpu,
      Fluidtype,
      gas_abs_floor, gas_rel_floor,
      dust_abs_floor, dust_rel_floor,
      energy_floor,
      sigma0, r0, aspectratio, flaringindex, sigmaslope, gm
  );

  cudaDeviceSynchronize();
}