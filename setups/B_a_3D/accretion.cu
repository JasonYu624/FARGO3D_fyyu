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

// Planetary Accretion (Athena++ Mode 2) - GPU version

extern int Pitch_gpu;
__device__ real d_accreted_mass_removed;

__device__ __forceinline__ real atomicAddReal(real *address, real val) {
#ifdef FLOAT
  return atomicAdd(address, val);
#else
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
#endif
}

__global__ void Accretion_kernel(real *rho,
#ifdef X
real *vx,
#endif
#ifdef Y
real *vy,
#endif
#ifdef Z
real *vz,
#endif
#ifdef ADIABATIC
real *e,
#endif
const real *ymin, const real *zmin, const real *xmin,
int pitch, int size_x, int size_y, int size_z,
real planet_x, real planet_y, real planet_z,
real planet_mass, real softening,
real accret_radius_ratio, real accret_tau_factor,
real g, int accret_flag,
real dt) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = size_x * size_y * size_z;
  if (idx >= total) return;
  if (accret_flag == 0 || planet_mass <= 0.0) return;

  int i = idx % size_x;
  int t = idx / size_x;
  int j = t % size_y;
  int k = t / size_y;
  int mem = i + pitch * (j + k * size_y);

  // Cell center position - calculate from face-centered coordinates
  real r_sph = ymin[j];
#ifdef SPHERICAL
  real theta = 0.5 * (zmin[k] + zmin[k+1]);
  real phi = 0.5 * (xmin[i] + xmin[i+1]);
  real x_cell = r_sph * sin(theta) * cos(phi);
  real y_cell = r_sph * sin(theta) * sin(phi);
  real z_cell = r_sph * cos(theta);
#else
  real x_cell = r_sph;
  real y_cell = 0.0;
  real z_cell = 0.0;
#endif

  // Distance to planet
  real dx = x_cell - planet_x;
  real dy = y_cell - planet_y;
  real dz = z_cell - planet_z;
  real dist2 = dx*dx + dy*dy + dz*dz;

  // Accretion radius
  real accretion_radius = softening * accret_radius_ratio;
  real accretion_radius2 = accretion_radius * accretion_radius;

  // Skip if outside accretion radius
  if (dist2 >= accretion_radius2) return;

  real distance = sqrt(dist2);
  if (distance < 1e-12) distance = 1e-12;

  // Sink shape function (Athena style)
  real d_ratio = distance / accretion_radius;
  real sink_shape;
  if (d_ratio <= 0.5) {
    sink_shape = ((6.0*d_ratio - 6.0)*d_ratio)*d_ratio + 1.0;
  } else if (d_ratio <= 1.0) {
    sink_shape = 2.0 * pow(1.0 - d_ratio, 3);
  } else {
    sink_shape = 0.0;
  }

  if (sink_shape <= 0.0) return;

  // Local timescale for accretion (Mode 2)
  real soft_dist2 = dist2 + softening * softening;
  real soft_distance = sqrt(soft_dist2);
  real planet_gm = g * planet_mass;
  real tau_local = accret_tau_factor * sqrt(soft_dist2 * soft_distance / planet_gm);
  if (tau_local < 1e-30) tau_local = 1e-30;

  // Remove fraction
  real dt_over_tau = dt / tau_local;
  if (dt_over_tau > 50.0) dt_over_tau = 50.0;

  real remove_fraction = sink_shape * (1.0 - exp(-dt_over_tau));
  if (remove_fraction > 0.5) remove_fraction = 0.5;
  if (remove_fraction <= 0.0) return;

  // Apply to density
  real retain = 1.0 - remove_fraction;
  {
    real cell_volume = (real)1.0;
#ifdef SPHERICAL
    real r_in = ymin[j];
    real r_out = ymin[j+1];
    real phi_in = xmin[i];
    real phi_out = xmin[i+1];
    real theta_in = zmin[k];
    real theta_out = zmin[k+1];
    cell_volume = ((r_out*r_out*r_out - r_in*r_in*r_in) / (real)3.0) *
                  (cos(theta_in) - cos(theta_out)) *
                  (phi_out - phi_in);
#endif
    atomicAddReal(&d_accreted_mass_removed, (real)(rho[mem] * remove_fraction * cell_volume));
  }
  rho[mem] *= retain;

#ifdef ADIABATIC
  e[mem] *= retain;
#endif

#ifdef X
  vx[mem] *= retain;
#endif
#ifdef Y
  vy[mem] *= retain;
#endif
#ifdef Z
  if (vz != NULL) vz[mem] *= retain;
#endif
}

extern "C" void Accretion_gpu(real dt) {
  int sx = Nx + 2 * NGHX;
  int sy = Ny + 2 * NGHY;
  int sz = Nz + 2 * NGHZ;
  int total_cells = sx * sy * sz;

  if (PLANETARY_ACCRETION_FLAG == 0 || MplanetVirtual <= (real)0.0) return;

  real *d_rho = Density->field_gpu;
#ifdef X
  real *d_vx = Vx->field_gpu;
#endif
#ifdef Y
  real *d_vy = Vy->field_gpu;
#endif
#ifdef Z
  real *d_vz = (Vz != NULL) ? Vz->field_gpu : NULL;
  real *d_zmin = Zmin_d;
#else
  real *d_zmin = NULL;
#endif
#ifdef ADIABATIC
  real *d_e = (Energy != NULL) ? Energy->field_gpu : NULL;
#endif

  int blockSize = 256;
  int gridSize = (total_cells + blockSize - 1) / blockSize;
  real zero_mass = (real)0.0;
  real removed_mass_local = (real)0.0;

  cudaMemcpyToSymbol(d_accreted_mass_removed, &zero_mass, sizeof(real), 0, cudaMemcpyHostToDevice);

  Accretion_kernel<<<gridSize, blockSize>>>(
      d_rho,
#ifdef X
      d_vx,
#endif
#ifdef Y
      d_vy,
#endif
#ifdef Z
      d_vz,
#endif
#ifdef ADIABATIC
      d_e,
#endif
      Ymin_d, d_zmin, Xmin_d,
      Pitch_gpu, sx, sy, sz,
      (real)Xplanet, (real)Yplanet, (real)Zplanet,
      (real)MplanetVirtual, (real)ROCHESMOOTHING,
      (real)ACCRETION_RADIUS_RATIO, (real)ACCRETION_TAU_FACTOR,
      (real)G, (int)PLANETARY_ACCRETION_FLAG,
      dt);

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Accretion launch failed on rank %d: %s\n",
            CPU_Rank, cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Accretion sync failed on rank %d: %s\n",
            CPU_Rank, cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }

  err = cudaMemcpyFromSymbol(&removed_mass_local, d_accreted_mass_removed, sizeof(real), 0, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Accretion diagnostics copy failed on rank %d: %s\n",
            CPU_Rank, cudaGetErrorString(err));
    fflush(stderr);
    exit(1);
  }

  if (Fluidtype == GAS) {
    AccretedGasMassRun += removed_mass_local;
  } else {
    AccretedDustMassRun += removed_mass_local;
  }
}
