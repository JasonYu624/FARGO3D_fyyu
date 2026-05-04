//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//<\INCLUDES>

// Planetary Accretion (Athena++ Mode 2)
// Accretes gas and dust inside a specified radius around the planet

void Accretion_cpu(real dt) {

//<USER_DEFINED>
  INPUT(Density);
  INPUT(Vx);
  INPUT(Vy);
  INPUT(Vz);
  OUTPUT(Density);
#ifdef ADIABATIC
  INPUT(Energy);
  OUTPUT(Energy);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* rho  = Density->field_cpu;
#ifdef X
  real* vx  = Vx->field_cpu;
#endif
#ifdef Y
  real* vy  = Vy->field_cpu;
#endif
#ifdef Z
  real* vz  = Vz->field_cpu;
#endif
#ifdef ADIABATIC
  real* e    = Energy->field_cpu;
#endif

  int pitch   = Pitch_cpu;
  int size_x  = Nx+2*NGHX;
  int size_y  = Ny+2*NGHY;
  int size_z  = Nz+2*NGHZ;

  real planet_x = Xplanet;
  real planet_y = Yplanet;
  real planet_z = Zplanet;
  real planet_mass = MplanetVirtual;
  real softening = ROCHESMOOTHING;
  real accret_radius_ratio = ACCRETION_RADIUS_RATIO;
  real accret_tau_factor = ACCRETION_TAU_FACTOR;
  real g = G;
  real mstar = MSTAR;
  int accret_flag = PLANETARY_ACCRETION_FLAG;
//<\EXTERNAL>

//<INTERNAL>
  int i, j, k, ll;
  real accretion_radius, accretion_radius2;
  real removed_mass_local = (real)0.0;
//<\INTERNAL>

//<MAIN_LOOP>

  if (accret_flag == 0 || planet_mass <= 0.0) {
    return;
  }

  // Accretion radius = softening_length * ratio
  accretion_radius = softening * accret_radius_ratio;
  accretion_radius2 = accretion_radius * accretion_radius;

  i = j = k = 0;

  for (k=0; k<size_z; k++) {
    for (j=0; j<size_y; j++) {
      for (i=0; i<size_x; i++) {
//<#>
        ll = l;

        // Cell center position in cylindrical coordinates
        real r_sph = ymed(j);
#ifdef SPHERICAL
        real theta = zmed(k);
        real phi = xmed(i);
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

        // Skip if outside accretion radius
        if (dist2 >= accretion_radius2) continue;

        real distance = sqrt(dist2);
        if (distance < 1e-12) distance = 1e-12;

        // Sink shape function (Athena style)
        // smooth transition from 1 at r=0 to 0 at r=accretion_radius
        real d_ratio = distance / accretion_radius;
        real sink_shape;
        if (d_ratio <= 0.5) {
          sink_shape = ((6.0*d_ratio - 6.0)*d_ratio)*d_ratio + 1.0;
        } else if (d_ratio <= 1.0) {
          sink_shape = 2.0 * pow(1.0 - d_ratio, 3);
        } else {
          sink_shape = 0.0;
        }

        if (sink_shape <= 0.0) continue;

        // Local timescale for accretion (Mode 2)
        // tau_local = tau_factor * sqrt(r^3 / GM_planet)
        real soft_dist2 = dist2 + softening * softening;
        real soft_distance = sqrt(soft_dist2);
        real planet_gm = g * planet_mass;
        real tau_local = accret_tau_factor * sqrt(soft_dist2 * soft_distance / planet_gm);
        if (tau_local < 1e-30) tau_local = 1e-30;

        // Remove fraction
        real dt_over_tau = dt / tau_local;
        if (dt_over_tau > 50.0) dt_over_tau = 50.0; // cap to avoid numerical issues

        real remove_fraction = sink_shape * (1.0 - exp(-dt_over_tau));
        remove_fraction = fmin(0.5, fmax(0.0, remove_fraction)); // cap at 50%

        if (remove_fraction <= 0.0) continue;

        // Apply to density
        real retain = 1.0 - remove_fraction;
        removed_mass_local += rho[ll] * remove_fraction * Vol(i,j,k);
        rho[ll] *= retain;

#ifdef ADIABATIC
        e[ll] *= retain;
#endif

#ifdef X
        vx[ll] *= retain;
#endif
#ifdef Y
        vy[ll] *= retain;
#endif
#ifdef Z
        if (vz != NULL) vz[ll] *= retain;
#endif

//<\#>
      }
    }
  }

//<\MAIN_LOOP>
  if (Fluidtype == GAS) {
    AccretedGasMassRun += removed_mass_local;
  } else {
    AccretedDustMassRun += removed_mass_local;
  }
}
