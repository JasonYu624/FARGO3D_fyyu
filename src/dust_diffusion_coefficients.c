//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"

// Enable planetary diffusivity reduction by default (0=off, 1=on)
#ifndef PLANET_DIFF_REDUCTION
#define PLANET_DIFF_REDUCTION 0
#endif

// Enable dust-to-gas diffusivity reduction (0=off, 1=on)
#ifndef DUSTDIFFD2G_REDUCTION
#define DUSTDIFFD2G_REDUCTION 0
#endif
//<\INCLUDES>

#if DUSTDIFFD2G_REDUCTION && !defined(ALPHAVISCOSITY)
#error "DUSTDIFFD2G_REDUCTION currently requires ALPHAVISCOSITY for gas density access"
#endif

void DustDiffusion_Coefficients_cpu() {

//<USER_DEFINED>
#ifdef ALPHAVISCOSITY
  INPUT(Energy);
#ifdef ADIABATIC
  INPUT(Density);
#else
  // When not adiabatic, we still might need Density for DUSTDIFFD2G_REDUCTION
  INPUT(Density);
#endif
#endif
  OUTPUT(Sdiffyczc);
  OUTPUT(Sdiffyfzc);
#ifdef Z
  OUTPUT(Sdiffyczf);
  OUTPUT(Sdiffyfzf);
#endif
//<\USER_DEFINED>

//<EXTERNAL>
  real* sdiff_yfzc = Sdiffyfzc->field_cpu;
  real* sdiff_yczc = Sdiffyczc->field_cpu;
#ifdef Z
  real* sdiff_yczf = Sdiffyczf->field_cpu;
  real* sdiff_yfzf = Sdiffyfzf->field_cpu;
#endif
#ifdef ALPHAVISCOSITY  
#ifdef ISOTHERMAL
  real* cs = Fluids[0]->Energy->field_cpu;
#endif
#ifdef ADIABATIC
  real* e = Fluids[0]->Energy->field_cpu;
  real* rhog = Fluids[0]->Density->field_cpu;
  real gamma = GAMMA;
#endif
  real alphavisc = ALPHA;
#else
  real nu = NU;
#endif
  real invstokes = INVSTOKES1;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real* rhod_field = Density->field_cpu;
//<\EXTERNAL>
  
//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
  int llym;
  int llzm;
#ifdef ALPHAVISCOSITY
  real r3yczc;
  real r3yfzc;
  real soundspeed2;
  real soundspeedf2;
  real soundspeedfz2;
#endif
  real St;
  real correction;
//<\INTERNAL>

//<CONSTANT>
// real xmin(Nx+1);
// real ymin(Ny+2*NGHY+1);
// real zmin(Nz+2*NGHZ+1);
// real Xplanet(1);
// real Yplanet(1);
// real Zplanet(1);
// real MplanetVirtual(1);
// real ROCHESMOOTHING(1);
//<\CONSTANT>

//<MAIN_LOOP>
  i = j = k = 0;
#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
#ifdef Y
	llym = lym;
#endif
#ifdef Z
	llzm = lzm;
#endif
#ifdef ALPHAVISCOSITY
#ifdef ISOTHERMAL
	soundspeed2 = cs[ll]*cs[ll];
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else
	  soundspeedf2 = 0.5*(cs[ll]+cs[llym])*0.5*(cs[ll]+cs[llym]);
#ifdef Z
	if(k==0)
          soundspeedfz2 = soundspeed2;
        else
          soundspeedfz2 = 0.5*(cs[ll]+cs[llzm])*0.5*(cs[ll]+cs[llzm]);
#endif //Z
#endif
#ifdef ADIABATIC
	soundspeed2 = gamma*(gamma-1.0)*e[ll]/rhog[ll];
	if(j==0)
	  soundspeedf2 = soundspeed2;
	else
	  soundspeedf2 = gamma*(gamma-1.0)*(e[ll]+e[llym])/(rhog[ll]+rhog[llym]);

#ifdef Z
	// Z (极角) 方向界面声速平方 (官方漏掉的核心补丁)
	if(k==0)
	  soundspeedfz2 = soundspeed2;
	else
	  soundspeedfz2 = gamma*(gamma-1.0)*(e[ll]+e[llzm])/(rhog[ll]+rhog[llzm]);
#endif // Z
#endif // ADIABATIC
	r3yczc = ymed(j)*ymed(j)*ymed(j);
	r3yfzc = ymin(j)*ymin(j)*ymin(j);
  St = 1.0 / 10.0;
  correction = 1.0 / (1.0 + St * St);
	sdiff_yczc[ll] = alphavisc*soundspeed2/sqrt(G*MSTAR/r3yczc)*correction;
	sdiff_yfzc[ll] = alphavisc*soundspeedf2/sqrt(G*MSTAR/r3yfzc)*correction;
#ifdef Z
	sdiff_yczf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yczc)*correction;
	sdiff_yfzf[ll] = alphavisc*soundspeedfz2/sqrt(G*MSTAR/r3yfzc)*correction;
#endif //Z

#if PLANET_DIFF_REDUCTION
	// Apply planetary diffusivity reduction near planet
	// Athena-style: ν_eff = ν × min(1, Ω_K / sqrt(Ω_K² + Ω_p²))
	// where Ω_p = sqrt(GMp / r_soft³), r_soft = sqrt(d² + rs²)
	if (MplanetVirtual > 0.0) {
	  // Cell position in spherical coordinates
	  real r_s = ymed(j);
#ifdef Z
	  real theta = zmed(k);
	  real phi = xmed(i);
	  real x_cell = r_s * sin(theta) * cos(phi);
	  real y_cell = r_s * sin(theta) * sin(phi);
	  real z_cell = r_s * cos(theta);
#else
	  real x_cell = r_s;
	  real y_cell = 0.0;
	  real z_cell = 0.0;
#endif
	  // Distance to planet
	  real dx = x_cell - Xplanet;
	  real dy = y_cell - Yplanet;
	  real dz = z_cell - Zplanet;
	  real dist2 = dx*dx + dy*dy + dz*dz;

	  // Cylindrical radius
	  real R_cyl = r_s * sin(zmed(k));
	  if (R_cyl < 1e-12) R_cyl = 1e-12;

	  // Keplerian frequency
	  real omega_k2 = G * MSTAR / (R_cyl * R_cyl * R_cyl);
	  real omega_k = sqrt(omega_k2);

	  // Softened distance
	  real soft_dist2 = dist2 + ROCHESMOOTHING * ROCHESMOOTHING;
	  real soft_dist3 = pow(soft_dist2, 1.5);

	  // Planet's induced frequency (GMp = G * MplanetVirtual)
	  real planet_gm = G * MplanetVirtual;
	  real omega_p2 = (soft_dist3 > 1e-30) ? planet_gm / soft_dist3 : 0.0;

	  // Reduction factor
	  real omega_eff = sqrt(omega_k2 + omega_p2);
	  real reduction = (omega_eff > 1e-30) ? omega_k / omega_eff : 1.0;
	  if (reduction > 1.0) reduction = 1.0;
	  if (reduction < 0.0) reduction = 0.0;

	  // Apply reduction
	  sdiff_yczc[ll] *= reduction;
	  sdiff_yfzc[ll] *= reduction;
#ifdef Z
	  sdiff_yczf[ll] *= reduction;
	  sdiff_yfzf[ll] *= reduction;
#endif
	}
#endif // PLANET_DIFF_REDUCTION

#if DUSTDIFFD2G_REDUCTION
	// Apply dust-to-gas ratio based diffusivity reduction
	// Athena: nu_eff = nu / (1 + rho_d/rho_g)
	// This reduces diffusion in dust trap regions where D2G is high
	// Get current dust density from the array
	real rhod = rhod_field[ll];
	real rhog_safe = rhog[ll];
	if (rhog_safe < 1e-30) rhog_safe = 1e-30;
	real d2g = rhod / rhog_safe;
	if (d2g > 0.0) {
	  real d2g_reduction = 1.0 / (1.0 + d2g);
	  sdiff_yczc[ll] *= d2g_reduction;
	  sdiff_yfzc[ll] *= d2g_reduction;
#ifdef Z
	  sdiff_yczf[ll] *= d2g_reduction;
	  sdiff_yfzf[ll] *= d2g_reduction;
#endif
	}
#endif // DUSTDIFFD2G_REDUCTION

#endif
#ifdef VISCOSITY
	sdiff_yczc[ll] = nu;
	sdiff_yfzc[ll] = nu;
#ifdef Z
	sdiff_yczf[ll] = nu;
	sdiff_yfzf[ll] = nu;
#endif
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
}
