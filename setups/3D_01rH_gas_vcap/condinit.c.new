#include "fargo3d.h"

/*
 * 3D initial conditions for Model B
 * ---------------------------------
 * Geometry: SPHERICAL (r, theta, phi) with cylindrical projections.
 * Aligned with Athena++ disk_planet_dust_spherical.cpp
 *
 * Gas:
 *   rho_g(R,z) = rho0 * (R/R0)^pvalue
 *                * exp[ GM/cs2(R) * ( 1/sqrt(R^2+z^2) - 1/R ) ]
 *   with cs2(R) = cs2_0 * (R/R0)^qvalue
 *
 *   v_R,g   = -(3/2) * alpha * cs2 / (Omega_K * R)
 *   v_phi,g = sqrt(GM/R) * sqrt( (pvalue+qvalue)*cs2/(GM/R) + 1 + qvalue - qvalue*R/sqrt(R^2+z^2) )
 *           - R * OmegaFrame
 *   v_z,g   = 0
 *
 * Dust:
 *   rho_d   = epsilon_init * rho_g
 *   v_R,d   = 2*St/(St^2 + (1+eps)^2) * eta * v_K
 *   v_phi,d = (1 + eps)*eta/(eps^2 + St^2) * v_K
 *   v_z,d   = 0
 *
 * Here:
 *   pvalue  = -2.25  (density slope)
 *   qvalue  = -0.5   (sound speed slope: cs^2 ∝ R^qvalue)
 *   eta     = 0.5 * (H/R)^2 * (pvalue + qvalue + 2)
 *   St      = 1 / INVSTOKES1
 *   eps     = EPSILON
 *
 * All velocities are finally expressed in the rotating frame:
 *   v_phi -> v_phi - OmegaFrame * R
 */

void _CondInit() {
  int i, j, k;
  real *rho = Density->field_cpu;
  real *e = Energy->field_cpu;
  real *vphi = Vx->field_cpu;  // v_phi
  real *vr = Vy->field_cpu;    // v_r (spherical)
  real *vtheta = Vz->field_cpu;// v_theta (spherical)

  real r_sph, theta, R_cyl, z_cyl; // geometry
  real omegaK, vK, H_over_R, cs2_loc;
  real rhog, rhod;
  real vR_g, vphi_g, vR_d, vphi_d; // cylindrical velocities
  real St, eta, eps_tot;
  // Model B parameters aligned with Athena++
  const real pvalue = -2.25;   // density power law index
  const real qvalue = -0.5;    // sound speed power law index
  real safe_rho_floor = 1.0e-12;

  i = j = k = 0;
  for (k = 0; k < Nz + 2 * NGHZ; k++) {
    for (j = 0; j < Ny + 2 * NGHY; j++) {
      for (i = 0; i < Nx + 2 * NGHX; i++) {

        // --- 1. Coordinate conversion ---
        // FARGO3D spherical: Y=r, Z=theta (colatitude)
        r_sph = Ymed(j);
        theta = Zmed(k);

        // Cylindrical coordinates
        R_cyl = r_sph * sin(theta);
        z_cyl = r_sph * cos(theta);

        // --- 2. Physical quantities ---
        // Safety for R_cyl
        if (R_cyl < 1.0e-6) R_cyl = 1.0e-6;

        // Keplerian orbital quantities
        omegaK = sqrt(G * MSTAR / (R_cyl * R_cyl * R_cyl));
        vK = omegaK * R_cyl;

        // Sound speed: cs^2 = cs2_0 * (R_cyl/R0)^qvalue
        // In FARGO, we use H/R = ASPECTRATIO * (R_cyl/R0)^FLARINGINDEX
        // and cs^2 = (H/R)^2 * GM/R = cs2_0 * (R_cyl/R0)^qvalue
        // => cs2_0 = (H/R)^2 * GM/R / (R_cyl/R0)^qvalue
        // but we can compute cs2 directly as:
        H_over_R = ASPECTRATIO * pow(R_cyl / R0, FLARINGINDEX);
        // Athena cs2 formula: cs2 = cs2_0 * (R_cyl/R0)^qvalue
        // We need to derive cs2_0 from our H/R setup
        // (H/R)^2 = ASPECTRATIO^2 * (R_cyl/R0)^(2*FLARINGINDEX)
        // cs2_from_H = (H/R)^2 * GM/R_cyl = ASPECTRATIO^2 * GM * (R_cyl/R0)^(2*FL-1) / R_cyl
        //            = ASPECTRATIO^2 * GM * (R_cyl/R0)^(2*FL-2)
        // For Athena: cs2_0 * (R_cyl/R0)^qvalue
        // So cs2_0_equiv = ASPECTRATIO^2 * GM * (R_cyl/R0)^(2*FL-2) / (R_cyl/R0)^qvalue
        //               = ASPECTRATIO^2 * GM * (R_cyl/R0)^(2*FL-2-qvalue)
        // At R_cyl = R0: cs2_0_equiv = ASPECTRATIO^2 * GM
        // But we can simplify: compute cs2 using the H/R approach directly

        // Use direct formula: cs^2 = cs2_0 * (R_cyl/R0)^qvalue
        // cs2_0 derived from: (H/R)^2 = ASPECTRATIO^2 at R_cyl=R0 for flat disk
        // But to match Athena exactly, we use the scale height consistency:
        // cs2 = (H/R)^2 * GM/R = ASPECTRATIO^2 * GM * (R_cyl/R0)^(2*FL) / R_cyl

        // Actually, let's use Athena's formula directly:
        // cs2_athena = cs2_0 * pow(R_cyl/R0, qvalue)
        // For our setup: cs2_0 = ASPECTRATIO^2 * GM (at R0 reference)
        // This gives: cs2 = ASPECTRATIO^2 * GM * pow(R_cyl/R0, qvalue)
        real cs2_0_ref = ASPECTRATIO * ASPECTRATIO * G * MSTAR;
        cs2_loc = cs2_0_ref * pow(R_cyl / R0, qvalue);

        // Gas Density - Athena formula
        // rho = rho0 * (R_cyl/R0)^pvalue * exp(GM/cs2 * (1/r_sph - 1/R_cyl))
        rhog = 1.0 * pow(R_cyl / R0, pvalue) *
               exp((G * MSTAR / cs2_loc) * (1.0 / r_sph - 1.0 / R_cyl));

        if (rhog < safe_rho_floor) rhog = safe_rho_floor;
        rhod = EPSILON * rhog;

        // Gas Velocities (Cylindrical)
        // vR = -1.5 * alpha * cs^2 / (Omega_K * R) = -1.5 * alpha * cs^2 * R / (GM/R^3)^0.5 / R
        //    = -1.5 * alpha * cs^2 * R / sqrt(GM*R)
        vR_g = -1.5 * ALPHA * cs2_loc * R_cyl / sqrt(G * MSTAR * R_cyl);

        // v_phi = sqrt(GM/R) * sqrt( (p+q)*cs2/(GM/R) + 1 + q - q*R/sqrt(R^2+z^2) )
        //       - R * OmegaFrame
        {
          real term1 = (pvalue + qvalue) * cs2_loc / (G * MSTAR / R_cyl);
          real term2 = 1.0 + qvalue;
          real term3 = qvalue * R_cyl / r_sph;
          real bracket = term1 + term2 - term3;
          if (bracket < 0.0) bracket = 0.0;
          vphi_g = sqrt(G * MSTAR / R_cyl) * sqrt(bracket) - R_cyl * OMEGAFRAME;
        }

        // Dust Velocities (Cylindrical) - NSH formulas
        eps_tot = 1.0 + EPSILON;
        eta = 0.5 * H_over_R * H_over_R * (pvalue + qvalue + 2.0);
        St = 1.0 / INVSTOKES1;
        vR_d = (2.0 * St / (St * St + eps_tot * eps_tot)) * eta * vK;
        vphi_d = (1.0 + eps_tot) * eta / (eps_tot * eps_tot + St * St) * vK + vK;

        // --- 3. Velocity projection (Cylindrical -> Spherical) ---
        // v_r     = v_R * sin(theta)
        // v_theta = v_R * cos(theta)
        // v_phi   = unchanged

        real vr_final, vtheta_final, vphi_final;

        if (Fluidtype == GAS) {
           rho[l] = rhog;
           // Adiabatic energy
           #ifdef ADIABATIC
             e[l] = rhog * cs2_loc / (GAMMA - 1.0);
           #else
             e[l] = sqrt(cs2_loc);
           #endif

           // Project gas velocities (vphi_g already includes OmegaFrame subtraction)
           vr_final     = vR_g * sin(theta);
           vtheta_final = vR_g * cos(theta);
           vphi_final   = vphi_g;
        }
        else { // DUST
           rho[l] = rhod;
           e[l] = 0.0;

           // Project dust velocities
           vr_final     = vR_d * sin(theta);
           vtheta_final = vR_d * cos(theta);
           vphi_final   = vphi_d - R_cyl * OMEGAFRAME;
        }

        // 写入场
        vr[l] = vr_final;
        vtheta[l] = vtheta_final;
        vphi[l] = vphi_final;
      }
    }
  }
}

void CondInit()
{

  int id_gas = 0;
  int feedback = YES;

  /* Gas fluid */
  Fluids[id_gas] = CreateFluid("gas", GAS);
  SelectFluid(id_gas);
  _CondInit();

  /* Dust fluid(s) */
  char dust_name[MAXNAMELENGTH];
  int id_dust;

  for (id_dust = 1; id_dust < NFLUIDS; id_dust++)
  {
    sprintf(dust_name, "dust%d", id_dust);
    Fluids[id_dust] = CreateFluid(dust_name, DUST);
    SelectFluid(id_dust);
    _CondInit();
  }

  /* Collision matrix (dust-gas drag) */
  ColRate(INVSTOKES1, id_gas, 1, feedback);
  /* Additional dust species would go here, e.g.:
   * ColRate(INVSTOKES2, id_gas, 2, feedback);
   * ColRate(INVSTOKES3, id_gas, 3, feedback);
   */
}
