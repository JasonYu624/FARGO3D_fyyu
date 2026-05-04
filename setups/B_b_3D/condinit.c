#include "fargo3d.h"

/*
 * 3D initial conditions for Model B
 * ---------------------------------
 * Geometry: CYLINDRICAL (R, phi, z).
 *
 * Gas:
 *   rho_g(R,z) = rho0 (R/R0)^(-9/4)
 *                * exp[ GM/c_s^2(R) * ( 1/sqrt(R^2+z^2) - 1/R ) ]
 *   with rho0 = 1, c_s^2(R) ∝ (R/R0)^(-1/2)
 *
 *   v_R,g   = -(3/2) * alpha * c_s^2 / (Omega_K * R)
 *           = -(3/2) * alpha * (H/R)^2 * Omega_K * R
 *   v_phi,g = R Omega_K * [ 1/2 - (11/4)(H/R)^2 + R/(2 sqrt(R^2+z^2)) ]^{1/2}
 *   v_z,g   = 0
 *
 * Dust:
 *   rho_d   = epsilon_init * rho_g
 *   v_R,d   = 2 / ( St + (1+eps)^2 / St ) * eta v_K
 *   v_phi,d = [ 1 + (1+eps) eta / ((1+eps)^2 + St^2) ] v_K
 *   v_z,d   = 0
 *
 * Here:
 *   H/R      = ASPECTRATIO * (R/R0)^FLARINGINDEX
 *   d ln P / d ln R = -11/4  (for Sigma ∝ R^-1, T ∝ R^-1/2)
 *   eta      = 0.5 (H/R)^2 d ln P / d ln R
 *   St       = 1 / INVSTOKES1
 *   eps      = EPSILON
 *
 * All velocities are finally expressed in the rotating frame:
 *   v_phi -> v_phi - OMEGAFRAME * R
 */

void _CondInit() {
  int i, j, k;
  real *rho = Density->field_cpu;
  real *e = Energy->field_cpu; 
  real *vphi = Vx->field_cpu;  // v_phi (不变)
  real *vr = Vy->field_cpu;    // 注意：这里变成了球坐标径向速度 v_r
  real *vtheta = Vz->field_cpu;// 注意：这里变成了极角速度 v_theta

  real r_sph, theta, R_cyl, z_cyl; // 几何变量
  real omegaK, vK, H_over_R, cs_loc;
  real rhog, rhod;
  real vR_g, vphi_g, vR_d, vphi_d; // 圆柱坐标下的速度
  real St, eta, eps_tot;
  const real dlnP_dlnR = -11.0 / 4.0;
  real safe_rho_floor = 1.0e-12;

  i = j = k = 0;
  for (k = 0; k < Nz + 2 * NGHZ; k++) {
    for (j = 0; j < Ny + 2 * NGHY; j++) {
      for (i = 0; i < Nx + 2 * NGHX; i++) {
        
        // --- 1. 坐标转换 ---
        // FARGO3D 球坐标: Y=r, Z=theta (colatitude)
        r_sph = Ymed(j);
        theta = Zmed(k);
        
        // 转换为圆柱坐标 (用于代入 Model B 公式)
        R_cyl = r_sph * sin(theta);
        z_cyl = r_sph * cos(theta);
        
        // --- 2. 计算物理量 (与之前逻辑完全一致，只是输入变成了 R_cyl, z_cyl) ---
        // 注意：分母中的距离 r 在论文里是球半径，即这里的 r_sph
        omegaK = sqrt(G * MSTAR / (R_cyl * R_cyl * R_cyl)); // Omega_K 依赖 R_cyl
        vK = omegaK * R_cyl;

        H_over_R = ASPECTRATIO * pow(R_cyl / R0, FLARINGINDEX);
        cs_loc = H_over_R * vK;

        // Gas Density
        rhog = pow(R_cyl / R0, -2.25) *
               exp((R_cyl / (ASPECTRATIO * ASPECTRATIO * pow(R_cyl / R0, 2.0 * FLARINGINDEX))) * (1.0 / r_sph - 1.0 / R_cyl));
        
        if (rhog < safe_rho_floor) rhog = safe_rho_floor;
        rhod = EPSILON * rhog;

        // Gas Velocities (Cylindrical)
        vR_g = -1.5 * ALPHA * H_over_R * H_over_R * omegaK * R_cyl;
        
        {
          real bracket = 0.5 - 2.75 * H_over_R * H_over_R + 0.5 * R_cyl / r_sph;
          vphi_g = vK * sqrt(bracket);
        }
        // vR_g = 0;
        // vphi_g = vK;

        // Dust Velocities (Cylindrical)
        eps_tot = 1.0 + EPSILON;
        eta = 0.5 * H_over_R * H_over_R * dlnP_dlnR;
        St = 1.0 / INVSTOKES1;
        vR_d = (2.0 / (St + eps_tot * eps_tot / St)) * eta * vK;
        // vR_d = 0;
        vphi_d = (1.0 + (eps_tot * eta) / (eps_tot * eps_tot + St * St)) * vK;

        // --- 3. 速度投影 (Cylindrical -> Spherical) ---
        // 论文 IC 假设 v_z_cyl = 0
        // v_r     = v_R * sin(theta) + v_z * cos(theta) = v_R * sin(theta)
        // v_theta = v_R * cos(theta) - v_z * sin(theta) = v_R * cos(theta)
        
        real vr_final, vtheta_final, vphi_final;

        if (Fluidtype == GAS) {
           rho[l] = rhog;
           // 绝热能量
           real cs2 = cs_loc * cs_loc;
           #ifdef ADIABATIC
             e[l] = rhog * cs2 / (GAMMA - 1.0);
           #else
             e[l] = cs_loc;
           #endif
           
           // 投影气体速度
           vr_final     = vR_g * sin(theta);
           vtheta_final = vR_g * cos(theta);
           vphi_final   = vphi_g;
        } 
        else { // DUST
           rho[l] = rhod;
           e[l] = 0.0;
           
           // 投影尘埃速度
           vr_final     = vR_d * sin(theta);
           vtheta_final = vR_d * cos(theta);
           vphi_final   = vphi_d;
        }

        // 写入场并转换到旋转坐标系
        vr[l] = vr_final;
        vtheta[l] = vtheta_final;
        vphi[l] = vphi_final - OMEGAFRAME * R_cyl; // 注意这里减去的是 R_cyl * Omega
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
