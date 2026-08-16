#include "fargo3d.h"
extern int NbRestart;

#ifndef DUST_HRATIO
#define DUST_HRATIO 1.0
#endif

void PostRestartHook() {
    if (NFLUIDS < 2) return;

    // --- 1. 绝对安全的释放条件：直接判断输出编号 ---
    // 你的 100 Orbits 对应的是第 20 个 Output
    if (NbRestart != 20) {
        if (CPU_Master) {
            printf("[Hook] Current Restart = %d. Waiting for Output 20 to release dust. Skipping...\n", NbRestart);
        }
        return;
    }

    if (CPU_Master) {
        printf("\n=================================================================\n");
        printf(">>> OUTPUT 20 REACHED: RESETTING DUST TO STEADY STATE ! <<<\n");
        printf("=================================================================\n\n");
    }

    // --- 2. 获取 CPU 端指针 ---
    real *rho_g = Fluids[0]->Density->field_cpu;
    real *rho_d = Fluids[1]->Density->field_cpu;
    real *vx_d  = Fluids[1]->Vx->field_cpu;
    real *vy_d  = Fluids[1]->Vy->field_cpu;

#ifdef Z
    real *vz_d  = Fluids[1]->Vz->field_cpu;
#else
    real *vz_d  = NULL;
#endif

#ifdef ADIABATIC
    real *e_d = NULL;
    if (Fluids[1]->Energy != NULL) {
        e_d = Fluids[1]->Energy->field_cpu;
    }
#endif

    // --- 3. 循环与重置 ---
    int i, j, k;
    real R_cyl, r_sph, theta;
    real omegaK, vK, H_over_R, rhod_init;
    real vR_d, vphi_d;
    real eps_tot, eta, St;
    const real dlnP_dlnR = -2.75;
    const real dust_hratio = DUST_HRATIO;

    int Nz_loop = 1;
#ifdef Z
    Nz_loop = Nz + 2 * NGHZ;
#endif

    int pitch = Pitch_cpu;
    int stride = Stride_cpu;

    for (k = 0; k < Nz_loop; k++) {
        for (j = 0; j < Ny + 2 * NGHY; j++) {
            for (i = 0; i < Nx + 2 * NGHX; i++) {

                int idx = i + j * pitch + k * stride;

#ifdef Z
                r_sph = Ymed(j);
                theta = Zmed(k);
                R_cyl = fabs(r_sph * sin(theta));
#else
                R_cyl = Ymed(j);
                r_sph = R_cyl;
                theta = 0.5 * M_PI;
#endif
                if (R_cyl < 1e-6) R_cyl = 1e-6;

                omegaK = sqrt(G * MSTAR / (R_cyl * R_cyl * R_cyl));
                vK = omegaK * R_cyl;
                H_over_R = ASPECTRATIO * pow(R_cyl / R0, FLARINGINDEX);

                // --- 密度重置 ---
                // Keep the current gas gaps/cavities, but normalize dust using
                // the current local gas depletion relative to the analytic gas profile.
                real current_gas_density = rho_g[idx];
                if (!(current_gas_density >= 0.0 && current_gas_density <= 1e30)) {
                    current_gas_density = 1e-12;
                }

                {
                    real rhog_analytic = pow(R_cyl / R0, -2.25) *
                                         exp((R_cyl / (ASPECTRATIO * ASPECTRATIO *
                                               pow(R_cyl / R0, 2.0 * FLARINGINDEX))) *
                                             (1.0 / r_sph - 1.0 / R_cyl));
                    if (!(rhog_analytic > 1e-30)) rhog_analytic = 1e-30;

                    {
                        real depletion = current_gas_density / rhog_analytic;
                        if (!(depletion >= 0.0 && depletion <= 1e30)) depletion = 0.0;

                        rhod_init = (EPSILON * depletion / dust_hratio) *
                                    pow(R_cyl / R0, -2.25) *
                                    exp((R_cyl / (dust_hratio * dust_hratio *
                                          ASPECTRATIO * ASPECTRATIO *
                                          pow(R_cyl / R0, 2.0 * FLARINGINDEX))) *
                                        (1.0 / r_sph - 1.0 / R_cyl));
                    }
                }
                if (rhod_init < 1e-12) rhod_init = 1e-12;

                rho_d[idx] = rhod_init;

                // --- 速度重置 (Nakagawa) ---
                eps_tot = 1.0 + EPSILON;
                eta = 0.5 * H_over_R * H_over_R * dlnP_dlnR;
                St = 1.0 / INVSTOKES1;

                vR_d = (2.0 / (St + eps_tot * eps_tot / St)) * eta * vK;
                vphi_d = (1.0 + (eps_tot * eta) / (eps_tot * eps_tot + St * St)) * vK;

                vx_d[idx] = vphi_d - OMEGAFRAME * R_cyl;
                vy_d[idx] = vR_d * sin(theta);
                if (vz_d) vz_d[idx] = vR_d * cos(theta);

                // --- 能量重置 ---
#ifdef ADIABATIC
                if (e_d != NULL) {
                    e_d[idx] = 1.0e-20;
                }
#endif
            }
        }
    }

    // --- 4. 强制同步到 GPU ---
    OUTPUT(Fluids[1]->Density);
    OUTPUT(Fluids[1]->Vx);
    OUTPUT(Fluids[1]->Vy);
#ifdef Z
    OUTPUT(Fluids[1]->Vz);
#endif
#ifdef ADIABATIC
    if (Fluids[1]->Energy != NULL) {
        OUTPUT(Fluids[1]->Energy);
    }
#endif

    if (CPU_Master) printf(">>> Hook Success: Healthy Dust fields synced to GPU! <<<\n");
}
