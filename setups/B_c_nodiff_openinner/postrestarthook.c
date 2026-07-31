#include "fargo3d.h"
extern int NbRestart;

void PostRestartHook() {
    if (NFLUIDS < 2) return;

    // Both stages use NINTERM=2000, so output 1 is 100 planetary orbits.
    // FARGO restores begin_i as NbRestart * NINTERM; keep their cadences equal.
    if (NbRestart != 1) {
        if (CPU_Master) {
            printf("[Hook] Current Restart = %d. Waiting for output 1 to release dust. Skipping...\n", NbRestart);
        }
        return;
    }

    if (CPU_Master) {
        printf("\n=================================================================\n");
        printf(">>> OUTPUT 1 (100 ORBITS) REACHED: RESETTING DUST TO STEADY STATE ! <<<\n");
        printf("=================================================================\n\n");
    }

    // --- 2. 获取 CPU 端指针 ---
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
    real omegaK, vK, H_over_R, rhog_init, rhod_init;
    real vR_d, vphi_d;
    real eps_tot, eta, St;
    const real dlnP_dlnR = -2.75;

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
                // 修改为：直接读取 Fluid 0 (Gas) 当前在 CPU 里的真实演化密度
                real *rho_g = Fluids[0]->Density->field_cpu;
                real current_gas_density = rho_g[idx];

                if (!(current_gas_density >= 0.0 && current_gas_density <= 1e30)) {
                    current_gas_density = 1e-12;
                }

                // 尘埃密度严格跟随气体的坑洼形态
                rhod_init = EPSILON * current_gas_density;
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
