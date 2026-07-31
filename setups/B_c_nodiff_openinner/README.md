# B_c no-diffusion, open-inner dust-boundary control

This is a fresh Model-B `B_c` (`St=0.01`, `alpha=1e-2`) control made from `B_c_nodiff`.

- Gas parameters, mesh, planet, gas damping, gas initial relaxation, and dust physics are unchanged.
- `DUST_STOCKHOLM_DAMPING=0` disables the dust relaxation branch of Stockholm damping while retaining gas damping.
- At `Ymin`, dust density and tangential velocities have zero-gradient ghost values; radial dust velocity is limited to `v_R <= 0`, matching the one-way inner `DUST_OPEN_BC` used by the Athena++ comparison run.
- The outer dust boundary remains the legacy analytic NSH condition. This intentionally isolates the suspected inner-reservoir effect; it is not yet a full two-radial-boundary Athena++ match.
- Both stages use `Ninterm=2000`, so output 1 is 100 orbits. The dust stage restarts with `-p -S 1`, writes every 100 orbits, and reaches output 15 at 1500 orbits. Keeping the two cadences equal is required because FARGO restores the step number as `restart_number * Ninterm`.

Build with `compile_B_c_nodiff_openinner.sh`; submit `B_c_nodiff_openinner.sh` only after the build and output-path checks pass.
