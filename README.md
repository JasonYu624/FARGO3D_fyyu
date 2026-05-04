# FARGO3D (Modified for Athena++ Alignment)

#### A versatile MULTIFLUID HD/MHD code that runs on clusters of CPUs or GPUs, with special emphasis on protoplanetary disks.

**Author**: Fangyuan (Jason) Yu - Princeton University

This is a modified version of FARGO3D aligned with Athena++ `disk_planet_dust_spherical.cpp` for 3D dusty planet-disk simulations (see [Huang, Yu et al. 2025](https://iopscience.iop.org/article/10.3847/1538-4357/addd1f), "Leaky Dust Traps in Planet-embedded Protoplanetary Disks", ApJ 988, 94).

### [Official Documentation](https://fargo3d.github.io/documentation)

### Modifications from Official FARGO3D

This version includes enhancements for improved stability and physical accuracy in 3D dusty protoplanetary disk simulations, aligning with Athena++ best practices.

#### Key Modifications

| Feature | Description | File(s) |
|---------|-------------|---------|
| **TV-Cap (Terminal Velocity Cap)** | Limits dust terminal velocity in low-density high-altitude regions | `setups/*/floor.cu` |
| **VCAP (Velocity Cap)** | Near-planet free-fall velocity protection | `setups/*/floor.cu` |
| **Relative Median Cap** | 6-neighbor median filtering to suppress spike artifacts | `setups/*/floor.cu` |
| **Anomaly Check** | 26-neighbor anomaly detection for bad cell values | `setups/*/floor.cu` |
| **Planetary Diffusivity Reduction** | Reduces diffusion near planet to prevent artificial spreading | `src/dust_diffusion_coefficients.c` |
| **D2G Diffusivity Reduction** | Reduces dust-to-gas diffusivity ratio for stability | `src/dust_diffusion_coefficients.c` |
| **Planetary Viscosity Reduction** | Reduces viscosity near planet | `src/visctensor_sph.c` |
| **Radial + Meridional Damping** | Stockholm-style damping for boundary zones | `src/stockholm.c` |
| **Planetary Accretion** | Mode-2 sink with Athena-style timescale | `setups/*_accret/accretion.c/.cu` |
| **Beta Cooling** | Cooling function for disk thermodynamics | `setups/*/edamp.c/.cu` |

#### New Setups

| Setup | Purpose |
|-------|---------|
| `3D_01rH_gas_vcap_accret` | Stage 1: Gas pre-evolution with stabilization (dust nearly off) |
| `3D_01rH_vcap_accret` | Stage 2: Dust evolution with drag and full stabilization |
| `B_a_3D`, `B_b_3D`, `B_c_3D` | Model B parameter studies (St = 0.1, 0.0316, 0.01; α=10⁻²) |
| `C_a_3D`, `C_b_3D`, `C_c_3D` | Model C parameter studies (St = 0.1, 0.01, 0.001; α=10⁻³) |
| `*_gas` variants | Gas-only versions |

#### Build Commands

```bash
module load cuda/12.6

# Standard build
make mrproper
make SETUP=<setup_name> UNITS=0 RESCALE=0 GPU=1 PARALLEL=1 MPICUDA=1

# Compile all parameter study setups
./compile_all.sh
```

#### Two-Step Running (Recommended)

```bash
# Stage 1: Gas pre-evolution
mpirun -np 4 ./setups/3D_01rH_gas_vcap_accret/fargo3d \
  setups/3D_01rH_gas_vcap_accret/3D_01rH_gas_vcap_accret.par

# Stage 2: Dust evolution (restart from Stage 1)
mpirun -np 4 ./setups/3D_01rH_vcap_accret/fargo3d -p -S 20 \
  setups/3D_01rH_vcap_accret/3D_01rH_vcap_accret.par
```

### Alignment Status with Athena++

- **Basic disk structure, gravity, cooling, diffusion**: Migrated
- **TV-cap, V-cap**: Migrated
- **Planetary accretion (Mode 2)**: Migrated (compiled, pending numerical validation)
- **Dust wave damping**: Partially migrated
- **Relative median cap / anomaly**: Migrated (iterative variant)
- **Multi-species Hratio**: Not yet implemented

### Known Limitations

1. Planetary accretion: Only Mode 2 implemented; momentum diagnostics not yet aligned
2. Strict FOFC (flux-centered scheme) not implemented (not compatible with FARGO structure)
3. Dust inner-edge open boundary not prioritized

### Description of Subdirectories

* ```in/```: Setup parameter files (equivalent to `.par` files in `setup/`)

* ```planets/```: Planet configuration files (modelB_J3e-4.cfg, modelC_J1e-4.cfg, etc.)

* ```scripts/```: Scripts used to build the code

* ```setups/```: Custom setup definitions including:
  - `3D_01rH_vcap/` - Base 3D setups with vcap features
  - `3D_01rH_vcap_accret/` - With planetary accretion (still testing)
  - `B_a_3D/` - Model B parameter studies
  - `C_a_3D/` - Model C parameter studies

* ```src/```: Source files; copied to `setups/` and modified there. The makefile uses `VPATH` to decide search order (setup directory has higher priority than src).

* ```std/```: Standard or default definitions including boundary conditions, units, scaling rules, default setup parameters.

* ```test_suite/```: Test scripts. Run with `make test[name]` for any script in this subdirectory.

* ```utils/```: Utilities to post-process simulation data.

------------------------

### Original FARGO3D Description

Report bugs to the [issues section](https://github.com/FARGO3D/fargo3d/issues) or to the [Google group](https://groups.google.com/forum/#!forum/fargo3d).