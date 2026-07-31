#!/bin/bash
# Build the paired gas/dust executables for the B_c no-diff open-inner control.
set -euo pipefail

cd /home/fyyu/projects/def-evelee/fyyu/Dust_trap/fargo3d

module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

for setup in B_c_nodiff_openinner_gas B_c_nodiff_openinner; do
  make mrproper
  # FARGO3D's Python make wrapper does not accept GNU make's -j flag.
  make SETUP="$setup" UNITS=0 RESCALE=0 GPU=1 PARALLEL=1 MPICUDA=1
  test -x fargo3d
  cp -f fargo3d "setups/$setup/fargo3d"
  sha256sum "setups/$setup/fargo3d" > "setups/$setup/fargo3d.sha256"
done

echo '=== B_c_nodiff_openinner build complete ==='
