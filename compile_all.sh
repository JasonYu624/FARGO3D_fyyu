#!/bin/bash

cd /home/fyyu/projects/def-evelee/fyyu/Dust_trap/fargo3d

module load cuda/12.6

all_setups="B_a_3D B_b_3D B_c_3D C_a_3D C_b_3D C_c_3D B_a_3D_gas B_b_3D_gas B_c_3D_gas C_a_3D_gas C_b_3D_gas C_c_3D_gas"

for setup in $all_setups; do
    echo "=== Building $setup ==="
    make mrproper > /dev/null 2>&1
    make SETUP=$setup UNITS=0 RESCALE=0 GPU=1 PARALLEL=1 MPICUDA=1 > /dev/null 2>&1
    if [ -f fargo3d ]; then
        cp fargo3d setups/$setup/fargo3d
        echo "  OK: copied to setups/$setup/fargo3d"
    else
        echo "  FAILED"
    fi
done

echo "=== All done ==="