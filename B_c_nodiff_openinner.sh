#!/bin/bash
#SBATCH --job-name=Bc-nodiff-open
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem-per-cpu=20G

# Fresh B_c no-diffusion control: dust Stockholm damping off and Athena-like
# one-way open dust boundary at Ymin.  The gas-only pre-relaxation reaches
# 100P, followed by a dust restart from output 1 through 1500P.
set -euo pipefail

cd /home/fyyu/projects/def-evelee/fyyu/Dust_trap/fargo3d

module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true

OUTPUT_LINK=outputs/B_c_nodiff_openinner
OUTPUT_TARGET=/home/fyyu/projects/def-rbdong/fyyu/fargo3d_outputs/B_c_nodiff_openinner

# Do not overwrite an existing control.  The output must reside in the
# def-rbdong allocation, which has sufficient room for this new run.
if [[ -e "$OUTPUT_LINK" || -L "$OUTPUT_LINK" || -e "$OUTPUT_TARGET" ]]; then
  echo "ERROR: control output already exists: $OUTPUT_LINK / $OUTPUT_TARGET" >&2
  exit 2
fi
mkdir -p "$OUTPUT_TARGET"
ln -s "$OUTPUT_TARGET" "$OUTPUT_LINK"

for setup in B_c_nodiff_openinner_gas B_c_nodiff_openinner; do
  test -x "setups/$setup/fargo3d" || {
    echo "ERROR: missing compiled executable setups/$setup/fargo3d" >&2
    exit 3
  }
done

sha256sum \
  setups/B_c_nodiff_openinner/fargo3d \
  setups/B_c_nodiff_openinner/B_c_nodiff_openinner.par \
  setups/B_c_nodiff_openinner/B_c_nodiff_openinner.bound.1 \
  > "$OUTPUT_TARGET/control_inputs.sha256"

echo "Job started on $(date)"
echo "SLURM_JOBID: ${SLURM_JOB_ID}"
srun nvidia-smi

mpirun -np 4 ./setups/B_c_nodiff_openinner_gas/fargo3d \
  setups/B_c_nodiff_openinner_gas/B_c_nodiff_openinner_gas.par

sync
rm -f "$OUTPUT_LINK"/*_2d.dat

mpirun -np 4 ./setups/B_c_nodiff_openinner/fargo3d -p -S 1 \
  setups/B_c_nodiff_openinner/B_c_nodiff_openinner.par

if [[ ! -s "$OUTPUT_LINK/summary15.dat" ]]; then
  echo "ERROR: control did not reach output 15 (1500P)." >&2
  exit 4
fi

echo "Job finished on $(date)"
