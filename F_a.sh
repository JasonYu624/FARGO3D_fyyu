#!/bin/bash
#SBATCH --job-name=F_a
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/fyyu/projects/def-evelee/fyyu/Dust_trap/fargo3d

output_dir=outputs/F_a_3D

echo "Model F job started on $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: ${SLURM_JOB_ID}"
echo "Model parameters: q=2e-4, alpha=1e-3, St=0.1"

if [[ -e "${output_dir}/gasdens0.dat" ]] ||
   [[ -e "${output_dir}/dust1dens0.dat" ]]; then
  echo "ERROR: ${output_dir} already contains output 0; refusing to overwrite." >&2
  exit 1
fi

module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true
srun nvidia-smi

# Stage 1: relax the gas for 100 planetary orbits (output 0 -> 20).
mpirun -np 4 \
  ./setups/F_a_3D_gas/fargo3d \
  setups/F_a_3D_gas/F_a_3D_gas.par

sync

required_gas_fields=(gasdens gasenergy gasvx gasvy gasvz dust1dens dust1vx dust1vy dust1vz)
for field in "${required_gas_fields[@]}"; do
  checkpoint="${output_dir}/${field}20.dat"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "ERROR: missing or empty gas-stage checkpoint ${checkpoint}" >&2
    exit 1
  fi
done

# The dust-stage merged restart must not see auxiliary 2-D output files.
rm -f "${output_dir}"/*_2d.dat
if compgen -G "${output_dir}/*_2d.dat" >/dev/null; then
  echo "ERROR: auxiliary 2-D files remain after cleanup." >&2
  exit 1
fi

# Stage 2: release dust at output 20 and evolve to output 300 (1500P).
mpirun -np 4 \
  ./setups/F_a_3D/fargo3d \
  -p -S 20 \
  setups/F_a_3D/F_a_3D.par

sync

if [[ ! -s "${output_dir}/summary300.dat" ]]; then
  echo "ERROR: Model F ended without summary300.dat" >&2
  exit 1
fi

echo "Model F reached output 300 (1500P)."
echo "Model F job finished on $(date)"
