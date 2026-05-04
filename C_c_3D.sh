#!/bin/bash
#SBATCH --job-name=C_c_3D
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem-per-cpu=20G

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOB_ID"

module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

cd /home/fyyu/projects/def-evelee/fyyu/Dust_trap/fargo3d

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true
srun nvidia-smi

mpirun -np 4 ./setups/C_c_3D_gas/fargo3d setups/C_c_3D_gas/C_c_3D_gas.par

sync

rm -f outputs/C_c_3D/*_2d.dat

if ls outputs/C_c_3D/*_2d.dat >/dev/null 2>&1; then
  echo "ERROR: still exists outputs/C_c_3D/*_2d.dat after rm"
  ls -l outputs/C_c_3D/*_2d.dat || true
  exit 1
fi

mpirun -np 4 ./setups/C_c_3D/fargo3d -p -S 20 setups/C_c_3D/C_c_3D.par

echo "Job finished on $(date)"
