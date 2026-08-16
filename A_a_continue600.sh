#!/bin/bash
#SBATCH --job-name=A_a_cont600
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem-per-cpu=20G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Slurm executes a spooled copy of this script.  Use the submit directory (or
# an explicit override) rather than BASH_SOURCE[0], which would point into
# node-local spool storage at runtime.
repo_root=${FARGO_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the FARGO3D repository root.}}
if [[ ! -f "$repo_root/setups/A_a_3D/A_a_3D_continue600.par" ]]; then
  echo "ERROR: repository root is not valid: $repo_root" >&2
  exit 2
fi
cd "$repo_root"

# Restart from the newest complete post-1500P output rather than an assumed
# number.  This preserves partial progress after a Slurm node failure while
# never allowing a continuation to overwrite already-written snapshots.
minimum_restart_output=300
restart_output=''
target_output=600
output_dir=outputs/A_a_3D
parameter_file=setups/A_a_3D/A_a_3D_continue600.par

echo "Continuation job started on $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: ${SLURM_JOB_ID}"
# FARGO3D interprets Ntot as the absolute number of DT intervals.  Therefore
# the continuation must use Ntot=60000 (rather than an additional 30000) to
# end at output 600.  Ninterm=100 gives a fixed five-orbit cadence throughout
# the complete 0--3000P trajectory.  Do not depend on the original parameter
# file here: the continuation parameter file is self-contained.
ninterm=$(awk '$1 == "Ninterm" {print $2; exit}' "$parameter_file")
ntot=$(awk '$1 == "Ntot" {print $2; exit}' "$parameter_file")
dt=$(awk '$1 == "DT" {print $2; exit}' "$parameter_file")

if [[ ! "$ninterm" =~ ^[0-9]+$ ]] ||
   [[ ! "$ntot" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse integer Ninterm/Ntot values." >&2
  exit 1
fi

if [[ "$ninterm" -ne 100 ]] ||
   [[ $((ntot / ninterm)) -ne "$target_output" ]] ||
   [[ $((ntot % ninterm)) -ne 0 ]]; then
  echo "ERROR: continuation cadence is inconsistent with the requested restart." >&2
  echo "  expected Ninterm=100 and Ntot=${target_output}*Ninterm; got Ninterm=${ninterm}, Ntot=${ntot}" >&2
  exit 1
fi

echo "Cadence validated: DT=${dt}, Ninterm=${ninterm} (5P/output), outputs 0--${target_output}."

required_fields=(
  gasdens
  gasenergy
  gasvx
  gasvy
  gasvz
  dust1dens
  dust1vx
  dust1vy
  dust1vz
  summary
)

mapfile -t candidate_outputs < <(
  find "$output_dir" -maxdepth 1 -type f -name 'summary*.dat' -printf '%f\n' |
    sed -n 's/^summary\([0-9][0-9]*\)\.dat$/\1/p' |
    sort -nr
)

for candidate in "${candidate_outputs[@]}"; do
  if (( candidate < minimum_restart_output || candidate >= target_output )); then
    continue
  fi

  complete=yes
  for field in "${required_fields[@]}"; do
    if [[ ! -s "${output_dir}/${field}${candidate}.dat" ]]; then
      complete=no
      break
    fi
  done

  if [[ "$complete" == yes ]]; then
    restart_output=$candidate
    break
  fi
done

if [[ -z "$restart_output" ]]; then
  echo "ERROR: no complete restart output in [${minimum_restart_output}, $((target_output - 1))]." >&2
  exit 1
fi

echo "Restart selected: output ${restart_output}; target: output ${target_output}."

if [[ ! -s "${output_dir}/planet0.dat" ]]; then
  echo "ERROR: missing planet restart history ${output_dir}/planet0.dat" >&2
  exit 1
fi

if [[ -e "${output_dir}/gasdens${target_output}.dat" ]] ||
   [[ -e "${output_dir}/dust1dens${target_output}.dat" ]]; then
  echo "ERROR: target output ${target_output} already exists; refusing to overwrite." >&2
  exit 1
fi

echo "Restart checkpoint validated."
test -x setups/A_a_3D/fargo3d
sha256sum "$0" setups/A_a_3D/fargo3d "${parameter_file}"
df -h "${output_dir}"

module purge
module load StdEnv/2023
module load cuda/12.6
module load openmpi/4.1.5

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true
srun nvidia-smi

# Deliberately omit -p: the PostRestartHook releases/reset dust only at output 20.
mpirun -np 4 ./setups/A_a_3D/fargo3d \
  -S "${restart_output}" \
  "${parameter_file}"

sync

if [[ ! -s "${output_dir}/summary${target_output}.dat" ]]; then
  echo "ERROR: continuation ended without summary${target_output}.dat" >&2
  exit 1
fi

echo "Continuation reached output ${target_output}."
echo "Continuation job finished on $(date)"
