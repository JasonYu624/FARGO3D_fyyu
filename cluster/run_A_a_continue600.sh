#!/bin/bash
# Shared body for the portable A_a output-306 -> output-600 continuation.
# It is sourced by the cluster-specific Slurm launchers.

set -euo pipefail

repo_root=${1:?usage: source run_A_a_continue600.sh REPO_ROOT PROFILE}
profile=${2:?usage: source run_A_a_continue600.sh REPO_ROOT PROFILE}
source "$profile"

: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"
: "${SCRATCH:?SCRATCH is not set by the cluster environment.}"
: "${FARGO_CLUSTER_NAME:?cluster profile did not set FARGO_CLUSTER_NAME}"
: "${FARGO_CUDA_MODULE:?cluster profile did not set FARGO_CUDA_MODULE}"
: "${FARGO_MPI_MODULE:?cluster profile did not set FARGO_MPI_MODULE}"
: "${FARGO_CUDA_ARCH:?cluster profile did not set FARGO_CUDA_ARCH}"

if [[ ${SLURM_NTASKS:-0} -ne 4 ]]; then
  echo "ERROR: expected four MPI ranks / H100 GPUs, got SLURM_NTASKS=${SLURM_NTASKS:-unset}." >&2
  exit 2
fi

cd "$repo_root"

setup=A_a_3D
output_name=A_a_3D
restart_output=306
target_output=600
parameter_file="setups/$setup/A_a_3D_continue600.par"
seed_dir=${FARGO_AA_RESTART_SEED:-"$repo_root/restarts/A_a_3D_output306"}
seed_manifest="$seed_dir/A_a_3D_output306.sha256"

# This includes the complete restart state plus the static metadata required by
# FARGO3D's restart reader.  It deliberately excludes all earlier snapshots.
required_seed_files=(
  gasdens306.dat gasenergy306.dat gasvx306.dat gasvy306.dat gasvz306.dat
  dust1dens306.dat dust1vx306.dat dust1vy306.dat dust1vz306.dat
  summary306.dat planet0.dat bigplanet0.dat
  domain_x.dat domain_y.dat domain_z.dat variables.par
)

if [[ ! -f "$parameter_file" || ! -f "setups/$setup/$setup.opt" ]]; then
  echo "ERROR: missing A_a continuation setup in $repo_root." >&2
  exit 2
fi
if [[ ! -f "$seed_manifest" ]]; then
  echo "ERROR: missing restart manifest: $seed_manifest" >&2
  echo "Run cluster/stage_A_a_restart306_to_rorqual.sh from Nibi first." >&2
  exit 2
fi
for file in "${required_seed_files[@]}"; do
  if [[ ! -s "$seed_dir/$file" ]]; then
    echo "ERROR: missing or empty restart seed: $seed_dir/$file" >&2
    exit 2
  fi
done

echo "=== A_a continuation on $FARGO_CLUSTER_NAME ==="
echo "Started: $(date --iso-8601=seconds)"
echo "Job: $SLURM_JOB_ID; host: $(hostname)"
echo "Restart seed: $seed_dir (output $restart_output)"
echo "Target: output $target_output"
(cd "$seed_dir" && sha256sum -c "$(basename "$seed_manifest")")

ninterm=$(awk '$1 == "Ninterm" {print $2; exit}' "$parameter_file")
ntot=$(awk '$1 == "Ntot" {print $2; exit}' "$parameter_file")
if [[ "$ninterm" != 100 || "$ntot" != 60000 ]]; then
  echo "ERROR: expected Ninterm=100 and Ntot=60000; got Ninterm=$ninterm, Ntot=$ntot." >&2
  exit 2
fi

output_root=${FARGO_OUTPUT_ROOT:-"$SCRATCH/FARGO3D_fyyu/$FARGO_CLUSTER_NAME"}
output_target="$output_root/${output_name}_cont600"
output_link="$repo_root/outputs/$output_name"

if [[ -e "$output_link" || -L "$output_link" || -e "$output_target" ]]; then
  echo "ERROR: refusing to overwrite existing output:" >&2
  echo "  link:   $output_link" >&2
  echo "  target: $output_target" >&2
  exit 3
fi

mkdir -p "$output_root" "$(dirname "$output_link")"
mkdir "$output_target"
ln -s "$output_target" "$output_link"

cleanup_failed_launch() {
  status=$?
  echo "ERROR: launch failed with status $status; preserving $output_target for diagnosis." >&2
  exit "$status"
}
trap cleanup_failed_launch ERR

rsync -a --checksum "$seed_dir/" "$output_target/"
(cd "$output_target" && sha256sum -c "$(basename "$seed_manifest")")

{
  echo "cluster=$FARGO_CLUSTER_NAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "submitted_from=$SLURM_SUBMIT_DIR"
  echo "repo_root=$repo_root"
  echo "seed_dir=$seed_dir"
  echo "restart_output=$restart_output"
  echo "target_output=$target_output"
  echo "output_target=$output_target"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_begin:"
  git status --short
} > "$output_target/provenance.txt" 2>&1

module --force purge
module load StdEnv/2023
module load "$FARGO_CUDA_MODULE"
module load "$FARGO_MPI_MODULE"

command -v nvcc
command -v mpirun
nvcc --version
mpirun --version

export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true
export FARGO_CUDA_ARCH
srun --ntasks=1 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee "$output_target/gpu_info.txt"

echo "=== Building $setup for $FARGO_CUDA_ARCH ===" | tee "$output_target/build_${setup}.log"
make mrproper 2>&1 | tee -a "$output_target/build_${setup}.log"
make SETUP="$setup" UNITS=0 RESCALE=0 GPU=1 PARALLEL=1 MPICUDA=1 \
  2>&1 | tee -a "$output_target/build_${setup}.log"
test -x fargo3d
install -m 0755 fargo3d "setups/$setup/fargo3d"
sha256sum \
  "$parameter_file" \
  "setups/$setup/$setup.bound.1" \
  "setups/$setup/fargo3d" \
  > "$output_target/continuation_inputs.sha256"

echo "=== Continuing dust evolution ===" | tee "$output_target/launch.log"
# Do not pass -p: dust was released in the original run and must not be reset.
mpirun -np "$SLURM_NTASKS" "./setups/$setup/fargo3d" -S "$restart_output" "$parameter_file" \
  2>&1 | tee "$output_target/dust_run.log"

if [[ ! -s "$output_target/summary${target_output}.dat" ]]; then
  echo "ERROR: continuation ended without summary${target_output}.dat." >&2
  exit 5
fi

echo "Completed: $(date --iso-8601=seconds)" | tee -a "$output_target/launch.log"
echo "Final output: summary${target_output}.dat" | tee -a "$output_target/launch.log"
