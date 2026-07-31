#!/bin/bash
# Shared body for the Fir/Rorqual H100 control jobs.  It is sourced by a
# Slurm submission script so that the module function stays available.

set -euo pipefail

model=${1:?usage: source run_control.sh MODEL PROFILE}
profile=${2:?usage: source run_control.sh MODEL PROFILE}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$profile"

case "$model" in
  F_a)
    gas_setup=F_a_3D_gas
    dust_setup=F_a_3D
    output_name=F_a_3D
    restart_number=20
    final_output=300
    model_description='F_a: q=2e-4, alpha=1e-3, St=0.1'
    ;;
  B_c_nodiff_openinner)
    gas_setup=B_c_nodiff_openinner_gas
    dust_setup=B_c_nodiff_openinner
    output_name=B_c_nodiff_openinner
    restart_number=1
    final_output=15
    model_description='B_c no-diffusion: open inner dust boundary, dust damping off'
    ;;
  *)
    echo "ERROR: unsupported control model: $model" >&2
    exit 2
    ;;
esac

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

# The default is cluster-local scratch so Fir and Rorqual cannot accidentally
# write to the same output directory.  Export FARGO_OUTPUT_ROOT before sbatch
# to use a durable project filesystem instead.
output_root=${FARGO_OUTPUT_ROOT:-"$SCRATCH/FARGO3D_fyyu/$FARGO_CLUSTER_NAME"}
output_target="$output_root/$output_name"
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

echo "=== $model on $FARGO_CLUSTER_NAME ===" | tee "$output_target/launch.log"
echo "Started: $(date --iso-8601=seconds)" | tee -a "$output_target/launch.log"
echo "Job: $SLURM_JOB_ID; host: $(hostname)" | tee -a "$output_target/launch.log"
echo "$model_description" | tee -a "$output_target/launch.log"
echo "Output target: $output_target" | tee -a "$output_target/launch.log"
srun --ntasks=1 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee "$output_target/gpu_info.txt"

{
  echo "cluster=$FARGO_CLUSTER_NAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "submitted_from=$SLURM_SUBMIT_DIR"
  echo "repo_root=$repo_root"
  echo "output_target=$output_target"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_begin:"
  git status --short
  echo "modules:"
  module list
} > "$output_target/provenance.txt" 2>&1

build_setup() {
  local setup=$1
  echo "=== Building $setup for $FARGO_CUDA_ARCH ===" | tee -a "$output_target/launch.log"
  make mrproper
  # FARGO3D's make wrapper does not accept GNU make's -j flag.  Build each
  # target serially because they share bin/ intermediates.
  make SETUP="$setup" UNITS=0 RESCALE=0 GPU=1 PARALLEL=1 MPICUDA=1 \
    CUDAOPT_LINUX="-O3 -w -arch=$FARGO_CUDA_ARCH" 2>&1 | tee "$output_target/build_${setup}.log"
  test -x fargo3d
  install -m 0755 fargo3d "setups/$setup/fargo3d"
  sha256sum "setups/$setup/fargo3d" > "setups/$setup/fargo3d.sha256"
}

build_setup "$gas_setup"
build_setup "$dust_setup"

sha256sum \
  "setups/$gas_setup/$gas_setup.par" \
  "setups/$dust_setup/$dust_setup.par" \
  "setups/$dust_setup/$dust_setup.bound.1" \
  "setups/$gas_setup/fargo3d" \
  "setups/$dust_setup/fargo3d" \
  > "$output_target/control_inputs.sha256"

echo "=== Gas relaxation ===" | tee -a "$output_target/launch.log"
mpirun -np "$SLURM_NTASKS" "./setups/$gas_setup/fargo3d" "setups/$gas_setup/$gas_setup.par" \
  2>&1 | tee "$output_target/gas_run.log"

required_gas_fields=(gasdens gasenergy gasvx gasvy gasvz dust1dens dust1vx dust1vy dust1vz)
for field in "${required_gas_fields[@]}"; do
  checkpoint="$output_link/${field}${restart_number}.dat"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: missing or empty gas-stage checkpoint $checkpoint" >&2
    exit 4
  fi
done

# The merged dust restart must not inherit auxiliary 2-D files.
rm -f "$output_link"/*_2d.dat
if compgen -G "$output_link/*_2d.dat" > /dev/null; then
  echo "ERROR: auxiliary 2-D files remain after cleanup." >&2
  exit 4
fi

echo "=== Dust release and evolution ===" | tee -a "$output_target/launch.log"
mpirun -np "$SLURM_NTASKS" "./setups/$dust_setup/fargo3d" -p -S "$restart_number" \
  "setups/$dust_setup/$dust_setup.par" 2>&1 | tee "$output_target/dust_run.log"

if [[ ! -s "$output_link/summary${final_output}.dat" ]]; then
  echo "ERROR: $model ended without summary${final_output}.dat." >&2
  exit 5
fi

echo "Completed: $(date --iso-8601=seconds)" | tee -a "$output_target/launch.log"
echo "Final output: summary${final_output}.dat" | tee -a "$output_target/launch.log"
