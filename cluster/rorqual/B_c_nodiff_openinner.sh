#!/bin/bash
#SBATCH --job-name=Bc-nodiff-open
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=slurm-%x-%j.out

repo_root=${FARGO_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the FARGO3D repository root.}}
if [[ ! -f "$repo_root/cluster/run_control.sh" ]]; then
  echo "ERROR: repository root is not valid: $repo_root" >&2
  exit 2
fi
source "$repo_root/cluster/run_control.sh" B_c_nodiff_openinner "$repo_root/cluster/profiles/rorqual-h100.sh"
