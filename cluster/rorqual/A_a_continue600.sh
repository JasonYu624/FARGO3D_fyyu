#!/bin/bash
#SBATCH --job-name=Aa-cont600
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=slurm-%x-%j.out

repo_root=${FARGO_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the FARGO3D repository root.}}
if [[ ! -f "$repo_root/cluster/run_A_a_continue600.sh" ]]; then
  echo "ERROR: repository root is not valid: $repo_root" >&2
  exit 2
fi
source "$repo_root/cluster/run_A_a_continue600.sh" "$repo_root" "$repo_root/cluster/profiles/rorqual-h100.sh"
