#!/bin/bash
#SBATCH --job-name=F_a
#SBATCH --account=def-evelee
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=slurm-%x-%j.out

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$repo_root/cluster/run_control.sh" F_a "$repo_root/cluster/profiles/fir-h100.sh"
