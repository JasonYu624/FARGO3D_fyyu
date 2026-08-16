#!/bin/bash
# Transfer the minimum complete A_a restart state from Nibi to Rorqual.
# Run this interactively on Nibi because Alliance MFA is required for Rorqual.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source_dir=${FARGO_AA_SOURCE_OUTPUT:-/project/6022842/fyyu/fargo3d_outputs/A_a_3D}
remote_host=${FARGO_RORQUAL_HOST:-fyyu@rorqual.alliancecan.ca}
remote_repo=${FARGO_RORQUAL_REPO:-/home/fyyu/links/projects/def-evelee/fyyu/FARGO3D_fyyu}
remote_seed="$remote_repo/restarts/A_a_3D_output306"
manifest_name=A_a_3D_output306.sha256

required_seed_files=(
  gasdens306.dat gasenergy306.dat gasvx306.dat gasvy306.dat gasvz306.dat
  dust1dens306.dat dust1vx306.dat dust1vy306.dat dust1vz306.dat
  summary306.dat planet0.dat bigplanet0.dat
  domain_x.dat domain_y.dat domain_z.dat variables.par
)

for file in "${required_seed_files[@]}"; do
  if [[ ! -s "$source_dir/$file" ]]; then
    echo "ERROR: missing or empty source restart file: $source_dir/$file" >&2
    exit 2
  fi
done

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
manifest="$tmpdir/$manifest_name"
(cd "$source_dir" && sha256sum "${required_seed_files[@]}") > "$manifest"

echo "Staging A_a restart output 306 to $remote_host:$remote_seed"
echo "Source: $source_dir"
echo "Manifest: $manifest"
ssh "$remote_host" "mkdir -p '$remote_seed'"

source_paths=()
for file in "${required_seed_files[@]}"; do
  source_paths+=("$source_dir/$file")
done
rsync -av --partial --progress --checksum \
  "${source_paths[@]}" "$manifest" \
  "$remote_host:$remote_seed/"

ssh "$remote_host" "cd '$remote_seed' && sha256sum -c '$manifest_name'"
echo "Restart seed transferred and checksum-verified on Rorqual."
echo "Next: git pull --ff-only in $remote_repo, then sbatch cluster/rorqual/A_a_continue600.sh"
