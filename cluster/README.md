# Portable Fir/Rorqual H100 launchers

These are self-locating Slurm launchers for the `F_a` and corrected
`B_c_nodiff_openinner` controls. They are intentionally separate from the
older top-level launchers, which contain historical, cluster-specific absolute
paths.

On the target cluster, use a clean clone in a durable filesystem, `cd` to its
repository root, and submit the launcher that matches that cluster. The
launcher uses Slurm's `SLURM_SUBMIT_DIR`; this is necessary because Slurm
executes a spooled copy of the batch script rather than the original file.

```bash
sbatch cluster/fir/F_a.sh
sbatch cluster/fir/B_c_nodiff_openinner.sh
sbatch cluster/rorqual/F_a.sh
sbatch cluster/rorqual/B_c_nodiff_openinner.sh
```

Each job requests four complete H100 GPUs under `def-evelee` and leaves
partition selection to the Alliance scheduler, which then selects a compatible
H100 partition for that account. It loads `StdEnv/2023`, `cuda/12.6`, and
`openmpi/4.1.5`, builds
both paired FARGO3D executables serially on the target compute node with
`sm_90`, and then runs gas relaxation followed by the dust restart. The
checkout can live at a different path on Fir and Rorqual.

By default, results are written to the cluster-local
`$SCRATCH/FARGO3D_fyyu/{fir,rorqual}/<model>` and the repository's ignored
`outputs/<model>` becomes a symlink to that location. To use a durable output
filesystem, set it explicitly when submitting:

```bash
FARGO_OUTPUT_ROOT=/path/with/sufficient/space sbatch cluster/fir/F_a.sh
```

The launchers refuse to overwrite either the output target or its repository
symlink. They preserve build logs, GPU information, the source commit/status,
checksums, and separate gas/dust logs in the output directory. Copy results
out of scratch before that filesystem's retention deadline.

`B_c_nodiff_openinner` uses 100-orbit cadence in both stages. Therefore the
gas relaxation ends at output 1 and the dust release is `-p -S 1`; its 1500P
completion sentinel is `summary15.dat`.

## A_a continuation on Rorqual

The A_a continuation is a separate, restart-only job.  It resumes the
validated output-306 checkpoint (1530P) without `-p`, so it preserves the
already released dust state and stops at output 600 (3000P).  It is intentionally
not handled by `run_control.sh`, because it does not rerun gas relaxation.

From the Nibi repository, stage the minimum restart bundle interactively (the
Rorqual login requires MFA):

```bash
bash cluster/stage_A_a_restart306_to_rorqual.sh
```

Then on Rorqual, update the same Git checkout and submit from its root:

```bash
git pull --ff-only
sbatch --test-only cluster/rorqual/A_a_continue600.sh
sbatch cluster/rorqual/A_a_continue600.sh
```

The launcher checks the transferred SHA256 manifest before and after copying
the checkpoint from the durable checkout to Rorqual scratch.  Its scratch
target is `$SCRATCH/FARGO3D_fyyu/rorqual/A_a_3D_cont600`, while the temporary
repository symlink remains `outputs/A_a_3D` because that is the output path in
the continuation parameter file.  The launcher refuses to overwrite either
location.
