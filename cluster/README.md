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
