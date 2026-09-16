# Modular RD-CrossSDC release

## File layout

```text
train_incremental_rd_crosssdc_modular.py   # training orchestration only
rd_crosssdc/
  exact_losses.py                          # original AVCIL/CrossSDC formulas
  rd_method.py                             # CMR, prototype bank, Trust×Need state
  diagnostics.py                           # RD-specific CSV output
  metrics.py                               # test/per-class metrics
rd_crosssdc_modular_easy2hard_commands.txt # three controlled experiments
verify_exact_crosssdc.py                   # synthetic formula-equivalence check
```

Place the training script and the entire `rd_crosssdc/` directory in the same
folder as the original `train_incremental_crosssdc_z1.py`.

## Three modes

### 1. `--rd_mode crosssdc`

Pure control. It directly calls the original `cross_sdc_z1_loss` and skips:

- prototype construction;
- teacher trust;
- Need EMA;
- CMR forward graph.

Its training loss is

```text
AVCIL + 0.1 CrossSDC-I + 0.3 CrossSDC-C
```

### 2. `--rd_mode crosssdc_cmr`

Uses the exact original CrossSDC-I/C and adds uniform CMR:

```text
AVCIL + 0.1 CrossSDC-I + 0.3 CrossSDC-C + 0.1 CMR
```

No trust or adaptive class weighting is computed.

### 3. `--rd_mode adaptive_crosssdc_cmr`

Keeps original CrossSDC-I, applies normalized Trust×Need weights to the
per-anchor form of the original CrossSDC-C formula, and applies trust-weighted
CMR:

```text
AVCIL + 0.1 CrossSDC-I + 0.3 TD-CrossSDC-C + 0.1 CMR
```

## Important comparability choices

1. Current/old temporal directions remain exactly:
   - current audio -> old visual;
   - old audio -> current visual.
2. The exact control uses the original one-hot `CE_loss`, not
   `torch.nn.functional.cross_entropy`.
3. The exact control uses the original positive-count-normalized CrossSDC-C.
4. CrossSDC is added before attention-score distillation, matching the working
   script.
5. A disabled module is skipped, not evaluated and multiplied by zero.
6. `--dataset` retains the historical CrossSDC identifier; `--experiment_name`
   changes only output directories.

## Output locations

For each `--experiment_name NAME`:

```text
save/NAME/                         # checkpoints
save/fig/NAME/                     # training curves
save/metrics/NAME/                 # per-class metrics
save/metrics/NAME/rd_crosssdc/     # CMR/weight diagnostics
```

## Running

```bash
mkdir -p logs
```

Then run one line at a time from
`rd_crosssdc_modular_easy2hard_commands.txt`, or use it with the existing
Slurm job-array wrapper.
