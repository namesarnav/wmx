# Troubleshooting

**`bad config: [...]` / `TypeError: unexpected keyword`.** A field is missing, unknown, or out of range. Every field must be present (docs/CONFIG_REFERENCE.md). Sweep `set`/`grid` keys must be real field names.

**Tests fail before any sweep.** Do not run sweeps until `poetry run pytest -q` passes. These tests are the CONTEXT.md §7 sanity checks:
- `test_sanity.py::test_6` fails: the optimizer is broken. Never tune it away.
- `test_sanity.py::test_2` fails: evaluation is broken. Never tune it away.

**MPS vs CUDA.**
- `torch.quantile` runs on CPU on purpose: it is unsupported or limited on MPS.
- If an op is missing on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` or `device: cpu`.
- bf16 (Sokoban LM) needs CUDA or recent MPS. Use CUDA for Sokoban.
- `device: auto` picks cuda, then mps, then cpu.

**Many parallel workers are slow.** `sweep.py --workers K` sets `OMP/MKL/OPENBLAS_NUM_THREADS` and torch threads to #CPU/K. Don't set them higher yourself. For GPU sweeps, `scripts/run/*.sh` defaults to one worker per GPU. Several workers on one GPU share memory: raise `WORKERS` only if the models are small.

**A run waits forever at the start.** It is waiting on the cache lock (`results/models/**/*.lock`) while another run trains the same world model. That is expected; use `--group` to avoid idle waits. If a node was killed, the OS has already released the lock. Delete a stale `*.tmp*` file if you see one.

**Crafter.**
- `crafter.Env` signature, the old-gym `step` and `info["discount"]` are marked VERIFY in `envs/crafter.py`.
- Rendering needs no display: Crafter renders with numpy/PIL.
- If the DreamerV3 replay converter finds no chunk streams, check the replay dir and file names (`scripts/convert_dreamer_replay.py`).
- The replay must hold ≥ 1M steps for the grid. Make sure DreamerV3 does not evict chunks (replay size ≥ run steps).

**OOM on large RSSMs.** Size l at batch 16 × 64 fits an H100 (estimate). Otherwise lower `batch` or `seq_len` (this changes the cache key), or use `--workers 1`. Evaluation episodes run in lockstep: lower `eval_episodes` or `imag_starts` if evaluation OOMs.

**Minari dataset names.** Ids are `mujoco/<env>/medium-v0` for minari ≥ 0.5. Older versions used `hopper-medium-v2`-style ids. `minari list remote` shows what exists. Datasets need the `minari[hf]` extra to download. The env version must match (`*-v5`); a dataset/env action-type mismatch raises in `train/vector.py`.

**Push-T.**
- `gym_pusht/PushT-v0` with `obs_type: state`.
- Friction is set on the pymunk shapes after each reset (VERIFY in `envs/vector.py`).
- If the attribute differs, print `env.unwrapped.__dict__.keys()`.

**Qwen memory / speed.**
- The 0.5B model fine-tunes in fp32 with Adam (~8 GB).
- Generation batch is `gen_batch`: lower it on OOM, raise it on H100 for speed.
- `invalid_frac` near 1 means the fine-tune has not converged: raise `wm_steps`.
- Qwen has no pad token; eos is used and masked.

**True return never falls / falls from d = 0.** See the bug signals in docs/EXPERIMENTS.md:
- Never falls: check the fill rule, data coverage, `ac_steps`, the entropy bonus (too large keeps d small), and `kl_coef`.
- Falls from d = 0: check π₀ (BC), and that d_real ≈ 0 at step 0.

**`analyze.py`: "no completed runs".** Only runs with a `done` file are read. A crashed run is retried by re-running its sweep.
