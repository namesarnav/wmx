# Metrics

Every quantity below maps to one implementation. No task has its own copy.

## d: distance from π₀ (CONTEXT.md §3.1)
The initial-state and transition terms cancel, leaving

  KL(p_π ‖ p_π₀) = E_{τ∼p_π} [ Σ_t KL(π(·|s_t) ‖ π₀(·|s_t)) ],   d = √KL.

| Setting | d_imag (states from the model) | d_real (states from the real env) | Code |
|---|---|---|---|
| FrozenLake | exact: Σ_s n_π(s) KL_s, with n_π the expected undiscounted visits under P̂ | same with P | `metrics.evaluate`, `metrics.kl_d` |
| Crafter / MuJoCo (AC) | imagined full episodes from real first observations, Σ_t w_t KL_t, w_t = Π ĉ (model survival) | real episodes, π acting on filtered model states | `train/imagination.evaluate_imag`, `evaluate_real`, `metrics.d_from_kl` |
| MuJoCo (CEM/MPPI) | same, with per-state KL(N(μ,σ) ‖ π₀(·|s)) of the planner's first-step Gaussian | same | `policies/planning.plan` |
| FrozenLake best-of-K | exact, from the closed-form best-of-K policy | exact | `policies/tabular.best_of_k` |
| Sokoban GRPO | imagined boards in the text WM, KL per game action | real boards | `train/sokoban.kl_sums`, `evaluate_real` |
| Sokoban best-of-K | √(log K − (K−1)/K) (trajectory-level selection) | same | `metrics.best_of_k_kl` |

Per-state KL is `torch.distributions.kl_divergence` (categorical or diagonal Gaussian). π₀ must be > 0 everywhere,
or have std > 0 (smoothing `pi0_eps`), otherwise d is infinite. `kl_step_*` logs the KL per step.

## δ: one-step error (CONTEXT.md §3.2)
| Setting | δ | Code |
|---|---|---|
| FrozenLake | exact Σ_a π(a\|s) · ½‖P̂(·\|s,a) − P(·\|s,a)‖₁, weighted quantiles over π's imagined visits | `metrics.delta` |
| Sokoban | exact 1 − p_WM(true outcome \| board, action) on π's imagined boards (deterministic truth) | `train/sokoban.delta_row` |
| Crafter, RSSM | proxy: mean-over-latents TV(prior, posterior) on π's real states | `WorldModel.observe_step` |
| Ensemble | proxy: RMS normalized error of the mean prediction on π's real states | `Ensemble.observe_step` |
| TD-MPC2 | proxy: RMS latent consistency error ‖dyn(z,a) − enc(o′)‖ on π's real states | `TDMPC2.observe_step` |

Logged as `delta_q{q}` for each q in `delta_quantiles`. `reward_error` = mean |r̂ − r| on the same states.
`holdout_error` (vector runs) is the one-step obs prediction error, mean of ((ô − o)/std)², on held-out episodes. It is comparable across architectures.

## Peak d* (CONTEXT.md §3.3)
- **fit:** the argmax of the fitted form (closed form below).
- **empirical:** the argmax of a centered moving average (`smooth_window` checkpoints) of improvement vs d.

CIs come from a percentile bootstrap over seeds (`analysis/fits.bootstrap_ci`, `analysis/summary.cell_summary`).

## Functional forms (Q1)
The y-axis is the true-return improvement over π₀ (row 0), optionally divided by (ceiling − π₀). Both forms are linear in (α, β) and are fitted exactly by least squares (`analysis/fits.fit_form`). Goodness of fit is R².

| Form | R(d) | d* | R(d*) |
|---|---|---|---|
| RL in imagination | d(α − β log d) | exp(α/β − 1) | β d* |
| sampling-based planning | d(α − β d) | α / (2β) | α² / (4β) |

The form used for a run's d* is set by its optimizer: `bok`/`plan` → planning form, otherwise RL. Both forms are always reported.

## Scaling laws (Q2)
Per cell, the analysis computes d*, peak, gap and α, β. The gap is imagined − true return at the checkpoint nearest d*.
- **Single-factor power laws:** y = a x^b (`power_law`).
- **Joint law:** log y = c + Σ_f b_f log x_f over f ∈ {N, M, H} (`loglinear_fit`), fitted on non-held-out cells.
- **Extrapolation:** relative error on held-out cells, and whether the prediction falls inside the cell's bootstrap CI.

## Safe horizon (Q3)
- **Onset:** the first d at which (imagined − true) exceeds its d=0 value by ε.
- **Predictors:** √(ε/δ), H and 1/(1−γ). δ is taken at the `delta_checkpoint`.
- **Comparison:** Spearman ρ on log–log axes (`analyze.py q3`).

## Disagreement (secondary)
The disagreement onset is the first d at which the ensemble disagreement on imagined states exceeds (1+κ) times its d=0 value. It is compared against d* (`analyze.py disagreement`).
