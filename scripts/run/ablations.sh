#!/usr/bin/env bash
# Mechanism ablations (Task 5), GPU. Usage: scripts/run/ablations.sh EXPERIMENT
#   policies     behavior policies + ceilings: SAC Hopper (mujoco sac), SAC per variant, PPO CartPole
#   data         collect every variant's dataset with the unmodified env's behavior policy (after `policies`)
#   sticky | termination | cartpole | pusht    the ablation sweeps (after `data`)
#   all          policies data sticky termination cartpole pusht
# Collection settings: N_COLLECT (default 260000 = N 200K + holdout 50K + margin), EPS (default 0.2).
source "$(dirname "$0")/_common.sh"
N=${N_COLLECT:-260000}
EPS=${EPS:-0.2}   # TODO(Arnav): decide the behavior noise
HOPPER=results/baselines/sac_mujoco/env_id=Hopper-v5_seed=0/policy.pt
PUSHT=results/baselines/ablations/ceilings/env_id=gym_pusht-PushT-v0_episode_length=300_sticky=0.0_friction=None_seed=0/policy.pt
CARTPOLE=results/baselines/ablations/ceilings_cartpole/sparse_reward=False_seed=0/policy.pt
collect() { local policy=$1 out=$2; shift 2; [[ -f $out/obs.npy ]] || py scripts/collect.py configs/mujoco/base.yaml "$policy" "$out" "$N" "$EPS" 0 "$@"; }
case ${1:-} in
  policies) sweep configs/baselines/sac_mujoco.yaml gpu
            sweep configs/ablations/ceilings.yaml gpu
            sweep configs/ablations/ceilings_cartpole.yaml cpu ;;
  data)
    for s in 0.0 0.1 0.25 0.5; do collect $HOPPER "data/hopper-sticky-$s" env_id=Hopper-v5 sticky=$s; done
    collect $HOPPER data/hopper-term env_id=Hopper-v5
    collect $HOPPER data/hopper-noterm env_id=Hopper-v5 'env_kwargs={terminate_when_unhealthy: false}'
    collect $CARTPOLE data/cartpole-dense env_id=CartPole-v1 episode_length=500 sparse_reward=false
    collect $CARTPOLE data/cartpole-sparse env_id=CartPole-v1 episode_length=500 sparse_reward=true
    for f in default 0.2 1.0 3.0; do
      collect $PUSHT "data/pusht-$f" env_id=gym_pusht/PushT-v0 'env_kwargs={obs_type: state}' episode_length=300 \
        "friction=$([[ $f == default ]] && echo null || echo $f)"; done ;;
  sticky|termination|cartpole|pusht) sweep "configs/ablations/$1.yaml" gpu ;;
  all) for e in policies data sticky termination cartpole pusht; do "$SELF" "$e"; done ;;
  *) usage ;;
esac
