"""Unit tests for Sokoban (engine, levels, text format, GRPO) and the best-of-K / GRPO pipelines with a fake text world
model that is the true dynamics (CONTEXT.md §7 test 2: imagined = true return, delta = 0). No LM download needed."""
import numpy as np
import pytest
import torch

from wmx.config import load_config, make_config
from wmx.envs.sokoban import distances, encode, generate, greedy_action, step_board, valid
from wmx.models.text_wm import outcome, parse_outcome
from wmx.policies.actor_critic import DiscreteActor
from wmx.policies.grpo import grpo_advantages, grpo_update
from wmx.train import sokoban

BOARD = "#####\n#@$.#\n#####\n#####\n#####"


def test_engine_push_reward_and_block():
    b, r, done = step_board(BOARD, 3)                     # right: push the box onto the goal, solved
    assert b.split("\n")[1] == "# @*#" and done and r == pytest.approx(-0.1 + 1 + 10)
    b, r, done = step_board(BOARD, 2)                     # left: wall, stay
    assert b == BOARD and r == pytest.approx(-0.1) and not done


def test_generated_levels_are_valid_and_solvable():
    rng = np.random.default_rng(0)
    for _ in range(5):
        b = generate(rng, 6, 2, 30)
        dist = distances(b)
        assert valid(b, 6) and b in dist                  # reachable solved board exists
        for _ in range(dist[b]):                          # greedy play solves it in exactly dist[b] steps
            b, _, done = step_board(b, greedy_action(b, dist, rng))
        assert done


def test_outcome_text_roundtrip():
    b = generate(np.random.default_rng(1), 6, 2, 30)
    assert parse_outcome(outcome(b, -0.1, False), 6) == (b, pytest.approx(-0.1), False)
    assert parse_outcome("garbage", 6)[0] is None
    assert encode([b], 6, torch.device("cpu")).shape == (1, 6 * 6 * 7)


def test_grpo_advantages_and_update():
    adv = grpo_advantages(torch.tensor([[1.0, 3.0], [2.0, 2.0]]))
    assert torch.allclose(adv, torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
    pi0 = DiscreteActor(4, 8, 2, 0.01).requires_grad_(False)
    pi = DiscreteActor(4, 8, 2, 0.01)
    pi.load_state_dict(pi0.state_dict())
    x, a = torch.randn(3, 2, 4), torch.tensor([[1, 0]] * 3)
    logs = grpo_update(pi, pi0, torch.optim.SGD(pi.parameters(), 0.5), x, a, torch.ones(3, 2),
                       torch.tensor([1.0, -1.0]), 0.2, 0.0, 3)
    assert logs["kl_step"] > 0 and (pi(x).probs[..., 1] > pi0(x).probs[..., 1])[:, 0].all()


class TrueWM(torch.nn.Module):
    """Text-WM stand-in that is the real dynamics: p(true outcome) = 1."""

    def __init__(self, size):
        super().__init__()
        self.size, self.lm = size, torch.nn.Linear(1, 1)

    def sample(self, boards, actions):
        return [step_board(b, a) for b, a in zip(boards, actions)]

    def logprob(self, boards, actions, outcomes):
        assert all(outcome(*step_board(b, a)) == o for b, a, o in zip(boards, actions, outcomes))
        return torch.zeros(len(boards))


@pytest.mark.parametrize("optimizer", ["bok", "grpo"])
def test_pipeline_with_true_model(tmp_path, monkeypatch, optimizer):
    cfg = make_config({**load_config("configs/sokoban/base.yaml").__dict__, "device": "cpu", "N": 300,
                       "optimizer": optimizer, "K_values": [1, 4], "starts": 4, "H": 4, "group": 2, "batch_starts": 2,
                       "updates": 2, "every": 1, "bc_steps": 20, "delta_pairs": 16, "episode_length": 10})
    monkeypatch.setattr(sokoban, "load_or_finetune", lambda cfg, *_: TrueWM(cfg.size))
    df = sokoban.run(cfg, tmp_path)
    assert (df.real_env_steps > 0).all() and (df["delta_q0.95"] == 0).all() and (tmp_path / "done").exists()
    assert df.d_real[0] == pytest.approx(0, abs=1e-6)
    if optimizer == "bok":                                # deterministic true model: imagined = true exactly
        assert np.allclose(df.imagined_return, df.true_return)
