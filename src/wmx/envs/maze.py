"""Tabular maze family for the presentation experiments (docs/RUNNING.md §9).

Text maps: '#' wall, '.' floor, 'S' start, 'G' goal (terminal, reward 1 on entry), 'H' hazard (terminal, reward 0),
'a' / 'b' floor cells labelled route A / route B (labels only, same dynamics as '.'). Off-grid moves act like walls.
Actions 0 left, 1 down, 2 right, 3 up (FrozenLake order). Slip p: the intended move w.p. 1 - p, each perpendicular move
w.p. p/2 (p = 2/3 is FrozenLake's is_slippery). Moving into a wall leaves the agent in place.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Maze:
    grid: tuple[str, ...]
    cells: tuple[tuple[int, int], ...]   # state index -> (row, col) of every non-wall cell
    T: np.ndarray                        # (S, A, S) true next-state probabilities; terminals self-loop
    r_next: np.ndarray                   # (S,) reward for entering the cell (1 at G)
    step_cost: float                     # subtracted from every step's reward
    term: np.ndarray                     # (S,) bool: G and H
    mu0: np.ndarray                      # (S,) start distribution (one-hot at S)
    route: np.ndarray                    # (S,) '' | 'a' | 'b'


def read_map(spec: str) -> list[str]:
    """A map file, or 'random:ROWSxCOLS:SEED' for random_maze."""
    if spec.startswith("random:"):
        _, size, seed = spec.split(":")
        rows, cols = map(int, size.split("x"))
        return random_maze(rows, cols, int(seed))
    return [l for l in Path(spec).read_text().splitlines() if l.strip()]


def make_maze(grid: list[str], slip: float, step_cost: float) -> Maze:
    W = len(grid[0])
    if any(len(r) != W for r in grid):
        raise ValueError("map rows must have equal length")
    cells = [(i, j) for i, row in enumerate(grid) for j, c in enumerate(row) if c != "#"]
    idx, ch = {c: k for k, c in enumerate(cells)}, [grid[i][j] for i, j in cells]
    if ch.count("S") != 1 or "G" not in ch:
        raise ValueError("map needs exactly one S and at least one G")
    T = np.zeros((len(cells), 4, len(cells)))
    for k, (i, j) in enumerate(cells):
        if ch[k] in "GH":
            T[k, :, k] = 1.0
            continue
        for a in range(4):
            for b, p in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a - 1) % 4, slip / 2)):
                T[k, a, idx.get((i + MOVES[b][0], j + MOVES[b][1]), k)] += p
    return Maze(tuple(grid), tuple(cells), T, np.array([float(c == "G") for c in ch]), float(step_cost),
                np.array([c in "GH" for c in ch]), np.array([float(c == "S") for c in ch]),
                np.array([c if c in "ab" else "" for c in ch]))


def random_maze(rows: int, cols: int, seed: int, p_wall: float = 0.2, p_hazard: float = 0.1) -> list[str]:
    """Border of walls, S top-left, G bottom-right, other cells wall / hazard / floor at random; resampled until G is
    reachable from S without crossing a hazard."""
    rng = np.random.default_rng(seed)
    while True:
        g = [["#"] * (cols + 2)] + [["#"] + [str(x) for x in rng.choice([".", "#", "H"], cols,
                                                                          p=[1 - p_wall - p_hazard, p_wall, p_hazard])]
                                    + ["#"] for _ in range(rows)] + [["#"] * (cols + 2)]
        g[1][1], g[rows][cols] = "S", "G"
        seen, todo = {(1, 1)}, deque([(1, 1)])
        while todo:
            i, j = todo.popleft()
            for di, dj in MOVES:
                n = (i + di, j + dj)
                if n not in seen and g[n[0]][n[1]] in ".G":
                    seen.add(n)
                    todo.append(n)
        if (rows, cols) in seen:
            return ["".join(r) for r in g]


def value_iteration(m: Maze, gamma: float, allowed: np.ndarray | None = None, tol: float = 1e-12) -> np.ndarray:
    """Greedy optimal policy (S, A), one-hot, on the true maze. Cells outside `allowed` (S,) bool count as terminal with
    value 0, which restricts the plan to a route. Ties -> lowest action index. gamma < 1."""
    stop = m.term | (False if allowed is None else ~allowed)
    R = m.T @ m.r_next - m.step_cost
    V = np.zeros(len(m.mu0))
    while True:
        Q = R + gamma * m.T @ V
        V_new = np.where(stop, 0.0, Q.max(1))
        if np.abs(V_new - V).max() < tol:
            return np.eye(m.T.shape[1])[Q.argmax(1)]
        V = V_new


def collect(m: Maze, pi0: np.ndarray, N: int, episode_len: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """N real transitions (s, a, s') of pi0 (S, A) from mu0; reset on a terminal or after episode_len steps."""
    cp, cT, c0 = np.cumsum(pi0, -1), np.cumsum(m.T, -1), np.cumsum(m.mu0)
    draw = lambda cum: min(int(np.searchsorted(cum, rng.random(), side="right")), len(cum) - 1)
    out = np.empty((3, N), dtype=np.int64)
    s, t = draw(c0), 0
    for i in range(N):
        a = draw(cp[s])
        s2 = draw(cT[s, a])
        out[:, i] = s, a, s2
        t += 1
        s, t = (draw(c0), 0) if m.term[s2] or t >= episode_len else (s2, t)
    return tuple(out)


def greedy_route(m: Maze, pi: np.ndarray) -> str:
    """Follow pi's greedy actions without slip from the start: 'B' if it reaches G through a b cell, 'A' if through a
    cells only, 'other' if it never reaches G (loops, or ends in a hazard)."""
    idx = {c: k for k, c in enumerate(m.cells)}
    s, seen, labels = int(np.argmax(m.mu0)), set(), set()
    while s not in seen and not m.term[s]:
        seen.add(s)
        labels.add(m.route[s])
        (i, j), (di, dj) = m.cells[s], MOVES[int(np.argmax(pi[s]))]
        s = idx.get((i + di, j + dj), s)
    if m.r_next[s] <= 0:
        return "other"
    return "B" if "b" in labels else "A" if "a" in labels else "other"
