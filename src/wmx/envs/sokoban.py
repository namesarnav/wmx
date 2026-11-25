"""Sokoban: a small pure-Python engine whose state is the text board, so the real env, the text world model
and the policy share one representation. Own implementation (docs/DECISIONS.md): gym-sokoban targets old gym, and we
need to load arbitrary boards (true next state of an imagined board, for delta).

Board: rows joined by '\\n'; '#' wall, ' ' floor, '.' goal, '$' box, '*' box on goal, '@' player, '+' player on goal.
Rewards (gym-sokoban/Boxoban): -0.1 per step, +1 box onto a goal, -1 box off a goal, +10 when all boxes are on goals.
"""
from collections import deque

import numpy as np
import torch

ACTIONS = ("up", "down", "left", "right")
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
CHARS = "# .$*@+"


def parse(board: str) -> tuple[set, set, set, tuple[int, int]]:
    """board -> (walls, goals, boxes, player); cells are (row, col)."""
    walls, goals, boxes, player = set(), set(), set(), None
    for i, row in enumerate(board.split("\n")):
        for j, ch in enumerate(row):
            if ch == "#":
                walls.add((i, j))
            if ch in ".*+":
                goals.add((i, j))
            if ch in "$*":
                boxes.add((i, j))
            if ch in "@+":
                player = (i, j)
    return walls, goals, boxes, player


def render(size: int, walls: set, goals: set, boxes: set, player: tuple[int, int]) -> str:
    def ch(c):
        if c in walls:
            return "#"
        if c == player:
            return "+" if c in goals else "@"
        if c in boxes:
            return "*" if c in goals else "$"
        return "." if c in goals else " "
    return "\n".join("".join(ch((i, j)) for j in range(size)) for i in range(size))


def valid(board: str, size: int) -> bool:
    """Well-formed: size x size of CHARS, exactly one player, as many boxes as goals (>= 1)."""
    rows = board.split("\n")
    if len(rows) != size or any(len(r) != size or set(r) - set(CHARS) for r in rows):
        return False
    _, goals, boxes, player = parse(board)
    return player is not None and sum(r.count("@") + r.count("+") for r in rows) == 1 and len(boxes) == len(goals) > 0


def step_board(board: str, a: int) -> tuple[str, float, bool]:
    """True dynamics: (board, action) -> (next board, reward, done). Done iff all boxes are on goals."""
    walls, goals, boxes, (pi, pj) = parse(board)
    di, dj = MOVES[a]
    nxt, beyond, r = (pi + di, pj + dj), (pi + 2 * di, pj + 2 * dj), -0.1
    if nxt in walls or (nxt in boxes and (beyond in walls or beyond in boxes)):
        nxt = (pi, pj)                                                       # blocked: stay
    elif nxt in boxes:
        boxes = boxes - {nxt} | {beyond}
        r += (beyond in goals) - (nxt in goals)
    done = boxes <= goals
    r += 10.0 * done
    return render(len(board.split("\n")), walls, goals, boxes, nxt), r, done


def generate(rng: np.random.Generator, size: int, n_boxes: int, reverse_steps: int) -> str:
    """Solvable level by reverse play: boxes start on goals in an empty walled room, then the player walks randomly,
    pulling an adjacent box behind it with prob. 1/2. Rejects levels that are already solved."""
    while True:
        walls = {(i, j) for i in range(size) for j in range(size) if i in (0, size - 1) or j in (0, size - 1)}
        free = [(i, j) for i in range(1, size - 1) for j in range(1, size - 1)]
        cells = [free[k] for k in rng.choice(len(free), n_boxes + 1, replace=False)]
        goals, player = set(cells[:-1]), cells[-1]
        boxes = set(goals)
        for _ in range(reverse_steps):
            di, dj = MOVES[rng.integers(4)]
            nxt, behind = (player[0] + di, player[1] + dj), (player[0] - di, player[1] - dj)
            if nxt in walls or nxt in boxes:
                continue
            if behind in boxes and rng.random() < 0.5:
                boxes = boxes - {behind} | {player}
            player = nxt
        if not boxes <= goals:
            return render(size, walls, goals, boxes, player)


def distances(board: str, max_states: int = 200_000) -> dict[str, int]:
    """Steps-to-solve of every board reachable from `board`: forward BFS over the reachable graph (solved boards are
    terminal), then backward BFS from the solved boards. Boards missing from the result cannot be solved."""
    preds, seen, solved, queue = {}, {board}, set(), deque([board])
    while queue:
        b = queue.popleft()
        for a in range(4):
            b2, _, done = step_board(b, a)
            preds.setdefault(b2, set()).add(b)
            if b2 not in seen:
                seen.add(b2)
                (solved.add if done else queue.append)(b2)
        if len(seen) > max_states:
            raise ValueError(f"more than {max_states} reachable boards; use a smaller level")
    dist, queue = {b: 0 for b in solved}, deque(solved)
    while queue:
        b = queue.popleft()
        for p in preds.get(b, ()):
            if p not in dist:
                dist[p] = dist[b] + 1
                queue.append(p)
    return dist


def greedy_action(board: str, dist: dict[str, int], rng: np.random.Generator) -> int:
    """Optimal action under `dist` (ties at random); random if the board cannot be solved any more."""
    d = np.array([dist.get(step_board(board, a)[0], np.inf) for a in range(4)])
    return int(rng.choice(np.flatnonzero(d == d.min())))


def encode(boards: list[str], size: int, device: torch.device) -> torch.Tensor:
    """One-hot of every cell over CHARS: (B, size * size * len(CHARS)) float. Boards must be valid."""
    idx = torch.tensor([[CHARS.index(c) for c in b.replace("\n", "")] for b in boards], device=device)
    return torch.nn.functional.one_hot(idx, len(CHARS)).flatten(1).float()


class SokobanEnv:
    """Real env: reset(board) -> board; step(a) -> (board, reward, terminated, done) with a time limit.
    `steps` counts real environment steps (CONTEXT.md §6)."""

    def __init__(self, length: int):
        self.length, self.steps, self.t, self.board = length, 0, 0, None

    def reset(self, board: str) -> str:
        self.board, self.t = board, 0
        return board

    def step(self, a: int) -> tuple[str, float, bool, bool]:
        self.board, r, term = step_board(self.board, int(a))
        self.steps, self.t = self.steps + 1, self.t + 1
        return self.board, r, term, term or self.t >= self.length
