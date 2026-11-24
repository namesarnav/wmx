"""Convert an official DreamerV3 replay directory into the wmx flat dataset (wmx.data.replay).
Usage: python scripts/convert_dreamer_replay.py ~/logdir/crafter/replay data/crafter
"""
import sys
from pathlib import Path

import numpy as np

from wmx.data.replay import KEYS, write_dataset

src, out = Path(sys.argv[1]), Path(sys.argv[2])
KEYMAP = {"obs": "image"}  # wmx key -> DreamerV3 key
# VERIFY: embodied saves chunks as {time}-{uuid}-{succ}-{length}.npz, where succ is the uuid of the next chunk of
# the same worker stream. Keys: image (T,64,64,3) uint8, action (T,) int or (T,17) one-hot, reward, is_first,
# is_terminal. Check one file with np.load(...).files before converting.
chunks = {}
for f in src.glob("*.npz"):
    _, uuid, succ, _ = f.stem.split("-")
    chunks[uuid] = (f, succ)
heads = sorted(set(chunks) - {succ for _, succ in chunks.values()}, key=lambda u: chunks[u][0].name)
if not heads:
    raise SystemExit(f"no chunk streams found in {src}")

parts = {k: [] for k in KEYS}
for u in heads:
    stream = {k: [] for k in KEYS}
    while u in chunks:
        f, u = chunks[u]
        with np.load(f) as c:
            for k in KEYS:
                v = c[KEYMAP.get(k, k)]
                stream[k].append(v.argmax(-1) if k == "action" and v.ndim == 2 else v)
    stream = {k: np.concatenate(v) for k, v in stream.items()}
    start = int(np.argmax(stream["is_first"]))  # drop a partial episode at the stream start (evicted chunks)
    for k in KEYS:
        parts[k].append(stream[k][start:])

data = {k: np.concatenate(v) for k, v in parts.items()}
data["obs"], data["action"] = data["obs"].astype(np.uint8), data["action"].astype(np.int64)
data["reward"], data["is_first"] = data["reward"].astype(np.float32), data["is_first"].astype(bool)
data["is_terminal"] = data["is_terminal"].astype(bool)
write_dataset(out, **data)
print(f"{len(data['action'])} steps, {data['is_first'].sum()} episodes -> {out}")
