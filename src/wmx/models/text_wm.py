"""Text world model for Sokoban (Task 4): a small Qwen causal LM fine-tuned on (board, action) -> (next board,
reward, done) text. Its state is the board string, so it has its own string interface instead of the tensor one:
  sample(boards, actions) -> [(next board | None, reward, done)]   (stochastic step; None = unparseable output)
  logprob(boards, actions, outcomes) -> log p(outcome text | prompt)  (predict_distribution, used for delta)
"""
import torch

from wmx.envs.sokoban import ACTIONS, valid


def prompt(board: str, a: int) -> str:
    return f"Board:\n{board}\nAction: {ACTIONS[a]}\nNext:\n"


def outcome(board: str, r: float, done: bool) -> str:
    return f"{board}\nReward: {r:.1f}\nDone: {int(done)}\n"


def parse_outcome(text: str, size: int) -> tuple[str | None, float, bool]:
    """Inverse of `outcome`. Returns (None, 0, True) if the text is not a valid outcome."""
    lines = text.split("\n")
    try:
        board, r, done = "\n".join(lines[:size]), float(lines[size].removeprefix("Reward: ")), lines[size + 1]
    except (IndexError, ValueError):
        return None, 0.0, True
    if not valid(board, size) or done not in ("Done: 0", "Done: 1"):
        return None, 0.0, True
    return board, r, done == "Done: 1"


class TextWorldModel(torch.nn.Module):
    def __init__(self, name: str, size: int, temperature: float, gen_batch: int, device: torch.device,
                 dtype: torch.dtype):
        """name: Hugging Face model id or a local save_pretrained dir; dtype float32 to fine-tune, bfloat16 frozen.
        VERIFY: Qwen tokenizers have no pad token; eos is used for padding (and masked out)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy import, only for Sokoban runs
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
        self.size, self.temperature, self.gen_batch, self.device = size, temperature, gen_batch, device
        self.max_new = len(outcome("x" * size * size + "\n" * (size - 1), -10.1, True)) + 8  # chars >= tokens

    def pack(self, prompts: list[str], targets: list[str]) -> tuple[torch.Tensor, ...]:
        """Tokenize prompt and target separately (no merges across the boundary) and right-pad.
        Returns input_ids, attention_mask, labels (-100 on prompt and padding), each (B, L)."""
        p = self.tok(prompts, add_special_tokens=False)["input_ids"]
        t = self.tok(targets, add_special_tokens=False)["input_ids"]
        seqs = [a + b + [self.tok.eos_token_id] for a, b in zip(p, t)]
        L, pad = max(map(len, seqs)), self.tok.eos_token_id
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
        mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
        labels = torch.tensor([[-100] * len(a) + s[len(a):] + [-100] * (L - len(s)) for a, s in zip(p, seqs)],
                              device=self.device)
        return ids, mask, labels

    def token_logprobs(self, prompts: list[str], targets: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log p of the target (+ eos) given the prompt: (B, L-1) log-probs and (B, L-1) target mask."""
        ids, mask, labels = self.pack(prompts, targets)
        logits = self.lm(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        y = labels[:, 1:]
        lp = logits.log_softmax(-1).gather(-1, y.clamp(min=0)[..., None])[..., 0]
        return lp, (y != -100).float()

    def loss(self, boards: list[str], actions: list[int], outcomes: list[str]) -> torch.Tensor:
        """Mean per-token NLL of the outcome text (fine-tuning objective)."""
        lp, m = self.token_logprobs([prompt(b, a) for b, a in zip(boards, actions)], outcomes)
        return -(lp * m).sum() / m.sum()

    @torch.no_grad()
    def logprob(self, boards: list[str], actions: list[int], outcomes: list[str]) -> torch.Tensor:
        """log p(outcome | board, action) of the whole outcome text incl. eos: (B,)."""
        lp, m = self.token_logprobs([prompt(b, a) for b, a in zip(boards, actions)], outcomes)
        return (lp * m).sum(-1)

    def sample(self, boards: list[str], actions: list[int]) -> list[tuple[str | None, float, bool]]:
        """Sampled next (board, reward, done) at self.temperature, in chunks of gen_batch."""
        return [o for i in range(0, len(boards), self.gen_batch)
                for o in self._sample(boards[i:i + self.gen_batch], actions[i:i + self.gen_batch])]

    @torch.no_grad()
    def _sample(self, boards: list[str], actions: list[int]) -> list[tuple[str | None, float, bool]]:
        p = self.tok([prompt(b, a) for b, a in zip(boards, actions)], add_special_tokens=False)["input_ids"]
        L, pad = max(map(len, p)), self.tok.eos_token_id
        ids = torch.tensor([[pad] * (L - len(s)) + s for s in p], device=self.device)
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in p], device=self.device)
        out = self.lm.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=self.temperature,
                               top_k=0, top_p=1.0, max_new_tokens=self.max_new, pad_token_id=pad,
                               eos_token_id=self.tok.eos_token_id)
        texts = self.tok.batch_decode(out[:, L:], skip_special_tokens=True)
        return [parse_outcome(t, self.size) for t in texts]
