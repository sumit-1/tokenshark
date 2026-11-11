"""Train a tiny image model from scratch — the same recipe as Qwen-Image, small enough to watch.

Big image models are trained with *flow matching*. The whole idea fits in four lines:

    picture x0, random noise ε, a random noise level σ between 0 and 1
    x_t   = (1 − σ)·x0 + σ·ε                 mix them: σ = 1 is pure noise, σ = 0 is the picture
    target v = ε − x0                         the direction from picture to noise
    loss  = (model(x_t, σ, prompt) − v)²      teach the model to point along it

To make a picture, start from pure noise and walk the other way in a few steps (Euler):

    x ← x + (σ_next − σ)·model(x, σ, prompt)

Here the pictures are 16×16 coloured shapes and the "prompt" is two words (a colour and a shape), each with
its own embedding. One combination is never shown during training: if the model draws it anyway, it has
learned colour and shape as separate ideas it can combine — the small version of what lets a big model draw
things nobody photographed.

Guidance (CFG) is trained the usual way: for some examples the prompt is dropped, so the same model can also
answer "what would any picture look like here?". At sampling time:
    v = v_no_prompt + guidance · (v_prompt − v_no_prompt)
"""
from __future__ import annotations

import base64
import io
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .train import Job

SIZE = 16
COLORS = {"red": (0.9, 0.15, 0.15), "green": (0.15, 0.8, 0.25), "blue": (0.2, 0.35, 0.95), "yellow": (0.95, 0.85, 0.15)}
SHAPES = ["circle", "square", "triangle"]
HELD_OUT = ("yellow", "triangle")            # never shown in training
C_NAMES = list(COLORS)


def draw(color: str, shape: str, rnd: np.random.RandomState) -> np.ndarray:
    """One 16×16 picture in [-1, 1], with a little random size and position."""
    r = rnd.uniform(3.6, 5.6)
    cx, cy = rnd.uniform(6.2, 8.8, 2)
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) + 0.5
    if shape == "circle":
        inside = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    elif shape == "square":
        inside = (np.abs(x - cx) <= r * 0.85) & (np.abs(y - cy) <= r * 0.85)
    else:                                                                # triangle, point up
        h = (y - (cy - r)) / (2 * r)
        inside = (h >= 0) & (h <= 1) & (np.abs(x - cx) <= h * r * 1.1)
    img = np.full((3, SIZE, SIZE), 0.06, dtype=np.float32)
    for c in range(3):
        img[c][inside] = COLORS[color][c]
    return img * 2 - 1


class TinyDenoiser(nn.Module):
    """A small convolutional network: picture in, velocity out. The noise level and the prompt are added to
    every block (the same job Qwen-Image's "modulation" and text tokens do)."""

    def __init__(self, width: int = 48, blocks: int = 4, emb: int = 64):
        super().__init__()
        self.emb = emb
        self.color = nn.Embedding(len(C_NAMES) + 1, emb)        # last index = "no prompt"
        self.shape = nn.Embedding(len(SHAPES) + 1, emb)
        self.time = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(3, width, 3, padding=1)
        self.c1 = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks))
        self.c2 = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks))
        self.cond = nn.ModuleList(nn.Linear(emb, width) for _ in range(blocks))
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, sigma, color, shape):
        half = self.emb // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half)
        ang = sigma[:, None] * 1000.0 * freqs[None]
        e = self.time(torch.cat([ang.sin(), ang.cos()], -1)) + self.color(color) + self.shape(shape)
        h = self.inp(x)
        for c1, c2, cond in zip(self.c1, self.c2, self.cond):
            h = h + c2(F.silu(c1(F.silu(h)) + cond(e)[:, :, None, None]))
        return self.out(F.silu(h))


def _png(grid: np.ndarray, scale: int = 6) -> str:
    """[rows, cols, 3, H, W] in [-1, 1] → one PNG (data URL), each picture enlarged with a thin gap."""
    R, Cn = grid.shape[:2]
    gap = 2
    H = R * SIZE * scale + (R + 1) * gap
    W = Cn * SIZE * scale + (Cn + 1) * gap
    canvas = np.full((H, W, 3), 22, dtype=np.uint8)
    for r in range(R):
        for c in range(Cn):
            im = ((grid[r, c].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            im = np.kron(im, np.ones((scale, scale, 1), dtype=np.uint8))
            y0, x0 = gap + r * (SIZE * scale + gap), gap + c * (SIZE * scale + gap)
            canvas[y0:y0 + SIZE * scale, x0:x0 + SIZE * scale] = im
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class ImageJob(Job):
    kind = "image"

    def __init__(self, steps: int = 1200, batch: int = 64, lr: float = 3e-3, width: int = 48, blocks: int = 4,
                 seed: int = 0, drop: float = 0.15):
        super().__init__()
        self.steps, self.batch, self.lr, self.seed, self.drop = int(steps), int(batch), float(lr), int(seed), float(drop)
        self.width, self.blocks = int(width), int(blocks)
        self.model: TinyDenoiser | None = None
        self.grids: list[dict] = []
        self.n_params = 0

    def extra(self):
        return {"n_params": self.n_params, "grids": self.grids[-5:], "colors": C_NAMES, "shapes": SHAPES,
                "held_out": list(HELD_OUT), "ready": self.model is not None and self.status in ("done", "stopped")}

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, color: list[int], shape: list[int], steps: int = 20, guidance: float = 2.0, seed: int = 0,
               keep: bool = False):
        """Euler sampling from noise. Returns the pictures [n, 3, 16, 16]; keep=True also returns, per step, the
        noisy input x_t and the model's guess of the finished picture x̂0 = x_t − σ·v."""
        m = self.model
        n = len(color)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 3, SIZE, SIZE, generator=g)
        c, s = torch.tensor(color), torch.tensor(shape)
        nc, ns = torch.full_like(c, len(C_NAMES)), torch.full_like(s, len(SHAPES))
        sig = torch.linspace(1.0, 0.0, steps + 1)
        trace = []
        for i in range(steps):
            t = sig[i].expand(n)
            v = m(x, t, c, s)
            if guidance != 1.0:
                v0 = m(x, t, nc, ns)
                v = v0 + guidance * (v - v0)
            if keep:
                trace.append((x.clone(), (x - sig[i] * v).clamp(-1, 1)))
            x = x + (sig[i + 1] - sig[i]) * v
        return (x.clamp(-1, 1), trace) if keep else x.clamp(-1, 1)

    def grid(self, steps: int = 20, guidance: float = 2.0, seed: int = 0) -> str:
        """Every colour × shape, one picture each (rows = shapes, columns = colours)."""
        ci = [c for _ in SHAPES for c in range(len(C_NAMES))]
        si = [s for s in range(len(SHAPES)) for _ in C_NAMES]
        x = self.sample(ci, si, steps, guidance, seed).numpy()
        return _png(x.reshape(len(SHAPES), len(C_NAMES), 3, SIZE, SIZE))

    def strip(self, color: str, shape: str, steps: int = 12, guidance: float = 2.0, seed: int = 0) -> dict:
        """One picture being made: top row what the model is given at each step, bottom row its guess."""
        if color not in COLORS or shape not in SHAPES:
            raise ValueError(f"colour: {', '.join(C_NAMES)} · shape: {', '.join(SHAPES)}")
        steps = max(1, min(int(steps), 60))
        x, trace = self.sample([C_NAMES.index(color)], [SHAPES.index(shape)], steps, guidance, seed, keep=True)
        show = sorted(set(np.linspace(0, steps - 1, min(steps, 12)).round().astype(int).tolist()))
        rows = np.stack([np.stack([trace[i][0][0].clamp(-1, 1).numpy() for i in show]),
                         np.stack([trace[i][1][0].numpy() for i in show])])
        return {"strip": _png(rows, 5), "final": _png(x.numpy()[None], 10), "steps_shown": show, "steps": steps,
                "sigmas": [round(1 - i / steps, 3) for i in show], "color": color, "shape": shape,
                "guidance": guidance, "seed": seed, "held_out": (color, shape) == HELD_OUT}

    # ---------------------------------------------------------------- training
    def run(self):
        torch.manual_seed(self.seed)
        rnd = np.random.RandomState(self.seed)
        pairs = [(c, s) for c in C_NAMES for s in SHAPES if (c, s) != HELD_OUT]
        self.model = m = TinyDenoiser(self.width, self.blocks)
        self.n_params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=self.lr, weight_decay=0.0)
        every = max(1, self.steps // 60)
        t0 = time.time()
        self.grids.append({"step": 0, "url": self.grid()})
        for step in range(1, self.steps + 1):
            if self.stop:
                break
            idx = rnd.randint(0, len(pairs), self.batch)
            x0 = torch.from_numpy(np.stack([draw(*pairs[i], rnd) for i in idx]))
            c = torch.tensor([C_NAMES.index(pairs[i][0]) for i in idx])
            s = torch.tensor([SHAPES.index(pairs[i][1]) for i in idx])
            dropped = torch.rand(self.batch) < self.drop               # train "no prompt" too (for guidance)
            c = torch.where(dropped, torch.full_like(c, len(C_NAMES)), c)
            s = torch.where(dropped, torch.full_like(s, len(SHAPES)), s)
            sigma = torch.rand(self.batch)
            noise = torch.randn_like(x0)
            xt = (1 - sigma)[:, None, None, None] * x0 + sigma[:, None, None, None] * noise
            loss = F.mse_loss(m(xt, sigma, c, s), noise - x0)
            for grp in opt.param_groups:
                grp["lr"] = self.lr * min(1.0, step / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / self.steps)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            self.step, self.seconds = step, time.time() - t0
            if step % every == 0 or step == 1 or step == self.steps:
                self.losses.append([step, round(loss.item(), 4)])
                self.message = f"step {step}/{self.steps} · loss {loss.item():.3f}"
            if step % max(1, self.steps // 5) == 0 or step == self.steps:
                self.grids.append({"step": step, "url": self.grid()})
        self.message = (f"trained for {self.step} steps — try the sampler below "
                        f"(the {HELD_OUT[0]} {HELD_OUT[1]} was never in the training pictures)")
