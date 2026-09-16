"""Action-chunking transformer, distilled from the scripted oracle.

An ACT-shaped network: convolutional tokens from each camera, proprioception and goal
conditioning as extra tokens, a small transformer encoder over the lot, and a decoder that
turns K learned queries into K future control vectors in one forward pass.

Two deliberate departures from the paper:

* **No CVAE latent.** ACT's variational latent exists to absorb the multi-modality of
  human teleoperation -- two operators solving the same scene differently. The teacher
  here is a deterministic scripted executor, so the conditional action distribution is
  unimodal and the latent would only add a KL term to tune. Plain L1 regression on the
  chunk is the right fit for distillation.
* **Separate trunks per camera.** The overhead and wrist views play different roles (where
  is the object on the table, versus where is it relative to the jaws), and with only two
  cameras the parameter saving from sharing is not worth the coupling.

Everything is export-friendly: static shapes, no data-dependent control flow, attention
written against ``nn.MultiheadAttention`` so OpenVINO lowers it to plain matmuls.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .schema import ACTION_DIM, CHUNK, COND_DIM, IMAGE_SIZE, PROPRIO_DIM


def _block(cin: int, cout: int, kernel: int = 3, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CameraTrunk(nn.Module):
    """128x128 RGB -> a grid of ``dim``-wide tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            _block(3, 32, kernel=5),  # 64
            _block(32, 64),  # 32
            _block(64, 128),  # 16
            _block(128, 256),  # 8
            _block(256, 256),  # 4
        )
        self.project = nn.Conv2d(256, dim, 1)

    def forward(self, image: Tensor) -> Tensor:
        feat = self.project(self.layers(image))
        return feat.flatten(2).transpose(1, 2)  # (B, 16, dim)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        h = self.norm1(queries)
        queries = queries + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        h = self.norm2(queries)
        queries = queries + self.drop(self.cross_attn(h, memory, memory, need_weights=False)[0])
        return queries + self.drop(self.ff(self.norm3(queries)))


class ActionChunkPolicy(nn.Module):
    """Two views + proprioception + goal -> ``(B, CHUNK, 12)`` joint-position targets."""

    def __init__(
        self,
        dim: int = 256,
        heads: int = 4,
        ff: int = 1024,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        dropout: float = 0.1,
        chunk: int = CHUNK,
    ):
        super().__init__()
        self.chunk = chunk
        self.overhead_trunk = CameraTrunk(dim)
        self.wrist_trunk = CameraTrunk(dim)

        tokens_per_camera = (IMAGE_SIZE // 32) ** 2
        self.image_pos = nn.Parameter(torch.zeros(1, 2 * tokens_per_camera, dim))
        self.state_pos = nn.Parameter(torch.zeros(1, 2, dim))
        self.proprio_embed = nn.Linear(PROPRIO_DIM, dim)
        self.cond_embed = nn.Linear(COND_DIM, dim)
        self.queries = nn.Parameter(torch.zeros(1, chunk, dim))
        nn.init.trunc_normal_(self.image_pos, std=0.02)
        nn.init.trunc_normal_(self.state_pos, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.encoder = nn.ModuleList(
            EncoderLayer(dim, heads, ff, dropout) for _ in range(encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(dim, heads, ff, dropout) for _ in range(decoder_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, ACTION_DIM)

    def forward(
        self, overhead: Tensor, wrist: Tensor, proprio: Tensor, cond: Tensor
    ) -> Tensor:
        image_tokens = torch.cat(
            [self.overhead_trunk(overhead), self.wrist_trunk(wrist)], dim=1
        ) + self.image_pos
        state_tokens = torch.stack(
            [self.proprio_embed(proprio), self.cond_embed(cond)], dim=1
        ) + self.state_pos

        memory = torch.cat([image_tokens, state_tokens], dim=1)
        for layer in self.encoder:
            memory = layer(memory)

        queries = self.queries.expand(memory.shape[0], -1, -1)
        for layer in self.decoder:
            queries = layer(queries, memory)
        return self.head(self.norm(queries))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def example_inputs(batch: int = 1, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Shape-correct dummy inputs, used for tracing and for benchmark sweeps."""
    return {
        "overhead": torch.zeros(batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device),
        "wrist": torch.zeros(batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device),
        "proprio": torch.zeros(batch, PROPRIO_DIM, device=device),
        "cond": torch.zeros(batch, COND_DIM, device=device),
    }
