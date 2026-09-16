"""The scene-state network: two camera views in, object poses out.

Architecture, and why:

* **Shared trunk, two views.** The overhead and front cameras see the same cell from
  unrelated viewpoints, so the two images are pushed through the *same* convolutional
  trunk independently and their descriptors are concatenated. Sharing the weights halves
  the parameter count and lets both views train the same low-level filters; keeping the
  forward passes separate stops a filter from straddling two viewpoints at the same pixel
  coordinate, which is what a naive 6-channel stack does. The 6-channel input tensor is
  only the transport format -- ``forward`` splits it.

* **Per-object spatial attention, not global pooling.** This is the part that decides
  whether the errors are centimetres or millimetres. A globally pooled descriptor has to
  encode five positions in one vector and, measured on held-out seeds, that costs ~54 mm
  of median error -- the plate, the largest object on the table, landed 77 mm out. Instead
  the trunk keeps a 32x32 feature map and a 1x1 convolution predicts one attention map per
  prop over it. Soft-argmax of that map gives an explicit image-space coordinate per
  object, and the same distribution pools an object-specific descriptor out of the feature
  map. The overhead camera is mounted looking straight down, so its soft-argmax coordinate
  is very nearly an affine function of table-frame (x, y) -- exactly the inductive bias the
  regression head needs, and it is learned end to end from the position loss alone, with no
  camera calibration baked in.

* **BatchNorm, not GroupNorm.** BN folds into the preceding convolution at export time, so
  the FP16 and INT8 IRs have no normalization layers left to quantize badly.

Outputs, per :mod:`tandem.perception.schema`: 5 normalized xyz triples, 5 (cos, sin) yaw
pairs, 5 visibility logits and 1 drawer-open logit -- 31 numbers, in that order.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .schema import IMAGE_SIZE, N_PROPS, OUTPUT_DIM


def _block(cin: int, cout: int, kernel: int = 3, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ObjectAttention(nn.Module):
    """One attention map per prop over a feature grid -> coordinates plus descriptors."""

    def __init__(self, channels: int, n_objects: int, height: int, width: int,
                 temperature: float = 1.0):
        super().__init__()
        # A 3x3 stage before the per-object logits: deciding "this cell is the fork, not the
        # spoon" needs a little spatial context, and a bare 1x1 has none.
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_objects, 1),
        )
        self.temperature = temperature
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs.reshape(1, 1, -1))
        self.register_buffer("grid_y", ys.reshape(1, 1, -1))

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, C, H, W)`` -> ``(B, N, 2)`` coordinates and ``(B, N, C)`` descriptors."""
        weights = torch.softmax(self.score(feat).flatten(2) / self.temperature, dim=-1)
        x = (weights * self.grid_x).sum(-1)
        y = (weights * self.grid_y).sum(-1)
        descriptors = torch.bmm(weights, feat.flatten(2).transpose(1, 2))
        return torch.stack([x, y], dim=-1), descriptors


class PerceptionNet(nn.Module):
    """Scene-state estimator over the overhead + front pair."""

    def __init__(self, dropout: float = 0.1, embed: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            _block(3, 48, kernel=5),  # 64
            _block(48, 96),  # 32 -- the attention resolution
        )
        self.deep = nn.Sequential(
            _block(96, 192),  # 16
            _block(192, 256),  # 8
            _block(256, 256),  # 4
        )
        grid = IMAGE_SIZE // 4
        self.attention = ObjectAttention(96, N_PROPS, grid, grid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.object_embed = nn.Parameter(torch.zeros(1, N_PROPS, embed))
        nn.init.trunc_normal_(self.object_embed, std=0.02)

        per_object = 2 * (2 + 96) + 2 * 256 + embed
        self.object_head = nn.Sequential(
            nn.Linear(per_object, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),  # x, y, z, cos yaw, sin yaw, visibility logit
        )
        self.drawer_head = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    def encode(self, view: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mid = self.trunk(view)
        coords, descriptors = self.attention(mid)
        glob = self.pool(self.deep(mid)).flatten(1)
        return coords, descriptors, glob

    def forward_with_keypoints(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Training path: also returns the attention coordinates, ``(B, 2, N, 2)``.

        The trainer supervises these against the analytic projection of the true object
        positions (see :mod:`tandem.perception.geometry`). Telling the attention where to
        look, rather than leaving it to be discovered through the position loss alone, is
        what takes the held-out error from centimetres to millimetres. Nothing in the
        exported IR depends on this head -- ``forward`` is the inference path.
        """
        raw, over_xy, front_xy = self._forward(images)
        return raw, torch.stack([over_xy, front_xy], dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """``images``: ``(B, 6, 128, 128)`` float32 in [0, 1], overhead then front."""
        return self._forward(images)[0]

    def _forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        over_xy, over_feat, over_glob = self.encode(images[:, :3])
        front_xy, front_feat, front_glob = self.encode(images[:, 3:])

        batch = images.shape[0]
        context = torch.cat([over_glob, front_glob], dim=1)
        per_object = torch.cat(
            [
                over_xy,
                front_xy,
                over_feat,
                front_feat,
                context.unsqueeze(1).expand(-1, N_PROPS, -1),
                self.object_embed.expand(batch, -1, -1),
            ],
            dim=-1,
        )
        objects = self.object_head(per_object)  # (B, N, 6)
        raw = torch.cat(
            [
                objects[:, :, 0:3].reshape(batch, -1),
                objects[:, :, 3:5].reshape(batch, -1),
                objects[:, :, 5],
                self.drawer_head(context),
            ],
            dim=1,
        )
        return raw, over_xy, front_xy


def split_outputs(raw: Tensor) -> dict[str, Tensor]:
    """Slice the flat output into its named parts (still normalized / still logits)."""
    from .schema import DRAWER_INDEX, POS_SLICE, VIS_SLICE, YAW_SLICE

    return {
        "pos": raw[:, POS_SLICE].reshape(raw.shape[0], -1, 3),
        "yaw": raw[:, YAW_SLICE].reshape(raw.shape[0], -1, 2),
        "vis_logit": raw[:, VIS_SLICE],
        "drawer_logit": raw[:, DRAWER_INDEX],
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def example_inputs(batch: int = 1, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return {"images": torch.zeros(batch, 6, IMAGE_SIZE, IMAGE_SIZE, device=device)}


assert OUTPUT_DIM == N_PROPS * 6 + 1, "output layout disagrees with the schema"
