from typing import Optional, Tuple

import torch
from torch import nn
from diffusers import UNet2DConditionModel

class ARCUNet(nn.Module):
    """UNet tailored for ARC tasks, repurposed from HuggingFace
    implementation.

    'size' parameter controls the setting of the sizes. Currently, there
    are three supported sizes:
        - Small: 3 feat. res., 1 block/res., channels 80,160,160, 4 heads
        - Medium: 3 feat. res., 1 block/res., channels 120,240,240, 6 heads
        - Big: 3 feat. res., 2 block/res., channels 160,320,320, 8 heads
            plus a middle block.

    Each ARC task gets a dedicated learnable token that is prepended to the
    sequence of flattened pixel embeddings. Pixels are represented by a
    discrete color vocabulary of size ``num_colors``.
    """

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 64,
        num_colors: int = 10,
        size: str = 'medium',
        num_task_tokens: int = 1,
    ) -> None:
        super().__init__()

        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
            raise ValueError("`num_tasks` must be > 0.")

        self.num_colors = num_colors
        self.image_size = image_size
        self.num_task_tokens = num_task_tokens

        if size == 'small':
            embed_dim = 160
        elif size == 'medium':
            embed_dim = 240
        elif size == 'big':
            embed_dim = 320

        self.color_embed = nn.Embedding(num_colors, 4)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * num_task_tokens)

        self.encoder = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=4,
            out_channels=num_colors,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            mid_block_type = 'UNetMidBlock2DCrossAttn' if size == 'big' else None,
            block_out_channels=(embed_dim//2, embed_dim, embed_dim),
            cross_attention_dim=embed_dim,
            layers_per_block=2 if size == 'big' else 1,
            attention_head_dim=embed_dim//40, # actually "num_attention_heads", naming issue
            norm_num_groups=embed_dim//10,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_class_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        tokens = self.color_embed(pixel_values.long()).permute(0, 3, 1, 2)
        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)

        logits = self.encoder(tokens, 0, task_tokens).sample
        return logits
