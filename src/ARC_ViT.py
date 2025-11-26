from typing import Optional, Tuple

from utils.pos_embed import VisionRotaryEmbeddingFast
import torch
from torch import nn

from timm.models.vision_transformer import PatchEmbed

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require the head dimension to be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        half_head_dim = embed_dim // num_heads // 2
        self.rotary = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=int(max_seq_len ** 0.5),
            no_rope=no_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                mask,
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class ARCTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            no_rope=no_rope,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)
        return x


class ARCTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_seq_len: int,
        no_rope: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ARCTransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    max_seq_len=max_seq_len,
                    no_rope=no_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class ARCViT(nn.Module):
    """Vision Transformer tailored for ARC tasks.

    Each ARC task gets a dedicated learnable token that is prepended to the
    sequence of flattened pixel embeddings. Pixels are represented by a
    discrete color vocabulary of size ``num_colors``.
    """

    def __init__(
        self,
        num_tasks: int,
        image_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_task_tokens: int = 1,
        patch_size: int = 2
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("`image_size` must be > 0.")
        if num_colors <= 0:
            raise ValueError("`num_colors` must be > 0.")
        if num_tasks <= 0:
            raise ValueError("`num_tasks` must be > 0.")

        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size//patch_size)**2
        self.patch_size = patch_size
        print(f"Patch size: {self.patch_size}, sequence length: {self.seq_length}")
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)

        total_seq_len = self.num_task_tokens + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.encoder = ARCTransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_seq_len=total_seq_len,
            no_rope=num_task_tokens,
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size)**2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if pixel_values.dim() != 3:
            raise ValueError("`pixel_values` must be (batch, height, width).")
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(
                "`pixel_values` height/width must match configured image_size="
                f"{self.image_size}. Received {pixel_values.shape[1:]}"
            )

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, : tokens.size(1), :]

        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError(
                    "`attention_mask` must match pixel grid size."
                )
            if self.patch_size is not None:
                attention_mask = attention_mask.reshape(batch_size, self.image_size//self.patch_size, self.patch_size, self.image_size//self.patch_size, self.patch_size)
                attention_mask = torch.max(torch.max(attention_mask, dim=2)[0], dim=3)[0]
            flat_mask = attention_mask.view(batch_size, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat(
                [torch.zeros(batch_size, self.num_task_tokens, device=device, dtype=torch.bool), pad_mask],
                dim=1,
            )
            key_padding_mask = pad_mask

        encoded = self.encoder(hidden_states, key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        pixel_states = encoded[:, self.num_task_tokens:, :]

        logits = self.head(pixel_states)
        logits = logits.reshape((-1, self.image_size//self.patch_size, self.image_size//self.patch_size, self.patch_size, self.patch_size, self.num_colors))
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(batch_size, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits