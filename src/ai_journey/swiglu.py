"""Position-wise SwiGLU feed-forward primitive."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Project ``silu(gate(x)) * up(x)`` back to the input width.

    The caller chooses the hidden width explicitly so model-level parameter
    budgets are not silently tied to one rounding convention.
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, *, bias: bool = False
    ) -> None:
        super().__init__()
        if (
            type(embedding_dim) is not int
            or embedding_dim <= 0
            or type(hidden_dim) is not int
            or hidden_dim <= 0
        ):
            raise ValueError("embedding_dim and hidden_dim must be positive integers")
        if type(bias) is not bool:
            raise TypeError("bias must be a bool")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("SwiGLU inputs must be a torch.Tensor")
        if not inputs.is_floating_point():
            raise TypeError("SwiGLU inputs must have a floating-point dtype")
        if inputs.ndim == 0 or inputs.shape[-1] != self.embedding_dim:
            raise ValueError("SwiGLU inputs must end in embedding_dim")

        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
