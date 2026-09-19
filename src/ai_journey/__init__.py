"""Small, deterministic exercises for the AI Journey repository."""

from .attention import AttentionConfig, run_attention_experiment
from .gradients import gradient_descent_x_squared
from .mlp_memory import run_memory_experiment
from .neural_net import default_parameters, forward, full_backward
from .scalar_autodiff import Value, run_autodiff_experiment
from .transformer_shapes import TransformerConfig, build_gpt_shape_flow
from .xor import train_xor

__all__ = [
    "AttentionConfig",
    "default_parameters",
    "forward",
    "full_backward",
    "gradient_descent_x_squared",
    "run_memory_experiment",
    "run_attention_experiment",
    "run_autodiff_experiment",
    "TransformerConfig",
    "Value",
    "build_gpt_shape_flow",
    "train_xor",
]
