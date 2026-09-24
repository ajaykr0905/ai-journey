"""Small, deterministic exercises for the AI Journey repository."""

from .attention import AttentionConfig, run_attention_experiment
from .bigram_lm import BigramModel, Vocabulary, run_bigram_experiment
from .corpus_shift import ShiftReport, analyze_corpus_shift
from .gradients import gradient_descent_x_squared
from .manual_backprop import manual_backward, run_manual_backprop_experiment
from .mlp_memory import run_memory_experiment
from .neural_net import default_parameters, forward, full_backward
from .scalar_autodiff import Value, run_autodiff_experiment
from .tiny_mlp import MLP, TrainingConfig, run_mlp_experiment, train_mlp
from .transformer_shapes import TransformerConfig, build_gpt_shape_flow
from .xor import train_xor

__all__ = [
    "MLP",
    "AttentionConfig",
    "BigramModel",
    "ShiftReport",
    "TrainingConfig",
    "TransformerConfig",
    "Value",
    "Vocabulary",
    "analyze_corpus_shift",
    "build_gpt_shape_flow",
    "default_parameters",
    "forward",
    "full_backward",
    "gradient_descent_x_squared",
    "manual_backward",
    "run_attention_experiment",
    "run_autodiff_experiment",
    "run_bigram_experiment",
    "run_manual_backprop_experiment",
    "run_memory_experiment",
    "run_mlp_experiment",
    "train_mlp",
    "train_xor",
]
