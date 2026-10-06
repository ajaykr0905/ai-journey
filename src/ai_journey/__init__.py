"""Small, deterministic exercises for the AI Journey repository."""

from .attention import AttentionConfig, run_attention_experiment
from .batch_normalization import ScratchBatchNorm
from .batchnorm_experiment import (
    BatchNormCriteria,
    BatchNormExperimentResult,
    run_batchnorm_experiment,
)
from .bigram_lm import BigramModel, Vocabulary, run_bigram_experiment
from .context_mlp import (
    ContextDataset,
    ContextMLP,
    build_context_dataset,
    train_context_mlp,
)
from .context_mlp import (
    TrainingResult as ContextTrainingResult,
)
from .corpus_shift import (
    ShiftAssessment,
    ShiftPolicy,
    ShiftReport,
    analyze_corpus_shift,
    assess_corpus_shift,
)
from .gradients import gradient_descent_x_squared
from .initialization_comparison import (
    ComparisonCriteria,
    KaimingComparisonResult,
    run_kaiming_comparison,
)
from .manual_backprop import manual_backward, run_manual_backprop_experiment
from .mlp_memory import run_memory_experiment
from .neural_net import default_parameters, forward, full_backward
from .scalar_autodiff import Value, run_autodiff_experiment
from .tiny_mlp import MLP, TrainingConfig, run_mlp_experiment, train_mlp
from .transformer_lab import (
    DecoderLanguageModel,
    TokenCorpus,
    run_transformer_experiment,
)
from .transformer_lab import (
    TrainingConfig as TransformerTrainingConfig,
)
from .transformer_lab import (
    TransformerConfig as TransformerModelConfig,
)
from .transformer_shapes import TransformerConfig, build_gpt_shape_flow
from .wavenet import (
    FlattenConsecutive,
    HierarchicalLanguageModel,
    WaveNetConfig,
    WaveNetDataset,
    WaveNetTrainingConfig,
    audit_wavenet_gradients,
    build_wavenet_dataset_split,
    run_wavenet_experiment,
    run_wavenet_overfit_probe,
)
from .xor import train_xor

__all__ = [
    "MLP",
    "AttentionConfig",
    "BatchNormCriteria",
    "BatchNormExperimentResult",
    "BigramModel",
    "ComparisonCriteria",
    "ContextDataset",
    "ContextMLP",
    "ContextTrainingResult",
    "DecoderLanguageModel",
    "FlattenConsecutive",
    "HierarchicalLanguageModel",
    "KaimingComparisonResult",
    "ScratchBatchNorm",
    "ShiftAssessment",
    "ShiftPolicy",
    "ShiftReport",
    "TokenCorpus",
    "TrainingConfig",
    "TransformerConfig",
    "TransformerModelConfig",
    "TransformerTrainingConfig",
    "Value",
    "Vocabulary",
    "WaveNetConfig",
    "WaveNetDataset",
    "WaveNetTrainingConfig",
    "analyze_corpus_shift",
    "assess_corpus_shift",
    "audit_wavenet_gradients",
    "build_context_dataset",
    "build_gpt_shape_flow",
    "build_wavenet_dataset_split",
    "default_parameters",
    "forward",
    "full_backward",
    "gradient_descent_x_squared",
    "manual_backward",
    "run_attention_experiment",
    "run_autodiff_experiment",
    "run_batchnorm_experiment",
    "run_bigram_experiment",
    "run_kaiming_comparison",
    "run_manual_backprop_experiment",
    "run_memory_experiment",
    "run_mlp_experiment",
    "run_transformer_experiment",
    "run_wavenet_experiment",
    "run_wavenet_overfit_probe",
    "train_context_mlp",
    "train_mlp",
    "train_xor",
]
