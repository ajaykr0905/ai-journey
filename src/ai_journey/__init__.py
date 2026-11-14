"""Small, deterministic exercises for the AI Journey repository.

The package-level exports load on first access (PEP 562) instead of at import
time. Many of them import torch or numpy, so importing them eagerly would make
every ``import ai_journey.<module>`` load torch first, even for the stdlib-only
labs and their command-line tools. Attribute access, ``from ai_journey import``,
``__all__`` and ``dir()`` expose every export, bound to the object its
submodule defines.
"""

from importlib import import_module as _import_module

# Defining submodule -> the names the package re-exports from it unchanged.
_EXPORTS_BY_MODULE = {
    "attention": ("AttentionConfig", "run_attention_experiment"),
    "batch_normalization": ("ScratchBatchNorm",),
    "batchnorm_experiment": (
        "BatchNormCriteria",
        "BatchNormExperimentResult",
        "run_batchnorm_experiment",
    ),
    "bigram_lm": ("BigramModel", "Vocabulary", "run_bigram_experiment"),
    "context_mlp": (
        "ContextDataset",
        "ContextMLP",
        "build_context_dataset",
        "train_context_mlp",
    ),
    "corpus_shift": (
        "ShiftAssessment",
        "ShiftPolicy",
        "ShiftReport",
        "analyze_corpus_shift",
        "assess_corpus_shift",
    ),
    "gradients": ("gradient_descent_x_squared",),
    "initialization_comparison": (
        "ComparisonCriteria",
        "KaimingComparisonResult",
        "run_kaiming_comparison",
    ),
    "manual_backprop": ("manual_backward", "run_manual_backprop_experiment"),
    "mlp_memory": ("run_memory_experiment",),
    "neural_net": ("default_parameters", "forward", "full_backward"),
    "scalar_autodiff": ("Value", "run_autodiff_experiment"),
    "tiny_mlp": ("MLP", "TrainingConfig", "run_mlp_experiment", "train_mlp"),
    "transformer_lab": (
        "DecoderLanguageModel",
        "TokenCorpus",
        "run_transformer_experiment",
    ),
    "transformer_shapes": ("TransformerConfig", "build_gpt_shape_flow"),
    "wavenet": (
        "FlattenConsecutive",
        "HierarchicalLanguageModel",
        "WaveNetConfig",
        "WaveNetDataset",
        "WaveNetTrainingConfig",
        "audit_wavenet_gradients",
        "build_wavenet_dataset_split",
        "run_wavenet_experiment",
        "run_wavenet_overfit_probe",
    ),
    "wavenet_certification": (
        "RebuildCertificationResult",
        "certify_rebuild",
        "load_certification_report",
        "verify_certification_report",
        "write_certification_report",
    ),
    "wavenet_rebuild": (
        "RebuildExperimentResult",
        "RebuiltWaveNet",
        "audit_rebuild_forward",
        "audit_rebuild_gradients",
        "compile_rebuild_plan",
        "run_rebuild_experiment",
        "verify_rebuild_report",
    ),
    "xor": ("train_xor",),
}
# Exports published under another name than the one their submodule defines.
_RENAMED_EXPORTS = {
    "ContextTrainingResult": ("context_mlp", "TrainingResult"),
    "TransformerModelConfig": ("transformer_lab", "TransformerConfig"),
    "TransformerTrainingConfig": ("transformer_lab", "TrainingConfig"),
}
# Public name -> (submodule, attribute) for every name in ``__all__``.
_EXPORTS = {
    name: (module, name)
    for module, names in _EXPORTS_BY_MODULE.items()
    for name in names
} | _RENAMED_EXPORTS
# Submodules that ``import ai_journey`` bound as attributes while its exports were
# eager (the last three via transformer_lab); attribute access still imports them.
_SUBMODULES = frozenset(_EXPORTS_BY_MODULE) | {"bpe", "tokenizer_codec", "tokenizer_io"}

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
    "RebuildCertificationResult",
    "RebuildExperimentResult",
    "RebuiltWaveNet",
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
    "audit_rebuild_forward",
    "audit_rebuild_gradients",
    "audit_wavenet_gradients",
    "build_context_dataset",
    "build_gpt_shape_flow",
    "build_wavenet_dataset_split",
    "certify_rebuild",
    "compile_rebuild_plan",
    "default_parameters",
    "forward",
    "full_backward",
    "gradient_descent_x_squared",
    "load_certification_report",
    "manual_backward",
    "run_attention_experiment",
    "run_autodiff_experiment",
    "run_batchnorm_experiment",
    "run_bigram_experiment",
    "run_kaiming_comparison",
    "run_manual_backprop_experiment",
    "run_memory_experiment",
    "run_mlp_experiment",
    "run_rebuild_experiment",
    "run_transformer_experiment",
    "run_wavenet_experiment",
    "run_wavenet_overfit_probe",
    "train_context_mlp",
    "train_mlp",
    "train_xor",
    "verify_certification_report",
    "verify_rebuild_report",
    "write_certification_report",
]


def __getattr__(name: str) -> object:
    """Import the submodule behind an export or listed submodule on first access.

    Exports are then cached in the package namespace and the import system binds
    submodules itself, so each name reaches this hook once; any other name raises
    the same AttributeError as a missing module attribute.
    """

    if name in _SUBMODULES:
        return _import_module(f"{__name__}.{name}")
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unloaded exports and submodules, so dir() and completion see them."""

    return sorted(set(globals()) | set(_EXPORTS) | _SUBMODULES)
