import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every public name a bare ``import ai_journey`` exposes: each export with the
# "submodule:qualname" of the object it resolves to, and the submodules bound as
# package attributes. However the package loads them, each name must keep
# resolving to that same object.
EXPECTED_EXPORTS = {
    "MLP": "tiny_mlp:MLP",
    "AttentionConfig": "attention:AttentionConfig",
    "BatchNormCriteria": "batchnorm_experiment:BatchNormCriteria",
    "BatchNormExperimentResult": "batchnorm_experiment:BatchNormExperimentResult",
    "BigramModel": "bigram_lm:BigramModel",
    "ComparisonCriteria": "initialization_comparison:ComparisonCriteria",
    "ContextDataset": "context_mlp:ContextDataset",
    "ContextMLP": "context_mlp:ContextMLP",
    "ContextTrainingResult": "context_mlp:TrainingResult",
    "DecoderLanguageModel": "transformer_lab:DecoderLanguageModel",
    "FlattenConsecutive": "wavenet:FlattenConsecutive",
    "HierarchicalLanguageModel": "wavenet:HierarchicalLanguageModel",
    "KaimingComparisonResult": "initialization_comparison:KaimingComparisonResult",
    "RebuildCertificationResult": "wavenet_certification:RebuildCertificationResult",
    "RebuildExperimentResult": "wavenet_rebuild:RebuildExperimentResult",
    "RebuiltWaveNet": "wavenet_rebuild:RebuiltWaveNet",
    "ScratchBatchNorm": "batch_normalization:ScratchBatchNorm",
    "ShiftAssessment": "corpus_shift:ShiftAssessment",
    "ShiftPolicy": "corpus_shift:ShiftPolicy",
    "ShiftReport": "corpus_shift:ShiftReport",
    "TokenCorpus": "transformer_lab:TokenCorpus",
    "TrainingConfig": "tiny_mlp:TrainingConfig",
    "TransformerConfig": "transformer_shapes:TransformerConfig",
    "TransformerModelConfig": "transformer_lab:TransformerConfig",
    "TransformerTrainingConfig": "transformer_lab:TrainingConfig",
    "Value": "scalar_autodiff:Value",
    "Vocabulary": "bigram_lm:Vocabulary",
    "WaveNetConfig": "wavenet:WaveNetConfig",
    "WaveNetDataset": "wavenet:WaveNetDataset",
    "WaveNetTrainingConfig": "wavenet:WaveNetTrainingConfig",
    "analyze_corpus_shift": "corpus_shift:analyze_corpus_shift",
    "assess_corpus_shift": "corpus_shift:assess_corpus_shift",
    "audit_rebuild_forward": "wavenet_rebuild:audit_rebuild_forward",
    "audit_rebuild_gradients": "wavenet_rebuild:audit_rebuild_gradients",
    "audit_wavenet_gradients": "wavenet:audit_wavenet_gradients",
    "build_context_dataset": "context_mlp:build_context_dataset",
    "build_gpt_shape_flow": "transformer_shapes:build_gpt_shape_flow",
    "build_wavenet_dataset_split": "wavenet:build_wavenet_dataset_split",
    "certify_rebuild": "wavenet_certification:certify_rebuild",
    "compile_rebuild_plan": "wavenet_rebuild:compile_rebuild_plan",
    "default_parameters": "neural_net:default_parameters",
    "forward": "neural_net:forward",
    "full_backward": "neural_net:full_backward",
    "gradient_descent_x_squared": "gradients:gradient_descent_x_squared",
    "load_certification_report": "wavenet_certification:load_certification_report",
    "manual_backward": "manual_backprop:manual_backward",
    "run_attention_experiment": "attention:run_attention_experiment",
    "run_autodiff_experiment": "scalar_autodiff:run_autodiff_experiment",
    "run_batchnorm_experiment": "batchnorm_experiment:run_batchnorm_experiment",
    "run_bigram_experiment": "bigram_lm:run_bigram_experiment",
    "run_kaiming_comparison": "initialization_comparison:run_kaiming_comparison",
    "run_manual_backprop_experiment": "manual_backprop:run_manual_backprop_experiment",
    "run_memory_experiment": "mlp_memory:run_memory_experiment",
    "run_mlp_experiment": "tiny_mlp:run_mlp_experiment",
    "run_rebuild_experiment": "wavenet_rebuild:run_rebuild_experiment",
    "run_transformer_experiment": "transformer_lab:run_transformer_experiment",
    "run_wavenet_experiment": "wavenet:run_wavenet_experiment",
    "run_wavenet_overfit_probe": "wavenet:run_wavenet_overfit_probe",
    "train_context_mlp": "context_mlp:train_context_mlp",
    "train_mlp": "tiny_mlp:train_mlp",
    "train_xor": "xor:train_xor",
    "verify_certification_report": "wavenet_certification:verify_certification_report",
    "verify_rebuild_report": "wavenet_rebuild:verify_rebuild_report",
    "write_certification_report": "wavenet_certification:write_certification_report",
}
EXPECTED_SUBMODULES = (
    "attention",
    "batch_normalization",
    "batchnorm_experiment",
    "bigram_lm",
    "bpe",
    "context_mlp",
    "corpus_shift",
    "gradients",
    "initialization_comparison",
    "manual_backprop",
    "mlp_memory",
    "neural_net",
    "scalar_autodiff",
    "tiny_mlp",
    "tokenizer_codec",
    "tokenizer_io",
    "transformer_lab",
    "transformer_shapes",
    "wavenet",
    "wavenet_certification",
    "wavenet_rebuild",
    "xor",
)
RESOLVED_NAMES = """
import json, sys, types
import ai_journey
resolved = {}
for name in [name for name in dir(ai_journey) if not name.startswith("_")]:
    value = getattr(ai_journey, name)
    if isinstance(value, types.ModuleType):
        resolved[name] = value.__name__.removeprefix("ai_journey.")
    elif getattr(sys.modules[value.__module__], value.__qualname__) is value:
        module = value.__module__.removeprefix("ai_journey.")
        resolved[name] = f"{module}:{value.__qualname__}"
print(json.dumps({"all": ai_journey.__all__, "resolved": resolved}))
"""


def run_fresh(code):
    """Run code in a new interpreter so no earlier test has imported anything."""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


class PackageImportTests(unittest.TestCase):
    maxDiff = None

    def test_public_names_resolve_to_the_objects_their_submodules_define(self) -> None:
        observed = run_fresh(RESOLVED_NAMES)
        expected = EXPECTED_EXPORTS | {name: name for name in EXPECTED_SUBMODULES}

        self.assertEqual(sorted(observed["all"]), sorted(EXPECTED_EXPORTS))
        self.assertEqual(observed["resolved"], expected)
