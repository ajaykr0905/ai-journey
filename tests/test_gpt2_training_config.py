from hashlib import sha256
from pathlib import Path

from ai_journey.gpt2_model import expected_gpt2_parameter_count
from ai_journey.gpt2_training import GPT2ByteCorpus, GPT2BatchCursor, load_training_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config" / "gpt2-training.json"
SOURCE = ROOT / "data" / "day-19-demo-names.txt"


def test_checked_in_training_protocol_matches_public_source():
    spec = load_training_spec(SPEC)
    source = SOURCE.read_bytes()
    corpus = GPT2ByteCorpus.from_bytes(source)
    cursor = GPT2BatchCursor(corpus, spec.training)

    assert sha256(source).hexdigest() == spec.source.sha256
    assert spec.source.license == "CC0-1.0"
    assert spec.training.effective_batch_size == 8
    assert spec.training.tokens_per_step == 64
    assert spec.training.total_target_tokens == 768
    assert expected_gpt2_parameter_count(spec.model) == 7_536
    assert cursor.discarded_samples_per_epoch == 0


def test_checked_in_protocol_changes_every_declared_training_dimension():
    spec = load_training_spec(SPEC)

    assert spec.training.accumulation_steps > 1
    assert spec.training.warmup_steps > 0
    assert spec.training.max_learning_rate > spec.training.min_learning_rate
    assert spec.training.weight_decay > 0
    assert spec.training.gradient_clip > 0
