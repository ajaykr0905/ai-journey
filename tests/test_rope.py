from __future__ import annotations

import math

import pytest
import torch

from ai_journey.rope import apply_rotary_position_embeddings


def _reference_rotation(values: torch.Tensor, offset: int, base: float) -> torch.Tensor:
    expected = torch.empty_like(values)
    head_dim = values.shape[-1]
    for time in range(values.shape[2]):
        for feature in range(0, head_dim, 2):
            angle = (time + offset) * base ** (-feature / head_dim)
            cosine, sine = math.cos(angle), math.sin(angle)
            left = values[:, :, time, feature]
            right = values[:, :, time, feature + 1]
            expected[:, :, time, feature] = left * cosine - right * sine
            expected[:, :, time, feature + 1] = left * sine + right * cosine
    return expected


def test_rope_matches_independent_pairwise_reference_with_grouped_query_heads() -> None:
    query = torch.arange(48, dtype=torch.float64).reshape(1, 4, 3, 4) / 10
    key = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4) / 7

    rotated_query, rotated_key = apply_rotary_position_embeddings(
        query, key, position_offset=7, base=100.0
    )

    torch.testing.assert_close(
        rotated_query, _reference_rotation(query, 7, 100.0), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        rotated_key, _reference_rotation(key, 7, 100.0), rtol=0, atol=1e-14
    )


def test_rope_preserves_pair_norm_and_position_zero_is_identity() -> None:
    generator = torch.Generator().manual_seed(52)
    query = torch.randn(2, 3, 5, 8, dtype=torch.float64, generator=generator)
    key = torch.randn(2, 1, 5, 8, dtype=torch.float64, generator=generator)

    rotated_query, rotated_key = apply_rotary_position_embeddings(query, key)

    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0], rtol=0, atol=0)
    for original, rotated in ((query, rotated_query), (key, rotated_key)):
        before = original.reshape(*original.shape[:-1], -1, 2).square().sum(-1)
        after = rotated.reshape(*rotated.shape[:-1], -1, 2).square().sum(-1)
        torch.testing.assert_close(after, before, rtol=0, atol=1e-14)


def test_cached_decode_offset_matches_full_sequence_rotation() -> None:
    generator = torch.Generator().manual_seed(52)
    query = torch.randn(2, 4, 7, 8, dtype=torch.float64, generator=generator)
    key = torch.randn(2, 2, 7, 8, dtype=torch.float64, generator=generator)

    full_query, full_key = apply_rotary_position_embeddings(query, key)
    chunk_query, chunk_key = apply_rotary_position_embeddings(
        query[:, :, 2:5], key[:, :, 2:5], position_offset=2
    )

    torch.testing.assert_close(chunk_query, full_query[:, :, 2:5], rtol=0, atol=0)
    torch.testing.assert_close(chunk_key, full_key[:, :, 2:5], rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16]
)
def test_rope_preserves_shape_dtype_and_gradients(dtype: torch.dtype) -> None:
    query = torch.randn(2, 3, 4, 6, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 1, 4, 6, dtype=dtype, requires_grad=True)
    query_before, key_before = query.detach().clone(), key.detach().clone()

    rotated_query, rotated_key = apply_rotary_position_embeddings(query, key)
    (rotated_query.square().sum() + rotated_key.square().sum()).backward()

    assert rotated_query.shape == query.shape and rotated_key.shape == key.shape
    assert rotated_query.dtype == query.dtype and rotated_key.dtype == key.dtype
    assert query.grad is not None and key.grad is not None
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    assert torch.equal(query, query_before) and torch.equal(key, key_before)


@pytest.mark.parametrize(
    ("query", "key", "error", "message"),
    [
        ([], torch.ones(1, 1, 1, 2), TypeError, "torch.Tensor"),
        (torch.ones(1, 1, 1, 2), [], TypeError, "torch.Tensor"),
        (
            torch.ones(1, 1, 1, 2, dtype=torch.long),
            torch.ones(1, 1, 1, 2),
            TypeError,
            "floating-point",
        ),
        (torch.ones(1, 1, 2), torch.ones(1, 1, 1, 2), ValueError, "non-empty"),
        (torch.empty(1, 1, 0, 2), torch.empty(1, 1, 0, 2), ValueError, "non-empty"),
        (torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3), ValueError, "even"),
        (torch.ones(2, 1, 1, 2), torch.ones(1, 1, 1, 2), ValueError, "share batch"),
        (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 2), ValueError, "share batch"),
        (torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 4), ValueError, "share batch"),
        (
            torch.ones(1, 1, 1, 2),
            torch.ones(1, 1, 1, 2, dtype=torch.float64),
            ValueError,
            "share dtype",
        ),
        (
            torch.ones(1, 1, 1, 2),
            torch.empty(1, 1, 1, 2, device="meta"),
            ValueError,
            "share dtype and device",
        ),
    ],
)
def test_rope_rejects_invalid_input_contracts(
    query: object, key: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        apply_rotary_position_embeddings(query, key)  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [-1, True, 1.5, 2**53])
def test_rope_rejects_invalid_position_offsets(offset: object) -> None:
    values = torch.ones(1, 1, 1, 2)

    with pytest.raises(ValueError, match="position_offset"):
        apply_rotary_position_embeddings(values, values, position_offset=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "base", [1.0, 0.0, -10.0, float("inf"), float("nan"), True, 10**1000]
)
def test_rope_rejects_invalid_frequency_bases(base: object) -> None:
    values = torch.ones(1, 1, 1, 2)

    with pytest.raises(ValueError, match="base"):
        apply_rotary_position_embeddings(values, values, base=base)  # type: ignore[arg-type]
