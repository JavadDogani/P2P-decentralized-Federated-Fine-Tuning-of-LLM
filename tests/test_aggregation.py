from __future__ import annotations

import pytest
import torch

from p2p_peft_fl.training.aggregation import dlora_aggregate, mean_aggregate, sparsify_topk, wlora_aggregate

# Key names as produced by PEFT's get_peft_model_state_dict for a BERT classifier.
ATTN_KEY = "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight"
ATTN_OUT_KEY = "base_model.model.bert.encoder.layer.0.attention.output.dense.lora_B.weight"
MLP_KEY = "base_model.model.bert.encoder.layer.0.intermediate.dense.lora_A.weight"
OTHER_KEY = "base_model.model.classifier.weight"
KEYS = [ATTN_KEY, ATTN_OUT_KEY, MLP_KEY, OTHER_KEY]


def make_states(n: int = 3, shape=(4, 3)):
    g = torch.Generator().manual_seed(0)
    return [{k: torch.randn(shape, generator=g) for k in KEYS} for _ in range(n)]


def test_mean_is_an_exact_average():
    # integer-valued tensors so the expected mean is exactly representable
    states = [
        {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([0.0, 3.0])},
        {"a": torch.tensor([[3.0, 4.0], [5.0, 6.0]]), "b": torch.tensor([3.0, 6.0])},
        {"a": torch.tensor([[5.0, 0.0], [7.0, 2.0]]), "b": torch.tensor([6.0, 0.0])},
    ]
    out = mean_aggregate(states)
    assert torch.equal(out["a"], torch.tensor([[3.0, 2.0], [5.0, 4.0]]))
    assert torch.equal(out["b"], torch.tensor([3.0, 3.0]))


def test_mean_matches_manual_average_and_preserves_shape_and_dtype():
    states = make_states(4, shape=(8, 5))
    out = mean_aggregate(states)
    assert list(out) == KEYS
    for k in KEYS:
        expected = sum(s[k] for s in states) / len(states)
        assert out[k].shape == states[0][k].shape
        assert out[k].dtype == states[0][k].dtype
        torch.testing.assert_close(out[k], expected)


def test_mean_of_a_single_state_is_identity():
    (state,) = make_states(1)
    out = mean_aggregate([state])
    for k in KEYS:
        assert torch.equal(out[k], state[k])


def test_mean_does_not_modify_inputs():
    states = make_states(3)
    before = [{k: v.clone() for k, v in s.items()} for s in states]
    mean_aggregate(states)
    for s, b in zip(states, before):
        for k in KEYS:
            assert torch.equal(s[k], b[k])


@pytest.mark.parametrize("fn", [mean_aggregate, lambda s: wlora_aggregate(s, 1.0, 1.0)])
def test_empty_input_raises(fn):
    with pytest.raises(ValueError):
        fn([])


def test_wlora_scales_the_mean_by_key_pattern():
    states = make_states(3)
    mean = mean_aggregate(states)
    out = wlora_aggregate(states, attn_mult=1.5, mlp_mult=0.5)
    torch.testing.assert_close(out[ATTN_KEY], mean[ATTN_KEY] * 1.5)
    # attention patterns are checked first, so attention.output.dense counts as attention
    torch.testing.assert_close(out[ATTN_OUT_KEY], mean[ATTN_OUT_KEY] * 1.5)
    torch.testing.assert_close(out[MLP_KEY], mean[MLP_KEY] * 0.5)
    # keys matching neither pattern (here: the classifier head) are left at the plain mean
    torch.testing.assert_close(out[OTHER_KEY], mean[OTHER_KEY])
    for k in KEYS:
        assert out[k].shape == states[0][k].shape


def test_wlora_with_unit_multipliers_equals_mean():
    states = make_states(3)
    mean = mean_aggregate(states)
    out = wlora_aggregate(states, attn_mult=1.0, mlp_mult=1.0)
    for k in KEYS:
        torch.testing.assert_close(out[k], mean[k])


def test_dlora_first_round_returns_the_mean():
    mean = mean_aggregate(make_states(3))
    assert dlora_aggregate(None, mean, beta=0.6) is mean


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.6, 1.0])
def test_dlora_is_an_ema_of_previous_and_new(beta):
    prev, new = make_states(2)
    out = dlora_aggregate(prev, new, beta=beta)
    for k in KEYS:
        torch.testing.assert_close(out[k], beta * prev[k] + (1 - beta) * new[k])
        assert out[k].shape == new[k].shape
    if beta == 0.0:
        for k in KEYS:
            torch.testing.assert_close(out[k], new[k])
    if beta == 1.0:
        for k in KEYS:
            torch.testing.assert_close(out[k], prev[k])


def test_sparsify_keep_all_is_a_noop():
    state = make_states(1)[0]
    assert sparsify_topk(state, 1.0) is state


def test_sparsify_keeps_largest_magnitudes_per_tensor():
    state = {"w": torch.tensor([[0.1, -5.0, 0.3], [2.0, -0.2, 4.0]]), "b": torch.tensor([1.0, -3.0])}
    out = sparsify_topk(state, 0.5)
    assert torch.equal(out["w"], torch.tensor([[0.0, -5.0, 0.0], [2.0, 0.0, 4.0]]))
    assert torch.equal(out["b"], torch.tensor([0.0, -3.0]))


def test_sparsify_keeps_at_least_one_entry_and_preserves_shapes():
    state = make_states(1, shape=(6, 7))[0]
    out = sparsify_topk(state, 0.001)
    for k, v in state.items():
        assert out[k].shape == v.shape
        assert int((out[k] != 0).sum()) == 1
        idx = v.abs().argmax()
        assert out[k].view(-1)[idx] == v.view(-1)[idx]
