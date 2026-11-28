"""Project 1 (AFD) MoE realism switches in project_param.

moe_single_launch: one kernel launch per grouped expert GEMM instead of one per expert.
moe_active_experts: price only the experts the batch is expected to touch.
Both are off by default. See docs/moe_realism_switches.md.
"""

from pathlib import Path

import pytest

import config
from inference_timing import TimeCalculationLLMInference
from train_timing import GemmType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = PROJECT_ROOT / "configs" / "model-config" / "course_project" / "qwen3_30b_a3b_inf.yaml"
HW_YAML = PROJECT_ROOT / "configs" / "hardware-config" / "course_project" / "a100_80GB_1gpu.yaml"

# Decode FFN1 of Qwen3-30B-A3B: hidden 2048 -> expert width 768.
FFN1 = (1, 2048, 768)


def _timing(**switches):
    hw = config.parse_config(str(HW_YAML), config_type="hardware")
    model = config.parse_config(str(MODEL_YAML), config_type="LLM")
    config.validate_configs(hw, model)
    tc = TimeCalculationLLMInference(hw, model, "LLM")
    tc.project_param = {**tc.project_param, **switches}
    return tc


def _ffn1_time(tc, batch_size, seq_len=1):
    return tc.get_moe_ffn_f(
        FFN1, "test_ffn1_f", gemm_type=GemmType.FFN1,
        batch_size=batch_size, seq_len=seq_len, allow_padding=tc._moe_allow_padding(),
    )[0]


def test_committed_config_has_both_switches_off():
    model = config.parse_config(str(MODEL_YAML), config_type="LLM")
    assert model.project_param.get("moe_single_launch") is False
    assert model.project_param.get("moe_active_experts") is False


@pytest.mark.parametrize(
    "tokens, expected",
    [(1, (8, 1)), (8, (52, 2)), (32, (112, 3)), (2048, (128, 128))],
)
def test_active_expert_count(tokens, expected):
    # E * (1 - (1 - k/E)^n) with E = 128, k = 8; tokens per active expert rounded up.
    assert _timing()._moe_active_experts(tokens, 128) == expected


def test_active_experts_needs_one_gpu_with_every_expert():
    with pytest.raises(NotImplementedError):
        _timing()._moe_active_experts(32, 64)


def test_single_launch_keeps_exactly_one_launch():
    off = _timing(moe_single_launch=False)
    on = _timing(moe_single_launch=True)
    saved = _ffn1_time(off, 32) - _ffn1_time(on, 32)
    assert saved == pytest.approx(127 * off.O, rel=1e-9)


def test_active_experts_cuts_small_batch_time():
    off = _ffn1_time(_timing(moe_active_experts=False), 1)
    on = _ffn1_time(_timing(moe_active_experts=True), 1)
    # 8 of 128 experts: 1/16 of the per-expert GEMMs.
    assert on == pytest.approx(off / 16, rel=1e-9)


def test_active_experts_no_change_when_every_expert_is_busy():
    # 2048 tokens touch all 128 experts, with exactly 128 tokens each, as without the switch.
    off = _ffn1_time(_timing(moe_active_experts=False), 1, seq_len=2048)
    on = _ffn1_time(_timing(moe_active_experts=True), 1, seq_len=2048)
    assert on == off
