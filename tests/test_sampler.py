"""
Tests for core/sampler.py — SegviGenFlowSampler.

Monkeypatches trellis2.samplers.FlowEulerGuidanceIntervalSampler at the
SOURCE module level (not at core.sampler level, since the import is lazy).
"""
import sys
import types
import pytest


class _MockSamplerOutput:
    def __init__(self):
        self.samples = "mock_samples"


class _MockFlowSampler:
    """Records kwargs passed to .sample() for assertion."""
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.last_sample_kwargs = {}

    def sample(self, model, noise, **kwargs):
        self.last_sample_kwargs = kwargs
        return _MockSamplerOutput()


@pytest.fixture
def mock_trellis_sampler(monkeypatch):
    """Install a mock FlowEulerGuidanceIntervalSampler into trellis2.samplers."""
    # Ensure the module path exists in sys.modules
    if "trellis2" not in sys.modules:
        sys.modules["trellis2"] = types.ModuleType("trellis2")
    if "trellis2.samplers" not in sys.modules:
        samplers_mod = types.ModuleType("trellis2.samplers")
        sys.modules["trellis2.samplers"] = samplers_mod
        sys.modules["trellis2"].samplers = samplers_mod

    # Pre-set the attribute so monkeypatch.setattr can find it
    samplers_mod = sys.modules["trellis2.samplers"]
    if not hasattr(samplers_mod, "FlowEulerGuidanceIntervalSampler"):
        samplers_mod.FlowEulerGuidanceIntervalSampler = None

    captured = {}

    def _mock_cls(sigma_min=1e-5):
        inst = _MockFlowSampler(sigma_min=sigma_min)
        captured["instance"] = inst
        return inst

    monkeypatch.setattr(
        "trellis2.samplers.FlowEulerGuidanceIntervalSampler",
        _mock_cls,
    )
    return captured


def test_flow_sampler_guidance_rescale_forwarded(mock_trellis_sampler):
    from core.sampler import SegviGenFlowSampler
    sampler = SegviGenFlowSampler(sigma_min=1e-5)
    sampler.sample(
        model="dummy_model",
        noise="dummy_noise",
        cond="pos",
        neg_cond="neg",
        steps=10,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 0.9),
    )
    inst = mock_trellis_sampler["instance"]
    assert inst.last_sample_kwargs["guidance_rescale"] == 0.7


def test_flow_sampler_rescale_t_forwarded(mock_trellis_sampler):
    from core.sampler import SegviGenFlowSampler
    sampler = SegviGenFlowSampler(sigma_min=1e-5)
    sampler.sample(
        model="dummy_model",
        noise="dummy_noise",
        cond="pos",
        neg_cond="neg",
        steps=10,
        guidance_strength=7.5,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        rescale_t=0.5,
    )
    inst = mock_trellis_sampler["instance"]
    assert inst.last_sample_kwargs["rescale_t"] == 0.5


def test_flow_sampler_extra_kwargs_forwarded(mock_trellis_sampler):
    from core.sampler import SegviGenFlowSampler
    sampler = SegviGenFlowSampler(sigma_min=1e-5)
    sampler.sample(
        model="dummy_model",
        noise="dummy_noise",
        cond="pos",
        neg_cond="neg",
        steps=10,
        guidance_strength=7.5,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        input_points="mock_points",
        concat_cond="mock_cond",
    )
    inst = mock_trellis_sampler["instance"]
    assert inst.last_sample_kwargs["input_points"] == "mock_points"
    assert inst.last_sample_kwargs["concat_cond"] == "mock_cond"
