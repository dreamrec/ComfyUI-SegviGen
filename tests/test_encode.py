"""
Tests for core/encode.py — extract_shape_data and sample_tex_slat.

All tests monkeypatch _load_trellis2_stages to return a mock stages module,
so no TRELLIS2 installation or GPU is needed.
"""
import sys
import types
import pytest
import torch


class _MockSparseTensor:
    """Minimal SparseTensor stand-in for testing."""
    def __init__(self, feats, coords):
        self.feats = feats
        self.coords = coords


def _make_mock_stages(expected_model_key=None):
    """Build a mock stages module with stub functions."""
    stages = types.ModuleType("mock_stages")
    stages._init_config_called = False
    stages._sample_tex_slat_calls = []
    stages._pipeline_config = {"args": {}}

    def _init_config():
        stages._init_config_called = True
    stages._init_config = _init_config

    def _deserialize_from_ipc(data, device):
        if isinstance(data, dict) and "mock" in data:
            return _MockSparseTensor(
                feats=torch.randn(100, 32),
                coords=torch.zeros(100, 4, dtype=torch.int32),
            )
        # Return a list for subs
        if isinstance(data, list):
            return [_MockSparseTensor(
                feats=torch.randn(50, 32),
                coords=torch.zeros(50, 4, dtype=torch.int32),
            )]
        return data
    stages._deserialize_from_ipc = _deserialize_from_ipc

    def _sample_tex_slat(tex_cond, model_key, shape_slat, sampler_params, device, dtype):
        stages._sample_tex_slat_calls.append({
            "model_key": model_key,
            "sampler_params": sampler_params,
        })
        return _MockSparseTensor(
            feats=torch.randn(100, 32),
            coords=torch.zeros(100, 4, dtype=torch.int32),
        )
    stages._sample_tex_slat = _sample_tex_slat

    return stages


@pytest.fixture
def mock_stages(monkeypatch):
    stages = _make_mock_stages()
    monkeypatch.setattr(
        "core.trellis2_shim.load_trellis2_stages",
        lambda: stages,
    )
    # Mock trellis2.modules.sparse.SparseTensor for _deserialize_sparse_tensor
    if "trellis2" not in sys.modules:
        sys.modules["trellis2"] = types.ModuleType("trellis2")
    if "trellis2.modules" not in sys.modules:
        mods = types.ModuleType("trellis2.modules")
        sys.modules["trellis2.modules"] = mods
        sys.modules["trellis2"].modules = mods
    if "trellis2.modules.sparse" not in sys.modules:
        sparse_mod = types.ModuleType("trellis2.modules.sparse")
        sparse_mod.SparseTensor = _MockSparseTensor
        sys.modules["trellis2.modules.sparse"] = sparse_mod
        sys.modules["trellis2.modules"].sparse = sparse_mod
    else:
        monkeypatch.setattr("trellis2.modules.sparse.SparseTensor", _MockSparseTensor)
    return stages


def _mock_ipc_sparse_tensor(n=100, ch=32):
    """Create a mock IPC-serialized SparseTensor dict matching real format."""
    return {
        "_type": "SparseTensor",
        "feats": torch.randn(n, ch),
        "coords": torch.cat([
            torch.zeros(n, 1, dtype=torch.int32),
            torch.randint(0, 64, (n, 3), dtype=torch.int32),
        ], dim=1),
    }


@pytest.fixture
def mock_shape_result():
    return {
        "shape_slat": _mock_ipc_sparse_tensor(),
        "subs": [_mock_ipc_sparse_tensor(50, 32)],
        "resolution": 512,
        "pipeline_type": "512",
    }


@pytest.fixture
def mock_conditioning():
    return {
        "cond_512": torch.randn(1, 768),
        "neg_cond": torch.randn(1, 768),
    }


def test_extract_shape_data_returns_tuple(mock_stages, mock_shape_result):
    from core.encode import extract_shape_data
    result = extract_shape_data(mock_shape_result, "cpu")
    assert len(result) == 4
    shape_slat, subs, resolution, pipeline_type = result
    assert hasattr(shape_slat, "feats")
    assert resolution == 512
    assert pipeline_type == "512"


def test_sample_tex_slat_calls_stages(mock_stages, mock_shape_result, mock_conditioning):
    from core.encode import sample_tex_slat
    result = sample_tex_slat(
        mock_shape_result, mock_conditioning, "cpu",
        seed=42, tex_guidance_strength=7.5, tex_sampling_steps=12,
    )
    assert hasattr(result, "feats")
    assert len(mock_stages._sample_tex_slat_calls) == 1
    call = mock_stages._sample_tex_slat_calls[0]
    assert call["sampler_params"]["steps"] == 12
    assert call["sampler_params"]["guidance_strength"] == 7.5


def test_sample_tex_slat_512_model_key(mock_stages, mock_conditioning):
    from core.encode import sample_tex_slat
    shape_result_512 = {
        "shape_slat": _mock_ipc_sparse_tensor(),
        "pipeline_type": "512",
    }
    sample_tex_slat(shape_result_512, mock_conditioning, "cpu")
    assert mock_stages._sample_tex_slat_calls[0]["model_key"] == "tex_slat_flow_model_512"


def test_sample_tex_slat_1024_model_key(mock_stages):
    from core.encode import sample_tex_slat
    shape_result_1024 = {
        "shape_slat": _mock_ipc_sparse_tensor(),
        "pipeline_type": "1024",
    }
    conditioning_1024 = {
        "cond_512": torch.randn(1, 768),
        "cond_1024": torch.randn(1, 768),
        "neg_cond": torch.randn(1, 768),
    }
    sample_tex_slat(shape_result_1024, conditioning_1024, "cpu")
    assert mock_stages._sample_tex_slat_calls[0]["model_key"] == "tex_slat_flow_model_1024"


def test_sample_tex_slat_1024_falls_back_to_512(mock_stages, mock_conditioning):
    """When pipeline_type=1024 but cond_1024 is missing, should fall back to cond_512."""
    from core.encode import sample_tex_slat
    shape_result_1024 = {
        "shape_slat": _mock_ipc_sparse_tensor(),
        "pipeline_type": "1024",
    }
    # mock_conditioning has no cond_1024
    sample_tex_slat(shape_result_1024, mock_conditioning, "cpu")
    # Should still use the 1024 model key, just with 512 conditioning
    assert mock_stages._sample_tex_slat_calls[0]["model_key"] == "tex_slat_flow_model_1024"
