"""Unit tests for tier 2.5 — CLIP-style embedding keyword scoring.

Covers:
- softmax_from_similarities math (distribution, candidate-set dependence,
  missing candidates, degenerate temperature)
- score_keywords_embedding adapter (calibrated=False, caveat notes)
- orchestrator rescue: tier 2 failure -> tier 2.5 -> tier 0
- TransformersCLIPScorer lazy loading + reuse (mocked transformers)
"""

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if "src.core" not in sys.modules:
    sys.path.insert(0, str(sys.path[0] or "."))

import pytest

from src.core.keyword_scoring import (
    SUM_TO_ONE_TOLERANCE,
    SCORING_TIER,
    softmax_from_similarities,
)
from src.core.keyword_scoring_adapters import (
    TransformersCLIPScorer,
    pick_embedding_tier,
    score_keywords,
    score_keywords_embedding,
)


# ---------------------------------------------------------------------------
# softmax_from_similarities
# ---------------------------------------------------------------------------


def test_similarity_softmax_sums_to_one():
    candidates = ["cat", "dog", "wallpaper"]
    sims = {"cat": 0.32, "dog": 0.21, "wallpaper": 0.05}
    result = softmax_from_similarities(sims, candidates)
    assert abs(sum(result.values()) - 1.0) <= SUM_TO_ONE_TOLERANCE
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_similarity_softmax_preserves_rank_ordering():
    sims = {"cat": 0.32, "dog": 0.21, "wallpaper": 0.05}
    result = softmax_from_similarities(sims, list(sims))
    assert result["cat"] > result["dog"] > result["wallpaper"]


def test_similarity_softmax_sharper_than_linear_gap():
    """CLIP sims differ by hundredths; the temperature must amplify that
    into a decisive distribution (0.32 vs 0.05 is not 'close')."""
    result = softmax_from_similarities(
        {"cat": 0.32, "wallpaper": 0.05}, ["cat", "wallpaper"]
    )
    assert result["cat"] > 0.9


def test_similarity_missing_candidates_default_to_minus_one():
    result = softmax_from_similarities({"cat": 0.3}, ["cat", "dog"])
    assert result["dog"] < result["cat"]
    assert abs(sum(result.values()) - 1.0) <= SUM_TO_ONE_TOLERANCE


def test_similarity_temperature_sharpness_control():
    sims = {"cat": 0.30, "dog": 0.25}
    sharp = softmax_from_similarities(sims, ["cat", "dog"], temperature=0.01)
    soft = softmax_from_similarities(sims, ["cat", "dog"], temperature=1.0)
    assert sharp["cat"] > soft["cat"]
    assert abs(soft["cat"] - 0.5) < abs(sharp["cat"] - 0.5)


def test_similarity_nonpositive_temperature_falls_back_to_default():
    sims = {"cat": 0.32, "dog": 0.05}
    result = softmax_from_similarities(sims, ["cat", "dog"], temperature=0.0)
    expected = softmax_from_similarities(sims, ["cat", "dog"], temperature=0.01)
    assert result == expected


def test_similarity_empty_candidates_returns_empty():
    assert softmax_from_similarities({"cat": 0.3}, []) == {}


def test_similarity_candidate_set_dependence_is_real():
    """The core calibration caveat: adding a distractor candidate absorbs
    probability mass and shifts every other score for the SAME image."""
    base_sims = {"cat": 0.32, "dog": 0.21}
    without = softmax_from_similarities(base_sims, ["cat", "dog"])
    with_distractor = softmax_from_similarities(
        {**base_sims, "wallpaper": 0.05}, ["cat", "dog", "wallpaper"]
    )
    assert without["cat"] > with_distractor["cat"]
    assert with_distractor["cat"] < without["cat"] + without["dog"] - 1e-9


# ---------------------------------------------------------------------------
# pick_embedding_tier
# ---------------------------------------------------------------------------


def _engine(provider="local", task="image-classification"):
    return SimpleNamespace(
        provider=provider,
        task=task,
        probability_mode="both",
        probability_candidates=["cat", "dog"],
        probability_threshold=0.0,
        probability_enabled=True,
        embedding_rescue_enabled=True,
        device="cpu",
    )


def test_embedding_gate_local_classification_only():
    assert pick_embedding_tier(_engine()) is True
    assert pick_embedding_tier(_engine(task="image-text-to-text")) is False
    assert pick_embedding_tier(_engine(provider="openrouter")) is False


# ---------------------------------------------------------------------------
# score_keywords_embedding adapter
# ---------------------------------------------------------------------------


def test_embedding_adapter_produces_uncalibrated_distribution():
    class Scorer:
        def cosine_similarities(self, image_path, candidates):
            return {c: 0.3 - i * 0.05 for i, c in enumerate(candidates)}

    result = score_keywords_embedding(Scorer(), "img.jpg", ["cat", "dog", "fox"])
    assert result.tier is SCORING_TIER.EMBEDDING
    assert result.calibrated is False  # NOT calibrated, despite summing to 1
    assert abs(sum(result.score_map.values()) - 1.0) <= SUM_TO_ONE_TOLERANCE
    assert result.score_map["cat"] > result.score_map["dog"] > result.score_map["fox"]
    assert all(s.match_type == "semantic" for s in result.scores)


def test_embedding_adapter_notes_carry_the_caveats():
    class Scorer:
        def cosine_similarities(self, image_path, candidates):
            return {c: 0.2 for c in candidates}

    result = score_keywords_embedding(Scorer(), "img.jpg", ["cat", "dog"])
    joined = " ".join(result.notes)
    assert "not calibrated" in joined
    assert "candidate set" in joined
    assert "rank ordering" in joined


def test_embedding_adapter_empty_candidates_is_tier0():
    result = score_keywords_embedding(MagicMock(), "img.jpg", [])
    assert result.tier is SCORING_TIER.UNAVAILABLE
    assert result.calibrated is False


# ---------------------------------------------------------------------------
# Orchestrator: tier 2 failure -> tier 2.5 rescue -> tier 0
# ---------------------------------------------------------------------------


def test_orchestrator_rescues_failed_tier2_with_embedding(tmp_path):
    """Candidates that match no classification label (tier 2 raises) are
    scored by the embedding tier instead of degrading straight to tier 0."""
    engine = _engine()

    class Scorer:
        def cosine_similarities(self, image_path, candidates):
            return {c: 0.25 for c in candidates}

    with patch(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer",
        return_value=Scorer(),
    ) as mock_cls, patch(
        "src.core.huggingface_utils.run_local_logprob_inference",
        side_effect=ValueError("none of the candidate tokens matched"),
    ):
        result = score_keywords(engine, "img.jpg", local_pipeline=MagicMock())

    mock_cls.assert_called_once()
    assert result.tier is SCORING_TIER.EMBEDDING
    assert result.calibrated is False
    assert result.score_map == {"cat": 0.5, "dog": 0.5}


def test_orchestrator_no_rescue_when_opted_out():
    """With the rescue flag off (the default), a failed tier 2 degrades
    straight to tier 0 and CLIP is never constructed or downloaded."""
    engine = _engine()
    engine.embedding_rescue_enabled = False

    with patch(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer"
    ) as mock_cls, patch(
        "src.core.huggingface_utils.run_local_logprob_inference",
        side_effect=ValueError("none of the candidate tokens matched"),
    ):
        result = score_keywords(engine, "img.jpg", local_pipeline=MagicMock())

    mock_cls.assert_not_called()
    assert result.tier is SCORING_TIER.UNAVAILABLE
    assert result.calibrated is False


def test_orchestrator_legacy_engine_without_flag_does_not_rescue():
    """Engines whose config predates ``embedding_rescue_enabled`` are treated
    as opted-out: getattr defaults to False, preserving legacy behavior."""
    engine = _engine()
    del engine.embedding_rescue_enabled

    with patch(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer"
    ) as mock_cls, patch(
        "src.core.huggingface_utils.run_local_logprob_inference",
        side_effect=ValueError("none of the candidate tokens matched"),
    ):
        result = score_keywords(engine, "img.jpg", local_pipeline=MagicMock())

    mock_cls.assert_not_called()
    assert result.tier is SCORING_TIER.UNAVAILABLE


def test_orchestrator_tier0_when_embedding_backend_unavailable():
    engine = _engine()

    with patch(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer",
        side_effect=OSError("model not downloaded and offline"),
    ), patch(
        "src.core.huggingface_utils.run_local_logprob_inference",
        side_effect=ValueError("none of the candidate tokens matched"),
    ):
        result = score_keywords(engine, "img.jpg", local_pipeline=MagicMock())

    assert result.tier is SCORING_TIER.UNAVAILABLE
    assert result.calibrated is False
    assert "Label-confidence scoring failed" in result.notes[0]


def test_orchestrator_successful_tier2_does_not_touch_embedding():
    engine = _engine()

    with patch(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer"
    ) as mock_cls, patch(
        "src.core.huggingface_utils.run_local_logprob_inference",
        return_value={"cat": 0.9, "dog": 0.1},
    ):
        result = score_keywords(engine, "img.jpg", local_pipeline=MagicMock())

    mock_cls.assert_not_called()
    assert result.tier is SCORING_TIER.LABEL_CONFIDENCE


# ---------------------------------------------------------------------------
# TransformersCLIPScorer (lazy load + reuse, transformers mocked)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TransformersCLIPScorer: true cosine from raw embeddings (torch mocked with
# real tensor semantics via tiny numpy-like stubs)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal torch-tensor stand-in supporting the ops the scorer uses.

    2D tensors are stored as ``rows``; 1D results (``sum(dim=-1)``) are
    stored as a flat ``values`` list, mirroring torch's shape change so
    ``tolist()`` returns the same flat structure real torch would.
    """

    def __init__(self, rows=None, *, values=None):
        self.rows = (
            [[float(v) for v in row] for row in rows] if rows is not None else None
        )
        self.values = [float(v) for v in values] if values is not None else None

    @property
    def shape(self):
        if self.rows is not None:
            return (len(self.rows), len(self.rows[0]) if self.rows else 0)
        return (len(self.values),)

    def norm(self, dim=-1, keepdim=True):
        norms = [math.sqrt(sum(v * v for v in row)) for row in self.rows]
        return _FakeTensor([[n] for n in norms])

    def __truediv__(self, other):
        # Broadcast: [n, d] / [n, 1] (or [n, d] / [1, 1]).
        denom = other.rows
        return _FakeTensor(
            [
                [v / denom[min(i, len(denom) - 1)][0] for v in row]
                for i, row in enumerate(self.rows)
            ]
        )

    def __mul__(self, other):
        # Element-wise with [1, d] <-> [n, d] broadcasting in both directions.
        if len(self.rows) == 1 and len(other.rows) > 1:
            self_rows = self.rows * len(other.rows)
            other_rows = other.rows
        elif len(other.rows) == 1 and len(self.rows) > 1:
            self_rows = self.rows
            other_rows = other.rows * len(self.rows)
        else:
            self_rows, other_rows = self.rows, other.rows
        return _FakeTensor(
            [[x * y for x, y in zip(a, b)] for a, b in zip(self_rows, other_rows)]
        )

    def sum(self, dim=-1):
        # Sum along the last dimension -> 1D result (flat values).
        if self.rows is not None:
            return _FakeTensor(values=[sum(row) for row in self.rows])
        return _FakeTensor(values=list(self.values))

    def to(self, device):
        return self

    def tolist(self):
        if self.rows is not None:
            return [row[:] for row in self.rows]
        return list(self.values)


def _install_fake_torch(monkeypatch):
    """Register a torch stub exposing no_grad + is_available flags."""
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: MagicMock()
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: False
    backends.mps = mps
    torch_stub.backends = backends
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def _install_fake_clip(monkeypatch, image_rows, text_rows, calls):
    """Register CLIPModel/CLIPProcessor stubs recording calls."""
    import types

    class FakeModel:
        def __init__(self):
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1
            return self

        def to(self, device):
            return self

        def get_image_features(self, **kwargs):
            calls.append(("image", sorted(kwargs)))
            return _FakeTensor(image_rows)

        def get_text_features(self, **kwargs):
            calls.append(("text", sorted(kwargs)))
            return _FakeTensor(text_rows)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.CLIPModel = type("CLIPModel", (), {"from_pretrained": staticmethod(lambda mid: FakeModel())})
    transformers_stub.CLIPProcessor = type(
        "CLIPProcessor", (), {"from_pretrained": staticmethod(lambda mid: MagicMock())}
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def _reset_clip_caches():
    TransformersCLIPScorer._loaded_models.clear()
    TransformersCLIPScorer._text_features_cache.clear()


def test_clip_scorer_returns_true_cosine_similarities(monkeypatch):
    """Unit-norm image row vs unit-norm text rows -> plain dot products."""
    _install_fake_torch(monkeypatch)
    calls = []
    _install_fake_clip(
        monkeypatch,
        image_rows=[[1.0, 0.0]],  # unit vector along axis 0
        text_rows=[[1.0, 0.0], [0.0, 1.0], [0.7071067811865476, 0.7071067811865476]],
        calls=calls,
    )
    _reset_clip_caches()
    monkeypatch.setattr(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer._resolve_device",
        staticmethod(lambda torch=None: None),
    )

    scorer = TransformersCLIPScorer("openai/clip-vit-base-patch32")
    sims = scorer.cosine_similarities("img.jpg", ["parallel", "perpendicular", "diagonal"])

    assert sims["parallel"] == pytest.approx(1.0, abs=1e-6)
    assert sims["perpendicular"] == pytest.approx(0.0, abs=1e-6)
    assert sims["diagonal"] == pytest.approx(0.7071, abs=1e-3)
    # Both image and text encoders were used.
    kinds = {k for k, _ in calls}
    assert kinds == {"image", "text"}


def test_clip_scorer_normalizes_unnormalized_embeddings(monkeypatch):
    """Cosine must be scale-invariant: an image embedding scaled by 10 must
    yield identical similarities."""
    _install_fake_torch(monkeypatch)
    calls = []
    _install_fake_clip(
        monkeypatch,
        image_rows=[[10.0, 0.0]],
        text_rows=[[1.0, 0.0], [0.0, 1.0]],
        calls=calls,
    )
    _reset_clip_caches()
    monkeypatch.setattr(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer._resolve_device",
        staticmethod(lambda torch=None: None),
    )

    sims = TransformersCLIPScorer("m").cosine_similarities("img.jpg", ["a", "b"])
    assert sims["a"] == pytest.approx(1.0, abs=1e-6)
    assert sims["b"] == pytest.approx(0.0, abs=1e-6)


def test_clip_scorer_caches_text_features_per_candidate_set(monkeypatch):
    """Same candidate list across images: text encoder runs once, image
    encoder runs per image. A changed candidate list re-encodes text."""
    _install_fake_torch(monkeypatch)
    calls = []
    _install_fake_clip(
        monkeypatch,
        image_rows=[[1.0, 0.0]],
        text_rows=[[1.0, 0.0], [0.0, 1.0]],
        calls=calls,
    )
    _reset_clip_caches()
    monkeypatch.setattr(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer._resolve_device",
        staticmethod(lambda torch=None: None),
    )

    scorer = TransformersCLIPScorer("m")
    scorer.cosine_similarities("img1.jpg", ["cat", "dog"])
    scorer.cosine_similarities("img2.jpg", ["cat", "dog"])
    scorer.cosine_similarities("img3.jpg", ["cat", "fox"])

    text_calls = [c for c in calls if c[0] == "text"]
    image_calls = [c for c in calls if c[0] == "image"]
    assert len(image_calls) == 3  # every image encoded
    assert len(text_calls) == 2   # once per distinct candidate set


def test_clip_scorer_model_loaded_once_and_reused(monkeypatch):
    """Multiple scorer instances sharing a model id hit the process-level
    model cache; a different model id loads separately."""
    _install_fake_torch(monkeypatch)
    calls = []
    _install_fake_clip(
        monkeypatch,
        image_rows=[[1.0, 0.0]],
        text_rows=[[1.0, 0.0]],
        calls=calls,
    )
    _reset_clip_caches()
    monkeypatch.setattr(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer._resolve_device",
        staticmethod(lambda torch=None: None),
    )

    s1 = TransformersCLIPScorer("m").cosine_similarities("a.jpg", ["x"])
    s2 = TransformersCLIPScorer("m").cosine_similarities("b.jpg", ["x"])
    _ = TransformersCLIPScorer("other").cosine_similarities("c.jpg", ["x"])

    # One cached model per distinct model id.
    assert len(TransformersCLIPScorer._loaded_models) == 2
    assert s1 == s2 == {"x": pytest.approx(1.0)}
    _reset_clip_caches()


def test_clip_scorer_empty_candidates_skips_model_load(monkeypatch):
    _install_fake_torch(monkeypatch)
    calls = []
    _install_fake_clip(monkeypatch, [[1.0, 0.0]], [[1.0, 0.0]], calls)
    _reset_clip_caches()

    sims = TransformersCLIPScorer("m").cosine_similarities("img.jpg", [])
    assert sims == {}
    assert TransformersCLIPScorer._loaded_models == {}
    _reset_clip_caches()


def test_clip_scorer_gpu_moves_inputs_and_model(monkeypatch):
    """When CUDA resolves, model and inputs are moved via .to(device)."""
    _install_fake_torch(monkeypatch)
    moved = []
    calls = []
    _install_fake_clip(monkeypatch, [[1.0, 0.0]], [[1.0, 0.0]], calls)
    _reset_clip_caches()

    torch_stub = sys.modules["torch"]
    torch_stub.cuda.is_available = lambda: True

    class TrackingProcessor(MagicMock):
        def __call__(self, **kwargs):
            # The scorer moves the processor's OUTPUT tensors to the device,
            # so emit the standard output keys for either an image or a text
            # call, each backed by a tensor whose .to records the move.
            out = {}
            for k in ("pixel_values", "input_ids", "attention_mask"):
                t = _FakeTensor([[1.0, 0.0]])
                t.to = lambda device, _k=k: moved.append(_k) or t
                out[k] = t
            return out

    import transformers as transformers_stub_module
    transformers_stub_module.CLIPProcessor = type(
        "CLIPProcessor", (), {"from_pretrained": staticmethod(lambda mid: TrackingProcessor())}
    )

    monkeypatch.setattr(
        "src.core.keyword_scoring_adapters.TransformersCLIPScorer._resolve_device",
        staticmethod(lambda torch=None: "cuda"),
    )

    TransformersCLIPScorer("m").cosine_similarities("img.jpg", ["x"])
    assert moved  # tensors were sent to the device
    _reset_clip_caches()
