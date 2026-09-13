"""
Keyword Scoring Adapters (Local-only)
======================================

Tier adapters and the tier selector for arbitrary-user-keyword scoring,
as specified in ``docs/KEYWORD_SCORING_DESIGN.md``.

Every adapter returns a :class:`~src.core.keyword_scoring.ScoreResult`
so downstream code (thresholding, logging, results export) never
branches on how the scores were produced:

- Tier 2 ``label_confidence``: delegates to the existing local
  ``run_local_logprob_inference`` path (pipeline per-label confidence,
  NOT calibrated).
- Tier 2.5 ``embedding``: true CLIP image-text cosine similarities
  (raw embeddings, L2-normalized) softmaxed over the candidate set;
  semantic and prompt-driven, but candidate-set dependent (NOT calibrated).

Failures never raise out of the adapters: they degrade to a tier-0
``unavailable_result`` with the reason recorded, matching the
"hide on failure" contract of the existing probability pass.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.core.keyword_scoring import (
    SCORING_TIER,
    ScoreResult,
    build_score_result,
    softmax_from_similarities,
    unavailable_result,
)

logger = logging.getLogger(__name__)

# Hugging Face model id used for tier 2.5 (CLIP-style image-text similarity).
# A ~600MB download on first use; loaded lazily and cached.
EMBEDDING_MODEL_ID = "openai/clip-vit-base-patch32"

# CLIP image-text cosine similarities concentrate in a narrow ~[0.2, 0.35]
# band, so raw similarities are a poor absolute signal. We record them as
# notes and softmax with a fixed temperature for the distribution.
EMBEDDING_TEMPERATURE = 0.01


class ImageTextSimilarityScorer(Protocol):
    """Minimal protocol for CLIP-style image-text similarity backends.

    ``cosine_similarities`` returns a mapping of candidate keyword ->
    cosine similarity in [-1, 1] for the supplied image.
    """

    def cosine_similarities(
        self, image_path: str, candidates: List[str]
    ) -> Dict[str, float]: ...


# ---------------------------------------------------------------------------
# Tier selector
# ---------------------------------------------------------------------------


def pick_scoring_tier(engine: Any, mode: Optional[str] = None) -> Optional[SCORING_TIER]:
    """Return the highest available scoring tier for this engine config.

    Synapic supports local inference only.  Tier 2 (``label_confidence``)
    is available when the local model is an image-classification pipeline
    and probability scoring is active.  When the tier selector returns
    ``None`` no scoring pass runs at all (LLM-only mode, no candidates,
    or a non-classification local model).

    Args:
        engine: ``EngineConfig``.
        mode: Optional pre-normalized tagging mode ('llm' | 'probability'
            | 'both') as computed by the pipeline.  When omitted, the mode
            is derived from the engine with the same legacy rules the
            pipeline uses (``probability_mode`` missing/'llm' plus
            ``probability_enabled=True`` maps to 'both').

    Returns ``None`` when scoring cannot run at all (LLM-only mode, no
    candidates, local non-classification pipeline).  ``None`` means "no
    scoring pass", distinct from a tier-0 result which means "scoring was
    attempted and failed".
    """
    candidates = getattr(engine, "probability_candidates", None) or []

    # Synapic tags images with local (LFM) models only: any non-local
    # provider (legacy cloud IDs included) never selects a scoring tier.
    provider = str(getattr(engine, "provider", "") or "").lower()
    if provider != "local":
        return None

    if mode is None:
        mode = str(getattr(engine, "probability_mode", "") or "").lower()
        if mode not in ("llm", "probability", "both"):
            # Legacy configs only persisted probability_enabled.
            mode = (
                "both"
                if getattr(engine, "probability_enabled", False)
                else "llm"
            )
        elif mode == "llm" and getattr(engine, "probability_enabled", False):
            # Legacy sessions enable probabilities without a mode field.
            mode = "both"

    if mode == "llm" or not candidates:
        return None

    task = str(getattr(engine, "task", "") or "")
    if task == "image-classification":
        return SCORING_TIER.LABEL_CONFIDENCE

    # Local VLM could serve tier 2 in principle, but the current local VLM
    # path does not expose per-candidate label confidence and we do not
    # double-load a caption model just for scoring.  Explicitly unavailable
    # rather than pretending a tier will run.
    return None


# Set to True by tests to force the embedding tier to be skipped even when a
# real CLIP model is available on the machine. Production code never touches
# this; the rescue path checks it before attempting a model download.
EMBEDDING_TIER_DISABLED = False


def pick_embedding_tier(engine: Any) -> bool:
    """Whether tier 2.5 (CLIP-style embedding scoring) is appropriate.

    Tier 2.5 applies when arbitrary candidates cannot be scored by the
    local classification model — i.e. the local provider selected tier 2
    but the candidate set does not overlap the model's label space. In
    that situation the alternatives are a failed tier 2 or a slow tier 3
    (no local chat model); an embedding pass scores any candidates.

    The rescue is strictly opt-in (``engine.embedding_rescue_enabled``):
    the CLIP model is a ~600MB download on first use, so it must never be
    fetched unless the user asked for it. The embedding model itself may
    still need downloading at scoring time (see ``TransformersCLIPScorer``);
    a miss degrades to tier 0 per the design's no-silent-degradation rule.
    Engines whose configs predate the flag are treated as opted-out
    (``getattr`` default False), keeping legacy behavior unchanged.
    """
    if not getattr(engine, "embedding_rescue_enabled", False):
        return False
    provider = str(getattr(engine, "provider", "") or "").lower()
    if provider != "local":
        return False
    task = str(getattr(engine, "task", "") or "")
    return task == "image-classification"


# ---------------------------------------------------------------------------
# Tier 2 — local pipeline label confidence (existing path, not calibrated)
# ---------------------------------------------------------------------------


def score_keywords_local_label_confidence(
    pipe: Any,
    image_path: str,
    candidates: List[str],
    device: int = -1,
) -> ScoreResult:
    """Tier 2: per-label pipeline confidence for matched model labels.

    Delegates to :func:`src.core.huggingface_utils.run_local_logprob_inference`
    (which owns pipeline reuse, full top_k, exact/fuzzy matching, and the
    loud all-unmatched ValueError) and packages the result with explicit
    match types so the UI can distinguish "exact label" from "fuzzy string
    match" from "no match".
    """
    from src.core import huggingface_utils

    if not candidates:
        return unavailable_result("No candidate keywords supplied.", candidates)

    score_map = huggingface_utils.run_local_logprob_inference(
        pipe, image_path, candidates, device=device
    )

    # Derive match types from the same logic run_local_logprob_inference
    # uses (normalized exact over the returned labels, then difflib fuzzy)
    # so the typing reflects actual matching precedence, not score values:
    # a fuzzy match to a positively-scored label must not be typed "exact".
    match_types: Dict[str, str] = {}
    fuzzy_hits: List[str] = []
    unmatched: List[str] = []
    try:
        labels = huggingface_utils._get_pipeline_labels(pipe) or list(score_map.keys())
    except Exception:
        labels = list(score_map.keys())
    normalized_label_index = {
        str(lbl).strip().lower() for lbl in labels if lbl and str(lbl).strip()
    }
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized in normalized_label_index:
            match_types[candidate] = "exact"
            continue
        fuzzy_target = None
        try:
            fuzzy_target = huggingface_utils._fuzzy_match_label(normalized, labels)
        except Exception:
            logger.debug("Fuzzy re-derivation failed", exc_info=True)
        if fuzzy_target is not None:
            match_types[candidate] = "fuzzy"
            fuzzy_hits.append(candidate)
        else:
            match_types[candidate] = "none"
            unmatched.append(candidate)

    notes: List[str] = []
    if fuzzy_hits:
        notes.append(
            "Fuzzy-matched candidates score the nearest model label "
            "(string similarity, not the candidate concept): "
            + ", ".join(fuzzy_hits)
        )
    if unmatched:
        notes.append("Unmatched candidates score 0.0: " + ", ".join(unmatched))

    return build_score_result(
        candidates,
        score_map,
        SCORING_TIER.LABEL_CONFIDENCE,
        match_types=match_types,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tier 2.5 — CLIP-style image-text similarity (semantic, NOT calibrated)
# ---------------------------------------------------------------------------


def score_keywords_embedding(
    scorer: ImageTextSimilarityScorer,
    image_path: str,
    candidates: List[str],
    *,
    temperature: float = EMBEDDING_TEMPERATURE,
) -> ScoreResult:
    """Tier 2.5: softmaxed cosine similarities between image and prompts.

    Scores every arbitrary candidate semantically (no label-space
    constraint), but the result is explicitly ``calibrated=False``: the
    distribution is relative to the supplied candidate set (adding a
    nonsense candidate shifts every score) and CLIP cosine similarities
    cluster in a narrow band, so only rank ordering within one run is a
    meaningful signal — see the notes appended to the result.
    """
    if not candidates:
        return unavailable_result("No candidate keywords supplied.", candidates)

    sims = scorer.cosine_similarities(image_path, candidates)
    distribution = softmax_from_similarities(sims, candidates, temperature=temperature)

    # Surface the genuine cosine values alongside the derived distribution:
    # the softmax scores are candidate-set relative, but the raw similarities
    # are real measurements for this image and can be compared across runs.
    raw_line = "; ".join(
        f"{candidate}={sims.get(candidate, float('nan')):.3f}"
        for candidate in candidates
    )
    notes = [
        f"Raw CLIP cosine similarities (image vs prompt): {raw_line}.",
        "Distribution scores are CLIP similarities softmaxed over THIS "
        "candidate set; adding or removing candidates shifts every score "
        "without the image changing (not calibrated).",
        "Only rank ordering within one candidate set is meaningful for the "
        "distribution; the raw cosine values above are the comparable signal "
        "across runs.",
    ]

    return build_score_result(
        candidates,
        distribution,
        SCORING_TIER.EMBEDDING,
        match_types={c: "semantic" for c in candidates},
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Orchestrator: run the ladder for an engine config (local-only)
# ---------------------------------------------------------------------------


def score_keywords(
    engine: Any,
    image_path: str,
    candidates: Optional[List[str]] = None,
    *,
    mode: Optional[str] = None,
    local_pipeline: Any = None,
    model_name: str = "",
) -> ScoreResult:
    """Run the tier ladder (2 → 2.5 → 0) for the engine's candidate set.

    Synapic supports local (LFM) inference only.  The ladder attempts
    tier 2 (label confidence) and falls back to tier 2.5 (CLIP embedding)
    rescue when enabled for the engine and the tier-2 pass fails.

    Args:
        engine: ``EngineConfig`` (mode, candidates, threshold, device).
        image_path: Path to the image being scored.
        candidates: Override for ``engine.probability_candidates``.
        local_pipeline: The already-loaded local pipeline (tier 2). Required
            when the selected tier is ``label_confidence``.
        mode: Optional pre-normalized tagging mode, forwarded to the tier
            selector (see :func:`pick_scoring_tier`).
        model_name: Optional explicit model name (unused for local scoring,
            accepted for API compatibility with callers).

    Returns:
        A ScoreResult from the highest tier that ran, or a tier-0
        ``unavailable_result`` carrying the failure reason.  Never raises
        for provider/parse failures — those are recorded and degraded,
        matching the "hide on failure" pipeline contract.  (Programming
        errors like a bad factory signature may still raise.)
    """
    candidate_list = list(
        candidates
        if candidates is not None
        else (getattr(engine, "probability_candidates", None) or [])
    )
    tier = pick_scoring_tier(engine, mode=mode)
    if tier is None:
        return unavailable_result(
            "Scoring not available for this engine configuration (mode, "
            "provider, or candidate set).",
            candidate_list,
        )

    # Tier 2: local pipeline label confidence.
    if tier == SCORING_TIER.LABEL_CONFIDENCE:
        if local_pipeline is None:
            return unavailable_result(
                "No local pipeline loaded for label-confidence scoring.",
                candidate_list,
            )
        try:
            return score_keywords_local_label_confidence(
                local_pipeline,
                image_path,
                candidate_list,
                device=0
                if str(getattr(engine, "device", "cpu")).lower() == "cuda"
                else -1,
            )
        except Exception as exc:
            # Tier 2 failed (typically: no candidate matches the model's
            # label space). Tier 2.5 (embedding) is the natural rescue: it
            # scores arbitrary candidates semantically with no label-space
            # constraint. Only attempted when enabled for this engine.
            logger.warning(
                "Label-confidence scoring failed (%s: %s).",
                type(exc).__name__,
                exc,
            )
            if pick_embedding_tier(engine):
                embedding_result = _try_embedding_tier(
                    engine, image_path, candidate_list
                )
                if embedding_result is not None:
                    return embedding_result
            return unavailable_result(
                f"Label-confidence scoring failed: {type(exc).__name__}: {exc}",
                candidate_list,
            )

    return unavailable_result(
        f"Unhandled scoring tier: {tier}", candidate_list
    )


def _try_embedding_tier(
    engine: Any, image_path: str, candidate_list: List[str]
) -> Optional[ScoreResult]:
    """Attempt tier 2.5; return None (not a tier-0 result) when the CLIP
    backend is unavailable so the caller can append its own reason."""
    if EMBEDDING_TIER_DISABLED:
        return None
    try:
        scorer = TransformersCLIPScorer(EMBEDDING_MODEL_ID)
    except Exception as exc:
        logger.info(
            "Embedding tier unavailable (%s: %s).", type(exc).__name__, exc
        )
        return None
    try:
        return score_keywords_embedding(
            scorer,
            image_path,
            candidate_list,
            temperature=EMBEDDING_TEMPERATURE,
        )
    except Exception as exc:
        logger.warning(
            "Embedding scoring failed (%s: %s).", type(exc).__name__, exc
        )
        return None


class TransformersCLIPScorer:
    """True CLIP image-text cosine similarity scorer using transformers.

    Computes raw image and text embeddings directly with ``CLIPModel`` /
    ``CLIPProcessor``, L2-normalizes both, and returns genuine cosine
    similarities (dot products of unit vectors) — not the pipeline's
    post-processed scores, which fold in the model's learned logit scale.

    Loading is lazy (construction downloads/loads nothing; the first
    ``cosine_similarities`` call does) and cached: the model is loaded
    once per process, and text embeddings for a candidate set are cached
    until the candidates change, so batch processing neither reloads
    weights nor re-encodes identical prompts per image (same reuse rule
    as the local probability pass).
    """

    # Process-level caches so batch runs never reload weights or re-encode
    # identical candidate prompts. Keyed by model_id / (model_id, candidates).
    _loaded_models: Dict[str, Any] = {}
    _text_features_cache: Dict[Tuple[str, str], Any] = {}

    def __init__(self, model_id: str = EMBEDDING_MODEL_ID):
        self.model_id = model_id

    def _load_model_and_processor(self):
        """Lazily load and cache the CLIP model + processor for this model id."""
        cached = TransformersCLIPScorer._loaded_models.get(self.model_id)
        if cached is not None:
            return cached

        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(self.model_id)
        processor = CLIPProcessor.from_pretrained(self.model_id)
        model.eval()

        device = self._resolve_device(torch)
        if device is not None:
            model.to(device)

        entry = {"model": model, "processor": processor, "device": device}
        TransformersCLIPScorer._loaded_models[self.model_id] = entry
        return entry

    @staticmethod
    def _resolve_device(torch):
        """Prefer CUDA, then Apple MPS, then CPU (mirrors get_device_info)."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return None  # CPU is the default device; no explicit move needed.

    def _text_features(self, cache_entry, candidates: List[str]):
        """Return (unit-normalized) text features for the candidate prompts.

        Cached per (model_id, normalized candidate tuple) at process level so
        repeated scoring runs against the same candidate list skip text
        encoding entirely. Cache entries are cleared implicitly when the
        candidate tuple changes (new cache key), keeping memory bounded to
        the candidate sets actually used.
        """
        import torch

        cache_key = (self.model_id, "|".join(candidates))
        cached = TransformersCLIPScorer._text_features_cache.get(cache_key)
        if cached is not None:
            return cached

        model = cache_entry["model"]
        processor = cache_entry["processor"]
        device = cache_entry["device"]

        with torch.no_grad():
            text_inputs = processor(text=list(candidates), return_tensors="pt", padding=True)
            if device is not None:
                text_inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in text_inputs.items()
                }
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        TransformersCLIPScorer._text_features_cache[cache_key] = text_features
        return text_features

    def cosine_similarities(
        self, image_path: str, candidates: List[str]
    ) -> Dict[str, float]:
        """Return one true cosine similarity per candidate in [-1, 1].

        Both embeddings are L2-normalized before the dot product, so the
        result is the genuine cosine between the image embedding and each
        candidate's text-prompt embedding. Candidates are used verbatim as
        prompts (prompt phrasing shifts CLIP scores materially).
        """
        import torch

        if not candidates:
            return {}

        cache_entry = self._load_model_and_processor()
        model = cache_entry["model"]
        processor = cache_entry["processor"]
        device = cache_entry["device"]

        with torch.no_grad():
            image_inputs = processor(images=image_path, return_tensors="pt")
            if device is not None:
                image_inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in image_inputs.items()
                }
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self._text_features(cache_entry, candidates)

        # Cosine similarity: row-wise dot product of unit vectors.
        # (image_features [1, d] * text_features [n, d]).sum(-1) -> [n];
        # equivalent to image @ text.T and avoids a transpose op.
        cos = (image_features * text_features).sum(dim=-1)
        values = cos.tolist() if hasattr(cos, "tolist") else [float(cos)]
        return {candidate: float(v) for candidate, v in zip(candidates, values)}
