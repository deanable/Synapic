"""
Benchmark: Local LFM Tagging Throughput
========================================

Measures wall-clock time for the *real* ``ProcessingManager._run_job``
machinery (item fetch, processing loop, progress callbacks, shutdown) with
the only mock being the local model inference call — a configurable latency
sleep that models a local CPU/edge inference round trip.

Synapic processes images sequentially: local GPU/CPU inference uses
``max_workers=1`` so the shared in-memory model is never thrashed by
concurrent calls. This benchmark therefore reports sequential throughput
(the previous parallel-vs-sequential comparison no longer applies).

For each run it reports:

- best wall-clock time across ``--repeats`` runs
- throughput (items/sec)
- theoretical ideal ``latency`` (one item per inference bound)

It also verifies correctness after every run (all items processed, zero
failures).

Usage:
    python scripts/bench_parallel_tagging.py [--items 16] [--latency-ms 200]
        [--repeats 2]
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, ".")

from src.core import config
from src.core.processing import ProcessingManager

logging.disable(logging.CRITICAL)  # keep the benchmark output clean


# ---------------------------------------------------------------------------
# Mock local model inference (the only mocked part — a latency sleep)
# ---------------------------------------------------------------------------

CANNED_RESULT = [
    {"label": "Nature", "score": 0.99},
    {"label": "Outdoor", "score": 0.95},
    {"label": "Sky", "score": 0.88},
]


class FakeLocalModel:
    """Callable stand-in for a loaded local classification pipeline."""

    task = config.MODEL_TASK_IMAGE_CLASSIFICATION

    def __init__(self, latency_s: float):
        self._latency_s = latency_s

    def __call__(self, image, **kwargs):
        time.sleep(self._latency_s)  # the simulated local inference round trip
        return CANNED_RESULT


# ---------------------------------------------------------------------------
# Session + manager construction
# ---------------------------------------------------------------------------

def _make_session(tmpdir: str):
    session = SimpleNamespace(
        datasource=SimpleNamespace(
            type="local",
            local_path=tmpdir,
            local_recursive=False,
            max_items=0,
        ),
        engine=SimpleNamespace(
            provider="local",
            model_id="benchmark-model",
            task=config.MODEL_TASK_IMAGE_CLASSIFICATION,
            confidence_threshold=50,
            device="cpu",
            system_prompt="",
            probability_mode="llm",
            probability_enabled=False,
            probability_candidates=[],
            probability_threshold=0.0,
        ),
        daminion_client=None,  # local source: no DAM client (read unconditionally)
        processed_items=0,
        failed_items=0,
        total_items=0,
        is_processing=False,
        results=[],
    )

    def reset_stats():
        session.processed_items = 0
        session.failed_items = 0
        session.total_items = 0

    session.reset_stats = reset_stats
    return session


def _make_image_files(tmpdir: str, n: int):
    """Create ``n`` tiny JPEGs so the real image-loading path runs."""
    img = Image.new("RGB", (8, 8), "white")
    for i in range(n):
        img.save(Path(tmpdir) / f"img_{i}.jpg", "JPEG")


def run_tagging(tmpdir: str, n_items: int, latency_s: float) -> float:
    """Run the real sequential pipeline; returns wall-clock time."""
    session = _make_session(tmpdir)
    manager = ProcessingManager(
        session=session,
        log_callback=lambda *a, **kw: None,
        progress_callback=lambda *a, **kw: None,
        auto_paginate=False,
    )

    # Inject the fake local model the way _init_local_model would (its own
    # load path is skipped to keep the benchmark about pipeline throughput).
    def _install_fake_model():
        manager.model = FakeLocalModel(latency_s)

    with patch.object(ProcessingManager, "_init_local_model", _install_fake_model), \
         patch("src.core.image_processing.write_metadata", return_value=True):
        start = time.perf_counter()
        manager._run_job()
        elapsed = time.perf_counter() - start

    assert session.processed_items == n_items, (
        f"processed {session.processed_items}/{n_items}"
    )
    assert session.failed_items == 0, f"{session.failed_items} failures"
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=16, help="images to tag per run")
    parser.add_argument("--latency-ms", type=int, default=200,
                        help="simulated local inference latency in ms")
    parser.add_argument("--repeats", type=int, default=2, help="runs per config")
    args = parser.parse_args()

    latency_s = args.latency_ms / 1000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        _make_image_files(tmpdir, args.items)
        print(f"Scanning {args.items} images, {args.latency_ms} ms simulated inference")
        print(f"{'run':>4} {'elapsed(s)':>10} {'items/s':>9}")
        best = float("inf")
        for run in range(1, args.repeats + 1):
            elapsed = run_tagging(tmpdir, args.items, latency_s)
            best = min(best, elapsed)
            throughput = args.items / elapsed
            print(f"{run:>4} {elapsed:>10.3f} {throughput:>9.2f}")

    ideal = latency_s * args.items
    print(f"\nbest wall-clock: {best:.3f}s | theoretical ideal: {ideal:.3f}s")


if __name__ == "__main__":
    main()