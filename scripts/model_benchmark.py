"""Isolated CPU embedding throughput and peak RSS over actual feed descriptions."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from onecredit.config import Settings
from onecredit.embeddings import Embedder
from onecredit.events import InvalidEvent, parse_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    descriptions = set()
    for raw in Path("data/feed.jsonl").read_bytes().splitlines():
        try:
            event = parse_event(raw)
        except InvalidEvent:
            continue
        if event.eligible:
            descriptions.add(event.nature_of_business)
    settings = Settings()
    started = time.perf_counter()
    model = Embedder(settings)
    initialized = time.perf_counter()
    cpu_started = time.process_time()
    for description in sorted(descriptions):
        vector = model.embed(description)
        assert len(vector) == settings.embedding_dimensions and abs(sum(x * x for x in vector) - 1) < 1e-6
    elapsed = time.perf_counter() - initialized
    cpu_elapsed = time.process_time() - cpu_started
    before = model.calls
    model.embed(sorted(descriptions)[0])
    assert model.calls == before
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    report = {
        "model": settings.embedding_key,
        "provider": "PyTorch CPU" if settings.model.startswith("microsoft/harrier") else "CPUExecutionProvider",
        "dimensions": settings.embedding_dimensions,
        "threads": 2,
        "unique_normalized_feed_descriptions": len(descriptions),
        "initialization_seconds": initialized - started,
        "embedding_wall_seconds": elapsed,
        "embedding_cpu_seconds": cpu_elapsed,
        "descriptions_per_second": len(descriptions) / elapsed,
        "peak_process_rss_mib": rss_bytes / (1024 * 1024),
        "cache_hit_avoided_inference": True,
        "limitations": "Single local process with cached artifacts; RSS excludes database/broker; not cold-download time.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
