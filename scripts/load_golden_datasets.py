"""Loads the golden datasets in datasets/*.json into the platform as immutable version 1
(names match examples/ata_rag.yaml and examples/internship.yaml, so those configs run as-is).

Usage:
  uvicorn app.api.main:app &          # or: docker compose up -d
  python -m scripts.load_golden_datasets --api http://localhost:8000
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

TARGETS = [
    ("ata_rag_golden.json", "ata-rag-golden", "ata-rag",
     "ATA Assistant (RAG) golden set: real questions grounded in the live pomelo-9 assistant "
     "and akademiata.pl, plus paraphrase/multilingual variants and constructed robustness cases."),
    ("internship_golden.json", "internship-golden", "internship-coordinator",
     "Internship Coordinator golden set: real submissions observed on the live pomelo-3/pomelo-2 "
     "system plus boundary cases derived from its published university-rules.json policy."),
]


def load(api: str, filename: str, name: str, system: str, description: str) -> None:
    cases = json.loads((DATASETS_DIR / filename).read_text(encoding="utf-8"))
    r = httpx.post(f"{api.rstrip('/')}/datasets", json={
        "name": name, "system": system, "description": description,
        "cases": cases, "note": "initial golden version",
    }, timeout=60)
    if r.status_code == 409:
        print(f"'{name}' already exists, skipping ({r.json().get('detail')})")
        return
    r.raise_for_status()
    v = r.json()
    print(f"created {v['ref']} with {v['n_cases']} cases (hash {v['content_hash'][:12]})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args(argv)
    for filename, name, system, description in TARGETS:
        try:
            load(args.api, filename, name, system, description)
        except httpx.HTTPError as exc:
            print(f"error loading {name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
