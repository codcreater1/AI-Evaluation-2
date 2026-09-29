"""CI helper: `python -m app.cli gate --candidate <id> [--baseline <id>] [--gates gates.yaml]`
Exit code 0 = PASS, 1 = FAIL (block deployment), 2 = error."""
import argparse
import json
import os
import sys

import httpx

from app.experiments.gates import load_gates, render_gate_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ai-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="compare an experiment with its baseline and apply quality gates")
    g.add_argument("--api", default=os.environ.get("EVAL_API_URL", "http://localhost:8000"))
    g.add_argument("--candidate", required=True)
    g.add_argument("--baseline")
    g.add_argument("--gates", help="YAML file with gates (default: gates stored with the experiment)")
    g.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args(argv)

    payload: dict = {"candidate_id": args.candidate, "baseline_id": args.baseline}
    if args.gates:
        payload["gates"] = load_gates(args.gates).model_dump()
    try:
        r = httpx.post(f"{args.api.rstrip('/')}/experiments/compare", json=payload, timeout=60)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cmp = r.json()
    print(json.dumps(cmp, indent=2) if args.json else render_gate_report(cmp))
    gate = cmp.get("gate")
    return 0 if gate is None or gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
