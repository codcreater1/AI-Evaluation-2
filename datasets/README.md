# Golden datasets

Two golden sets, loaded via `python -m scripts.load_golden_datasets` into `ata-rag-golden-v1`
and `internship-golden-v1` (names match `examples/ata_rag.yaml` / `examples/internship.yaml`).

## `ata_rag_golden.json` — ATA Assistant (RAG), 104 cases

Built against the live `pomelo-9` ATA Assistant (its backend at `pomelo-8`, `/chat/ask/stream`),
which answers from `akademiata.pl` content. `metadata.source_note` marks how each case was made:

- **`live_verified`** (52 cases) — real question sent to the live assistant; `expected_output.answer`
  is a fact taken from the sources it actually cited (URLs kept as `relevant_doc_ids` /
  `expected_documents`, so retrieval/citation evaluators have real ground truth). Covers tuition
  per programme/campus, admissions, documents, dormitories, scholarships, Erasmus, campuses,
  contacts, plus questions the assistant correctly (or incorrectly — see below) declined to
  answer, in English, Polish and Ukrainian.
- **`paraphrase_of_verified_fact`** (38 cases) — same verified fact, reworded or translated, to
  test robustness to phrasing rather than re-verifying the same fact twice.
- **`constructed_edge_case`** (14 cases) — not fetched live: prompt-injection attempts, privacy
  probes ("what are another student's grades"), typo'd/gibberish/empty input, mixed-scope and
  compound questions, a non-existent programme. `metadata.edge_type` labels each.

Cases where the correct behavior is to decline or say "not found" set
`expected_output.answerable = false` (so `citation_exists` / retrieval evaluators correctly
report "not applicable" instead of failing them) and drop `relevant_doc_ids`.

Two real findings surfaced while collecting this set (kept as cases, flagged in
`metadata.observed_note`): asked "What programmes are offered at the Wrocław campus?" in
English, the assistant answered in Polish; asked "Do you offer PhD programmes?" it refused
scope entirely instead of saying the information isn't in its knowledge base.

## `internship_golden.json` — Internship Coordinator, 54 cases

The live `pomelo-3` reviewer (backend `pomelo-2`) publishes its decision policy at
`/reports/rules` (`university-rules.json`): min. 20 working days, report >= 500 words, employer
score >= 60, originality reject at >= 80% similarity, 3 required attachments. The case generator
re-implements that policy from the published thresholds (reject-severity finding => `not_eligible`,
else any clarify-severity finding => `needs_review`, else `eligible`); every finding code (`DAYS_SHORT`, `REPORT_SHORT`, `REPORT_NOT_ORIGINAL`,
`FUTURE_DATES`, `NAME_MISMATCH`, `EVAL_UNSIGNED`, `EVAL_UNSTAMPED`, `DOCUMENT_MISSING`,
`WEEKEND_DAYS`) comes from finding codes actually observed on the live system, not invented.

- **`live_system_observed`** (26 cases) — real submissions read from `/reports/` and
  `/reports/by-id/{id}` (read-only; nothing was signed or rejected through the UI), converted to
  structured input signals. For every one of these, the policy replica's decision was asserted
  equal to the live system's own `status` before being written out — all 26 matched, which is
  the first data point behind the "LLM judge vs. human/system" comparison in the writeup.
- **`constructed_boundary_from_policy`** (28 cases) — exact-threshold and combined-violation
  cases (e.g. exactly 20 days vs. 19, exactly 60 vs. 59 score, exactly the 80% reject line vs.
  just under it, multiple simultaneous findings) that the live queue's 255 real records did not
  happen to contain, derived directly from the published thresholds rather than guessed.

`expected_output.decision` is one of `eligible` / `not_eligible` / `needs_review`, matching
`examples/internship.yaml`'s `classification.positive_label: eligible`.

## Regenerating

The generator scripts used to build these files from the raw live-fetched records are not
checked in (they read scratch data collected during a browser session); the two JSON files here
are the reviewed, final output. Re-verify a fact by asking the same question at
https://pomelo-9.codewithpeter.com or re-reading a submission at https://pomelo-3.codewithpeter.com.
