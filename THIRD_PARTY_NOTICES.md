# Third-party notices

The root MIT license applies to original RSI-Audit contributions. It does not
replace licenses for benchmark data, models, dependencies, or adapted code.

## EvalPlus and HumanEval

The type restoration rules in `rsi_audit/data.py` and oracle semantics in
`rsi_audit/scoring.py` adapt the following fixed upstream version:

- Project: EvalPlus, by the EvalPlus contributors.
- Commit: `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2` (v0.3.1).
- Source: https://github.com/evalplus/evalplus/tree/e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2
- License: Apache License 2.0; full upstream text is retained in
  [licenses/EvalPlus-LICENSE.txt](licenses/EvalPlus-LICENSE.txt).
- Files under upstream `evalplus/eval/` also retain the MIT notice for OpenAI
  HumanEval. That notice is preserved in `rsi_audit/scoring.py`.

Modifications: the original evaluator's in-process candidate execution is
replaced by a bounded Docker wire protocol and host-side result comparison.
Data loading adds fixed archive hashes, safe public-example parsing, family
partitioning, private bundles and commitments. The repository documents
differences in timeout, resource and output-type support; it does not claim
byte-for-byte identity with the upstream evaluation program.

Benchmark archives and model weights are downloaded separately from their
original sources and are not redistributed in this repository. Their upstream
terms continue to apply. Qwen2.5-Coder-7B-Instruct's pinned model card and data
source URLs are recorded in `evidence/baseline-lock.json`.
