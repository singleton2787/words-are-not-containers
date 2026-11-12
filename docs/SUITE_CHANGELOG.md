# Suite hardening change log

Contributed by Nelson Spence (June 2026). Kept as written, apart from this header.

Concise change log for the extracted `rmg_suite_v093` harness.

## Scope

These changes are code-level hygiene and runtime-hardening only. I did not
change the experimental design, data, thresholds, metrics, prompts, model
choices, or methodology.

## Changes Made

- Added `rmg_suite_v093/scripts/stats.py`, adapted from
  his `ordvec-harness` stats module and extended with
  RMG-safe helpers for deterministic bootstrap mean CIs, empirical p-values,
  paired sign-flip p-values, paired t stats, and BH-FDR.
- Wired active scripts to the shared stats helpers where they previously had
  duplicated bootstrap/permutation/FDR code or zero-capable empirical p-values.
- Fixed stale p-value reporting labels so output text matches the actual null
  direction.
- Fixed `run_suite.py` dry-run/runtime wiring: training now invokes
  `hsg_wiki_train.py --all`, see-residual analysis runs after fingerprint
  producers, and the three idiom runs write to model-specific output
  directories instead of clobbering each other.
- Fixed the train-to-Experiment-A handoff: `hsg_wiki_train.py` now saves the
  trained GPT-2 model under `results/wiki_model`, and `run_suite.py` passes
  that model directory to `hsg_affordance_test.py`.
- Fixed `hsg_see_residual.py` to read fingerprint outputs from the suite-local
  `results/` tree rather than `~/hsg`.
- Fixed `hsg_affordance_test.py` default output handling so it writes under
  suite-local `results/affordance_results` and creates parent directories.
- Fixed a Transformers loader typo in
  `hsg_identical_target_context_test.py` (`torch_dtype`, not `dtype`).
- Fixed OrdVec save-path crashes caused by importing `Path` as `_Path` and then
  calling `Path(...)` after expensive model runs.
- Added missing runtime dependencies to `requirements.txt`: `requests`,
  `matplotlib`, and `scikit-learn`.
- Added `scripts/hf_cache.py` and wired active model-loading scripts to use the
  actual Hugging Face hub cache directory (`HF_HUB_CACHE`, or `$HF_HOME/hub`,
  or `~/.cache/huggingface/hub`) instead of passing `HF_HOME` directly as
  `cache_dir`.
- Used a temporary smoke/cache helper for bounded model-cache checks and
  smoke-start verification without running full experiments. That helper and
  its JSON reports are not included in the cleaned redistribution tarball.
- Hoisted operational/training constants in `hsg_wiki_train.py` to module level
  without changing values.
- Cleaned active harness lint issues: unused imports, placeholder-free
  f-strings, unused local variables, duplicate unused imports, ambiguous
  one-letter locals, bare `except`, and intentional path-bootstrap import order.
- Applied Ruff formatting to `rmg_suite_v093/run_suite.py` and
  `rmg_suite_v093/scripts/`.
- Left `rmg_suite_v093/legacy/original_sources/` unchanged as archival source
  material.

## Verification

- `ruff check rmg_suite_v093/run_suite.py rmg_suite_v093/scripts --select F401,F841,F811,F541,F821,F822,F823,E401,E402,E741 --output-format=concise`
- `ruff format rmg_suite_v093/run_suite.py rmg_suite_v093/scripts`
- `python3 -m compileall -q rmg_suite_v093/run_suite.py rmg_suite_v093/scripts`
- `python3 rmg_suite_v093/run_suite.py --dry-run`
- `.venv/bin/python rmg_suite_v093/smoke_cache_and_start.py --cache-local-only --skip-smoke --report rmg_suite_v093/results/model_cache_local_report.json`
- `.venv/bin/python rmg_suite_v093/smoke_cache_and_start.py --skip-cache --timeout 45`
- Cached model load smoke for Qwen/Qwen2-7B and
  mistralai/Mistral-7B-v0.1 with 8-bit quantization on the RTX 3090.

Static checks passed after the changes. The bounded smoke-start pass reached
15/20 runner steps. Remaining smoke blockers are expected without full upstream
execution or external access: `wiki_affordance_corpus.txt` is produced by the
prior corpus step, `results/wiki_model` is produced by the train step,
fingerprint JSONs are produced by the prior fingerprint steps, and
`google/gemma-7b` is gated for the supplied Hugging Face account.

## Packaging Note

The revised tarball excludes local development artifacts: virtualenvs, Ruff
caches, Python bytecode, smoke JSON reports, temporary smoke tooling, and model
weights/cache payloads. The Hugging Face cache used during smoke testing lives
outside this package and is not part of the tarball.
