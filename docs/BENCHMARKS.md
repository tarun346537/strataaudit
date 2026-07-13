# Benchmark protocol

## Results and run outputs

`benchmarks/reference.json` records experiment figures and definitions.
`baseline.json` contains the measured CPU fixture baseline used by `gate.json`.
Training and evaluation commands write new measurements under the configured
output directory.

Keep source release/checksum, world splits, model/checkpoint IDs, training
manifest, seed, hypothesis/pair counts, correctness tolerance, temperature fit,
budget and call weights with each run. Changing these changes the experiment.

## Fixed CPU reference

```bash
make lint typecheck test run benchmark
```

Alternatively run `python -m strataaudit.regression_gate --output outputs/regression-gate.json`.
The fixed workload uses seed 7, 180 independent planar worlds, four hypotheses
per world and a 25% case-verification budget: 432 training, 144 calibration
and 144 test hypotheses, grouped by world.

A completed Windows/Python 3.12 CPU run yielded these selected fields:

```json
{
  "cases": 144,
  "temperature": 0.49658530379140964,
  "ece_uncalibrated": 0.10208217899927075,
  "ece_calibrated": 0.04886279532654364,
  "online_verifier_calls": 36,
  "online_verification_fraction": 0.25
}
```

On this numerical workload, direct judging falsely accepted 9 of 82 incorrect
hypotheses, versus zero after budgeted verification. Record the corresponding
workload and hosts when evaluating served models or field observations.

## File-based PyTorch calibration

```bash
python scripts/generate_sample_data.py --worlds 30 --output-dir outputs/fixtures
python -m strataaudit.evaluate --boreholes outputs/fixtures/boreholes.csv --hypotheses outputs/fixtures/hypotheses.jsonl --calibration-backend torch --budget 0.23 --output outputs/evaluation.json
```

This smaller example has 24 test hypotheses. A 23% limit permits five complete
verification calls. For source data, replace the paths and retain the declared
splits. Reports contain ECE, explicit acceptance denominators, decisions, and
online versus evaluator-only verification counts.

## GemPy integration

```bash
python -m pip install -e ".[gempy]" -c constraints/research.txt
python -m strataaudit.gempy_worlds --worlds 15 --output-dir outputs/gempy
python -m strataaudit.evaluate --boreholes outputs/gempy/boreholes.csv --hypotheses outputs/gempy/hypotheses.jsonl --calibration-backend torch --output outputs/gempy-evaluation.json
```

GemPy 2025.2.0 sampling completed for 15 worlds and 60 hypotheses, with
12 hypotheses in three test worlds. Each world has five visible sites and
sixteen withheld sites at different coordinates. Sampling used 301 depth
positions over 25 metres. Provenance records its discretization and zero
coordinate overlap.

| GemPy CPU evaluation | Measurement |
|---|---|
| Held-out ECE before temperature scaling | 0.0629177 |
| Held-out ECE after temperature scaling | 0.0878714 |
| Direct false acceptances / false hypotheses | 0 / 5 |
| Budgeted verification calls | 3 / 12 |

Temperature fitting minimized calibration NLL. On these three test worlds,
held-out ECE increased from 0.0629177 to 0.0878714. Retain the world split and
workload settings when comparing calibration metrics.

## Challenger and equal-budget auditing

Use the README's served-model/training commands with the chosen data and hosts.
Preference collection records accepted-pair, tie and failure counts. Training
writes `training_manifest.json`, checkpoints, adapter/tokenizer and
`metrics.jsonl`. Test worlds are excluded from preferences and checked against
the saved adapter manifest before evaluation. Exported weights, configuration
and tokenizer files are hashed in that manifest. Incomplete or mismatched exports
are rejected. A remote base-model revision is resolved once and retained in the
PEFT configuration so inference loads the same revision.

```bash
python -m strataaudit.orchestrate --boreholes outputs/fixtures/boreholes.csv --hypotheses outputs/fixtures/hypotheses.jsonl --judge-model Qwen/Qwen2.5-3B-Instruct --challenger-adapter outputs/challenger-lora --device cuda --budget-units 100 --output outputs/orchestrated-audit.json
```

Reports retain both strategies' predictive accuracy, withheld RMSE, decision
accuracy, false acceptance, actual spent units and call counts. Calibration
requests occur outside the test budget. Compare equal limits and weights, and
retain actual spending as well as the common cap. Fresh judge clients isolate
circuit state between strategies, and `actual_spending_matched` distinguishes
equal spending from a common ceiling. File checksums, world splits, tolerance,
model IDs and cheap-judge weights accompany the orchestrated report.

For training integration with a tiny locally initialized model:

```bash
python scripts/smoke_dpo.py
```

The completed check trained a tiny initialized GPT-2 for one CPU DPO step,
resumed checkpoint 1 to optimizer step 2, exported a LoRA adapter and loaded it
through inference. Its JSON metrics cover the tiny-model training, checkpoint,
export and adapter-loading paths.

## Gates and test scope

The suite passed 65 tests locally with 73% combined line/branch coverage.
Coverage included 87% of numerical auditing, 89% of data handling, 88% of transport,
87% of staged auditing, 88% of orchestration, 83% of settings and 95% of logging. Optional model-host
and some CLI paths contribute to the uncovered total.
Three remote-revision tests skip when the optional training dependencies are
absent. The nightly CPU job installs the `dev` and `calibration` extras,
including Transformers for callback and model-revision checks.

Hand-defined probability vectors check ECE calculations of 0.19 and 0.06.
Known decisions check false-acceptance denominators. These are definition
checks. Controlled HTTP responses exercise actual clients, cost accounting,
retry caps, circuit recovery and exclusion of hidden observations. Other tests
inject independent ECE, false-acceptance and verification-budget regressions.

The workflow is scheduled at 07:17 UTC on Ubuntu 24.04/Python 3.12. It installs
CPU PyTorch, runs lint/type/tests and enforces the fixed benchmark. Missing or
nonfinite metrics, quality drift, overspending, changed case counts and excess
runtime fail the gate. Reports upload even on failure. YAML and every Makefile
target were validated locally.
Missing/nonfinite baseline values also fail drift checks rather than disabling
them. The nightly workload is the fixed numerical audit. Served-model and full
DPO/GemPy/source-data runs use their selected data and hosts.

The numerical gate has a 120-second runtime ceiling. The job separately allows
25 minutes for setup, checks and reports. Measure model throughput in the
corresponding training or serving run.
Per-world hypotheses share observations, so use worlds as the independent
sampling unit for uncertainty estimates. Repeat training seeds when estimating variability.
