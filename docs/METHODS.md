# Audit methods

## Geological task

A hypothesis has `[a1, a2, s1, s2]` parameters and boundary depths
`d1(x)=a1+s1*x`, `d2(x)=a2+s2*x`. Correctness requires ordered boundaries
and withheld-boundary RMSE within the configured tolerance. Sourced/GemPy
inputs use metres after normalization.

The reference sampler creates independent worlds, noisy visible boreholes,
noise-free withheld samples and perturbed candidate hypotheses. GemPy generation
instead interpolates actual surfaces and samples lithology along boreholes.
Transition depths are discretized at the recorded resolution.

## Evidence judge and challenger

The numerical judge is regularized logistic regression over visible residuals
and geometric features. Training normalization is retained for calibration/test
scoring. Its least-squares alternative is a numerical reference, separate from
the trained LLM challenger.

A served judge returns support, confidence, rationale and visible borehole
citations. Decision confidence is converted into probability of support.
Decision confidence must lie in `[0.5,1]`. A smaller value contradicts the chosen
binary label and is rejected before routing.
Schema and citation checks validate output shape and visible-evidence access.
Independent geological evaluation measures the scientific quality of proposals.

The LLM challenger proposes a critique and alternative parameters. RLAIF pairs
come from a distinct model ranking two challenger responses using visible
evidence. TRL DPO optimizes these conversations with LoRA adapters. Independent
geological evaluation measures the resulting prediction quality.

## Temperature and holdouts

Temperature minimizes binary negative log-likelihood on calibration worlds.
PyTorch LBFGS fits log-temperature and exponentiates it to obtain a positive
scalar. The small reference also provides a NumPy grid search. A numerical safeguard retains temperature one if fitting
worsens the calibration objective.

Training, calibration and test regions must be disjoint. Visible and withheld
boreholes must have distinct IDs and coordinates. Coordinates within either
role must also be unique, so relabelled repeats cannot add metric weight.
Exact whole-world observation copies are rejected across splits independently
of IDs and row order. These checks use the one-dimensional transect
representation. Group source records by region/site and use a consistent
coordinate origin before applying transect-level deduplication.
Served calibration checks overlap before
remote requests. Adapter provenance is checked before evaluating proposals.
Scaling preserves logit signs and decisions at probability 0.5. Test ECE is
measured with the frozen temperature and independent labels.

## Metric definitions

| Field | Definition |
|---|---|
| `false_acceptances` | Accepted hypotheses that fail independent correctness checks |
| `false_hypotheses` | All independently incorrect hypotheses in the evaluated set |
| `false_acceptance_rate` | False acceptances / false hypotheses |
| `false_discovery_rate` | False acceptances / accepted hypotheses |
| `accuracy`, `audit_decision_accuracy` | Accept/reject decisions matching correctness |
| `hidden_borehole_predictive_accuracy` | Final proposed models meeting withheld correctness |
| `mean_hidden_boundary_rmse` | Mean per-model RMSE over withheld boundary observations |
| `ece_calibrated` | Test calibration gap after frozen temperature scaling |
| `online_verification_fraction`, `verification_fraction` | Online verifier calls / case count |

ECE uses ten equal-width probability bins weighted by case count. Zero
acceptance/discovery denominators produce JSON `null`. Decision accuracy scores
accept/reject decisions. Predictive accuracy scores the final model parameters.

Both strategy accuracies are retained. Relative gain is
`100*(assisted/direct-1)` when direct accuracy is nonzero, and percentage-point
change is `100*(assisted-direct)`. Retain the experiment's chosen convention
when comparing with the project reference.

## Cost accounting

The numerical reference selects exactly `floor(N*budget_fraction)` cases from
observable uncertainty. The staged pipeline uses weighted external-call units.
Reports include maximum budget, actual spending and each call count.
The two strategies get independent judge clients with the same frozen
temperature. This keeps transient circuit state from crossing between arms.
The report distinguishes the common cap from actual spending through
`actual_spending_matched`.

Offline truth checks happen after decisions and are counted separately.
Challenger RMSE covers the rejected/routed proposal subset. Full-set predictive
accuracy evaluates the final model for every original hypothesis.

The verifier compares proposed boundaries with precomputed withheld
observations, including GemPy exports and normalized drill-core records.
Each routed comparison is recorded as one observation check. Weighted audits
charge the configured verification cost.

## Trace records

Each trace contains `timestamp`, `level`, `trace_id` and case/world IDs.
Integrated traces add probabilities, model/verification events, decisions and
final parameters. The evaluator adds `evaluation_only_*` fields. Caller payloads
cannot overwrite trace metadata.

## References

- [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [GemPy model construction](https://docs.gempy.org/tutorials/b_fundamentals/a01_basics.html)
