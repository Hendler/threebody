# 100-Item Execution TODO: Novel Math + Strict Science

Legend:
- [ ] pending
- [x] completed

## A. Protocol: strict science before novelty claims (1-20)
- [x] 001. Separate engineering progress from scientific-claim progress in reporting outputs.
- [x] 002. Add explicit evidence tier labels (`engineering_only`, `predictive_signal`, `physics_candidate`).
- [x] 003. Add explicit boolean gate `science_claim_allowed` in aggregate reports.
- [ ] 004. Add preregistered experiment contract JSON artifact for each major run.
- [ ] 005. Require fixed-seed matched before/after for all claim-bearing comparisons.
- [ ] 006. Add automatic same-seed pairing checks that fail noisy comparisons.
- [x] 007. Add input-schema validation for Takens report ingestion.
- [ ] 008. Add strict schema version field checks on all report readers.
- [ ] 009. Add explicit leak-check status field into every report.
- [ ] 010. Add holdout continuity checks for horizon metrics.
- [ ] 011. Add benchmark-suite identity hash to claim artifacts.
- [ ] 012. Add deterministic split manifests persisted with report.
- [ ] 013. Add confidence-interval minimum-width sanity checks.
- [ ] 014. Add minimum-effect-size threshold separate from p-value/CI pass.
- [ ] 015. Add regression-to-the-mean safeguard via repeated reruns.
- [ ] 016. Add power-analysis estimator for seed-count planning.
- [ ] 017. Add novelty-grade tags (`none`, `directional`, `high_bar`) to summaries.
- [ ] 018. Add automatic downgrade rules when benchmark non-regression fails.
- [ ] 019. Add immutable protocol IDs for paper tables.
- [ ] 020. Add CLI command to validate protocol completeness before paper updates.

## B. Deep temporal baselines and architecture quality (21-40)
- [ ] 021. Keep `delta_mlp` as baseline reference model.
- [x] 022. Implement distinct `delta_tcn` model kind (no aliasing).
- [x] 023. Add temporal-convolutional feature map path for `delta_tcn`.
- [x] 024. Route model parsing so `delta_tcn` and `delta_mlp` are distinct candidates.
- [x] 025. Add tests that `delta_tcn` beats persistence on nonlinear residual fixture.
- [x] 026. Add explicit architecture metadata in report (`model_family`, depth, width).
- [x] 027. Add parameter-count reporting for every neural candidate.
- [ ] 028. Add deterministic early-stop criterion based on validation curve.
- [ ] 029. Add learning-curve artifact export (epoch vs val loss).
- [ ] 030. Add gradient-norm telemetry and instability flags.
- [ ] 031. Add regularization sweep for neural models (weight decay + sparsity).
- [ ] 032. Add calibration check (error vs confidence bins) for neural outputs.
- [ ] 033. Add ablation: with/without sensor gates for NN variants.
- [ ] 034. Add ablation: shallow vs deep TCN blocks.
- [ ] 035. Add true causal dilated TCN stack with learnable kernels.
- [ ] 036. Add small transformer encoder baseline for delay windows.
- [ ] 037. Add physics-informed neural loss terms (energy/angular penalties).
- [ ] 038. Add Hamiltonian/symplectic-inspired neural baseline branch.
- [ ] 039. Add deterministic architecture search budget for NN family.
- [ ] 040. Add side-by-side symbolic-vs-neural leaderboard per channel.

## C. Math representation and identifiability (41-60)
- [ ] 041. Add dimensional-consistency checker for discovered symbolic terms.
- [ ] 042. Add unit-normalization diagnostics for feature libraries.
- [ ] 043. Add symbolic equivalence detection to collapse reparameterized duplicates.
- [ ] 044. Add stronger collinearity diagnostics to ranking outputs.
- [ ] 045. Add identifiability score into candidate metadata.
- [ ] 046. Add Jacobian-condition penalties for ill-posed candidate fits.
- [ ] 047. Add sparse group penalties by feature family.
- [ ] 048. Add invariance tests under translation/rotation/permutation.
- [ ] 049. Add residual-target mode for hierarchical decomposition studies.
- [ ] 050. Add orbital-element feature family integration tests.
- [ ] 051. Add encounter-map objective for event-level predictability.
- [ ] 052. Add symbolic return-map baseline and holdout scoring.
- [ ] 053. Add Koopman/EDMD baseline for post-horizon statistical prediction.
- [ ] 054. Add recurrence-quantification feature family for chaos structure.
- [ ] 055. Add symbolic complexity penalties tied to MDL code length.
- [ ] 056. Add uncertainty-aware ranking that penalizes fragile equations.
- [ ] 057. Add explicit Pareto front export (error, complexity, sensitivity).
- [ ] 058. Add structural risk checks for overfitted symbolic forms.
- [ ] 059. Add cross-regime transfer checks for candidate equations.
- [ ] 060. Add candidate provenance graph across generations/runs.

## D. Search loop, memory, and decision quality (61-80)
- [x] 061. Add recursive archive seeding from directories.
- [x] 062. Merge archive statistics across runs with stable deduplication.
- [x] 063. Persist archive-seed provenance metadata.
- [x] 064. Add tests for recursive archive ingestion.
- [x] 065. Add tests for archive merge statistics correctness.
- [ ] 066. Add age-aware decay to stale archive nodes.
- [ ] 067. Add novelty bonus for structurally distinct high-quality nodes.
- [ ] 068. Add archive stratification by regime/channel/horizon.
- [ ] 069. Add seeding policies (`best_only`, `diverse_topk`, `family_balanced`).
- [ ] 070. Add explicit exploitation/exploration budget accounting per iteration.
- [ ] 071. Add MCTS tree-depth telemetry and branching diagnostics.
- [ ] 072. Add Bayesian model averaging baseline for candidate selection.
- [ ] 073. Add Thompson-sampling selector over candidate uncertainty.
- [ ] 074. Add robust reranking under perturbation (stability of rank).
- [ ] 075. Add archive replay mode for long offline discovery loops.
- [ ] 076. Add resume-safe run manifests for multi-day search.
- [ ] 077. Add periodic checkpoint pruning policy with reproducibility hash.
- [ ] 078. Add lock-free parallel candidate evaluation pipeline.
- [ ] 079. Add run scheduler for balanced channel coverage.
- [ ] 080. Add automatic rerun triggers for near-threshold claim cases.

## E. Metrics, horizons, reporting, and manuscript quality (81-100)
- [x] 081. Add multi-horizon holdout metrics to Takens best-model report.
- [x] 082. Add recursive horizon evaluation helper with deterministic logic.
- [x] 083. Add unit tests for horizon metric availability/validity.
- [ ] 084. Add horizon metrics for all channels in `predictability report` aggregation.
- [ ] 085. Add shadowing-time metric and threshold summaries.
- [ ] 086. Add horizon-wise non-regression checks in compare command.
- [ ] 087. Add uncertainty bands for each horizon metric.
- [ ] 088. Add bootstrap over time windows (not only channels).
- [ ] 089. Add report sections that separate one-step vs multi-step claims.
- [ ] 090. Add automated table generator for paper from JSON artifacts.
- [ ] 091. Add auto-figure scripts for horizon curves and CI bars.
- [ ] 092. Add manuscript linter for over-claim language.
- [ ] 093. Add explicit “model-truth vs physical-truth” disclaimer block template.
- [ ] 094. Add cross-audience summary block (physicist + novice) generated from same metrics.
- [ ] 095. Add negative-results registry appendix table.
- [ ] 096. Add reproducibility appendix command transcript generator.
- [ ] 097. Add paper section for epistemic limits and underdetermination criteria.
- [ ] 098. Add “what changed this revision” machine-generated bullet list.
- [ ] 099. Add claim-checklist table mapping each claim to artifact path + test.
- [ ] 100. Add release gate requiring tests + benchmark + paper consistency pass.

## Executed In This Revision
- Distinct `delta_tcn` model path (no alias) with tests.
- Multi-horizon recursive holdout metrics added to Takens reports.
- Stronger Takens report schema validation in efficacy ingestion.
- Evidence-tier and science-claim gating fields added to efficacy aggregate output.
