# SCOPE (frozen)

## In scope: contributions K1-K5
| ID | Contribution | Falsifiable claim | Experiment |
| --- | --- | --- | --- |
| K1 | Observer-conditioned amortised solver | One network conditioned on an observer embedding approximates the exact solver on unseen observers/scenes with small optimality gap and >=100x lower latency | Gap, latency, constraint violations vs exact solver |
| K2 | Posterior-robust correction | Worst-case/CVaR ΔE00 over an observer posterior beats point-estimate correction under profile error | Success under profile error, with/without robustness |
| K3 | Lighting- and adaptation-aware confusion prediction | Lit colour (SH-L2), emissive sources and chromatic adaptation predict confusion better than albedo-only, esp. emissive small-field objects | Ablation: albedo-only, SH, SH+CAT16, rendered ground truth |
| K4 | Budgeted allocation + temporal stability | MCKP with learned benefit beats uniform, round-robin, greedy at equal compute; event-driven updates + slew limiting cut flicker | Quality at fixed budget, flicker metric, frame-time percentiles |
| K5 | Engine-agnostic contract, white-box and black-box | Same unchanged core corrects two structurally different environments; white-box vs black-box gap is measurable | Held-out-scene transfer, perception-gap measurement |

Scope rules: core never imports an engine; protocol (schema_version 1.x) is the only coupling; all benchmark numbers come from seeded scripted runs; scoring only on held-out observers and scenes.

## Deferred (out of scope for v1)
- PSO, SA (metaheuristic solvers): exact solver (K1 reference) and learned solver cover the claim.
- NSGA-III (many-objective search): single scalarised/CVaR objective only.
- BO (Bayesian optimisation of hyper-parameters / calibration): fixed seeded configs.
- Unity (or other engine) adapter: Three.js + headless synthetic generator suffice; Unity only proves K5 further.
- Cockpit / ICU domain environments: generic dynamic scenes only.

Anything on the deferred list needs an explicit scope change here before work starts.
