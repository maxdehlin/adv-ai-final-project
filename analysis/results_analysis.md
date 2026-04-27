# ANTIDOTE: Results Analysis

**Date:** 2026-04-27  
**Evaluation:** Direct reward quality — no RL training. Metrics are computed by
scoring 50 held-out expert trajectories and 10–50 held-out poison trajectories
with the learned linear reward r_θ(τ) = θ · mean_φ(τ).

- **Separation** = mean r_θ(expert) − mean r_θ(poison). Positive = reward correctly ranks expert higher.
- **AUC** = P(r_θ(expert) > r_θ(poison)). 1.0 = perfect discrimination, 0.5 = chance, 0.0 = inverted.

---

## 1. poison_random (Random Action Noise)

Training demos contaminated with uniform-random action trajectories.  
Eval held-out: 50 poison_random trajectories (same dir, tail split).

| Dataset | Poison% | Baseline sep/AUC | β_OD sep/AUC | β_PC sep/AUC | β_RC sep/AUC |
|---------|---------|-----------------|--------------|--------------|--------------|
| D0 | 0%  | 10.22 / 1.00 | 10.30 / 1.00 | 10.22 / 1.00 | 2.89 / 1.00 |
| D1 | 10% | 9.20 / 1.00  | 9.46 / 1.00  | 10.22 / 1.00 | 2.35 / 1.00 |
| D2 | 20% | 8.17 / 1.00  | 8.05 / 1.00  | 10.19 / 1.00 | 1.90 / 1.00 |
| D3 | 30% | 7.16 / 1.00  | 6.12 / 1.00  | 10.20 / 1.00 | 1.84 / 1.00 |
| D4 | 40% | 6.09 / 1.00  | 4.86 / 1.00  | 10.15 / 1.00 | 1.49 / 1.00 |
| D5 | 50% | 5.06 / 1.00  | 3.67 / 1.00  | 10.08 / 1.00 | 1.09 / 1.00 |

**Observations:**
- All methods maintain AUC = 1.00 at every concentration. Random noise is sufficiently
  unlike expert behavior that the reward signal is never corrupted enough to flip.
- β_PC is the standout: separation stays flat at ~10.2 across all poison levels. The
  logistic classifier on action summaries cleanly identifies random-action trajectories
  and down-weights them to near zero, effectively shielding the IRL objective.
- Baseline separation degrades gradually (~50% drop from D0 to D5) but never fails.
- β_OD performs slightly below baseline at higher levels — its KNN score penalizes some
  expert trajectories that happen to be outliers in action-summary space.
- β_RC has the lowest absolute separation throughout, but is also fully stable (AUC=1.00).

---

## 2. poison_stop (Intentional Stopping Poison)

Training demos contaminated with trajectories from an agent that always brakes/stops.  
Eval held-out: 10 poison_stop trajectories (same dir, tail split — only 100 files total).  
D5 uses 90 poison demos (not 100) due to the held-out reservation.

| Dataset | Poison% | Baseline sep/AUC | β_OD sep/AUC | β_PC sep/AUC | β_RC sep/AUC |
|---------|---------|-----------------|--------------|--------------|--------------|
| D0 | 0%   | 13.96 / 1.00 | 13.63 / 1.00 | 13.94 / 1.00 | 3.62 / 1.00  |
| D1 | 10%  | 8.19 / 1.00  | 2.64 / 1.00  | 8.12 / 1.00  | 1.55 / 1.00  |
| D2 | 20%  | 2.29 / 0.98  | -6.53 / 0.00 | 2.18 / 0.96  | 3.31 / 1.00  |
| D3 | 30%  | -3.70 / 0.00 | -9.50 / 0.00 | -3.84 / 0.00 | -2.78 / 0.00 |
| D4 | 40%  | -9.50 / 0.00 | -16.46 / 0.00 | -9.66 / 0.00 | -0.85 / 0.00 |
| D5 | ~45% | -13.96 / 0.00 | -21.59 / 0.00 | -14.12 / 0.00 | -1.80 / 0.00 |

**Observations:**
- poison_stop is a significantly harder adversary. All methods lose AUC by D3 (30%), and
  the reward function becomes fully inverted — it learns that stopping is the expert behavior.
- β_RC is the most robust method: it maintains positive separation at D2 (20%) while baseline
  and β_PC are already near zero, and it suffers the least degradation at D3–D5 (separation
  closest to zero rather than deeply negative).
- β_OD is actively harmful on this poison type. At every concentration it performs worse than
  baseline — sometimes dramatically so (D5: -21.6 vs. baseline -14.0). Likely cause: stopping
  trajectories have low action entropy and long action runs, which are also features of cautious
  expert driving. KNN cannot distinguish them, and by mis-labeling some experts as outliers, it
  makes the reward worse.
- β_PC mirrors baseline almost exactly on poison_stop, providing no benefit. The action-summary
  features used by the logistic classifier are insufficient to separate stop poison from experts.
- The D0 baseline separation is higher for poison_stop (13.96) than poison_random (10.22),
  suggesting the reward features are naturally more discriminative against stop behavior than
  random behavior when no poison is present.

---

## 3. Cross-Poison Comparison

| Metric | poison_random | poison_stop |
|--------|--------------|-------------|
| Hardest concentration handled (AUC=1) | 50% (all methods) | 10% (baseline/PC), 20% (RC) |
| Best method overall | β_PC | β_RC |
| Worst method overall | β_OD | β_OD |
| Reward inversion threshold | Never | ~30% |

---

## 4. Key Findings

1. **Poison type matters more than concentration for this feature set.** Random noise is
   trivially handled; intentional stopping poison breaks all methods by 30%.

2. **β_RC is the most robust against structured/intentional poison.** Its EM self-correction
   loop identifies stopping trajectories as inconsistent with the current reward estimate,
   partially compensating. It is the only method to show positive improvement over baseline
   on poison_stop at D2.

3. **β_OD should not be used with structured poison.** It is the worst performer on both
   poison types and actively degrades reward quality. Action-summary KNN is unreliable when
   poison behavior overlaps with expert behavior in feature space.

4. **β_PC is best for random/unstructured noise but blind to intentional poison.** Its
   logistic classifier can cleanly identify random-action trajectories but cannot distinguish
   stopping behavior from expert driving using action frequencies alone.

5. **The linear reward parameterization (8 features) may be too coarse.** The reward learns
   to associate brake/stop actions with low reward from experts, but this signal is overwhelmed
   at ~30% stop-poison contamination, suggesting the feature set lacks sufficient discriminative
   power against structured adversaries.

---

## 5. Open Questions / Next Steps

- **Third poison type (manual input pending):** Will likely fall between random and stop in
  difficulty. The methods' relative ordering may change depending on how structured the behavior is.
- **Mixed poison:** Combining random + stop + manual may reveal whether methods generalize
  across adversarial strategies or specialize to one.
- **Improving β_OD / β_PC for structured poison:** Richer features (visual features from
  CarRacingFeaturesV2, trajectory-level road coverage statistics) may help distinguish stop
  poison from expert behavior.
- **β_RC initialization:** Warm-starting β_RC with β_OD weights (as described in the plan)
  may help at high poison concentrations where the initial uniform-weight IRL is badly corrupted.
