# ANTIDOTE: Detailed Implementation Plan

> **Adversarial Noisy Trajectory Inference via Detection Of Trajectories for Trust Estimation**
>
> Status: Planning (no code yet)

---

## 0. Problem Statement & Goals

We modify MaxEnt IRL to be robust against poisoned demonstration datasets. Instead of treating all trajectories equally, we assign a trust weight `w_i ∈ [0, 1]` to each trajectory `τ_i` and maximize:

```
Σ_i  w_i · log P(τ_i | r)
```

We compare three trust estimation methods (β functions) against a baseline (unweighted MaxEnt IRL) across six poisoning levels (0%, 10%, 20%, 30%, 40%, 50%).

---

## 1. Environment & Tooling

**Environment**: OpenAI Gym `CarRacing-v2`
- Observation: 96×96 RGB image (continuous)
- Action: `[steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]`
- Ground-truth reward: score based on tiles visited (used only for final evaluation, NOT during IRL training)

**Downstream RL**: After IRL produces a reward function, we train a standard RL agent (e.g., PPO or SAC) using that reward. Final evaluation score is what we report.

**Why CarRacing?**
- Continuous state/action space makes poisoning realistic and interesting.
- Provides a natural "poison" scenario: an agent that swerves, brakes randomly, or drives off-track is clearly suboptimal.
- Ground-truth reward exists for evaluation.

---

## 2. Data Pipeline

### 2.1 Expert Demonstrations

Collect `N_expert` clean trajectories using one of:
- **Option A (preferred)**: A pre-trained RL agent (e.g., PPO trained to convergence on ground-truth reward). Deterministic or near-deterministic policy gives consistent quality.
- **Option B**: Human demonstrations (noisier but more realistic).

Recommended: `N_total = 200` trajectories total across all datasets. Each trajectory runs for a fixed horizon `T` (e.g., 500 steps or one full lap).

Feature representation per trajectory (for methods that need it):
- Mean speed
- Mean distance from track center
- Total tiles visited
- Variance in steering
- Mean absolute steering angle
- Whether the car left the track (boolean)
- Summary statistics of CNN features extracted from states (optional, harder)

### 2.2 Poisoned Demonstrations

Two classes of poison to consider (matching the proposal's "adversarial and unintentional" framing):

| Type | Description | Motivation |
|------|-------------|------------|
| **Unintentional** | Suboptimal behavior: random actions, excessive swerving, frequent braking | Mimics a novice or distracted demonstrator |
| **Adversarial** | Crafted to steer the reward function in a specific wrong direction | Mimics a malicious actor |

For the initial implementation, use **unintentional poison** (simpler and well-motivated by the car racing example in the proposal). Generate poison trajectories using a random or scripted bad-behavior policy.

Adversarial poison can be added later if time permits (requires solving an optimization problem over the reward function space).

### 2.3 Dataset Construction

```
D0: 200 expert    + 0   poison   (0%)
D1: 180 expert    + 20  poison   (10%)
D2: 160 expert    + 40  poison   (20%)
D3: 140 expert    + 60  poison   (30%)
D4: 120 expert    + 80  poison   (40%)
D5: 100 expert    + 100 poison   (50%)
```

All datasets are fixed before any training begins (no data leakage).

**Important**: the poison trajectories are not labeled during IRL training. The trust estimation methods must infer which are noisy without supervision.

---

## 3. Baseline: Standard MaxEnt IRL

### 3.1 Algorithm Summary (Ziebart et al. 2008)

MaxEnt IRL learns a reward function `r(s)` (or `r(s,a)`) by finding parameters `θ` such that the feature expectations under the induced policy match the empirical feature expectations from the demonstrations:

```
maximize  Σ_i log P(τ_i | θ)
subject to  E_π[f] = E_demo[f]
```

The gradient is:
```
∇_θ L = f_demo - E_π[f]
```

The expected feature counts under policy `π` are computed via soft value iteration (dynamic programming on a discretized or tabular representation).

### 3.2 Practical Considerations for CarRacing

CarRacing has a continuous state space (96×96 images). We need one of:
- **Feature-based IRL**: Hand-craft a feature function `φ(s)` (speed, lane position, steering variance, etc.), learn `r = θ^T φ(s)`. This is simpler and tractable.
- **Deep MaxEnt IRL**: Use a neural network `r_θ(s)` as the reward. Requires soft value iteration approximation. More expressive but harder to train.

**Recommendation**: Start with feature-based IRL using 5–8 hand-crafted features. This is consistent with the original MaxEnt IRL paper and makes the trust weight comparison cleaner.

---

## 4. Trust Estimation Methods (The Three β Functions)

These are the core contribution of ANTIDOTE. Each method assigns `w_i ∈ [0, 1]` to trajectory `τ_i`.

---

### 4.1 Method 1: Outlier Detection (β_OD)

**Idea**: Represent each trajectory as a fixed-length feature vector and use unsupervised anomaly detection to identify outliers. Poison trajectories are expected to be behaviorally different from the majority.

**Steps**:
1. Compute a trajectory summary vector `x_i = φ(τ_i)` using the hand-crafted features above (mean speed, steering variance, track adherence, etc.).
2. Fit an anomaly detector on `{x_i}`:
   - **Isolation Forest** (recommended first choice): assigns an anomaly score in `[-1, 1]`; robust, works in moderate dimensions.
   - Alternative: Local Outlier Factor (LOF), One-Class SVM, or a simple Mahalanobis distance from the empirical mean.
3. Convert anomaly scores to weights:
   ```
   w_i = sigmoid(-λ · anomaly_score_i)
   ```
   where `λ` is a temperature hyperparameter. Normalize so weights are in `[0, 1]`.

**Strengths**: Fully unsupervised; no labeled data needed; fast.

**Weaknesses**: Only detects trajectories that look behaviorally different — sophisticated adversarial poison that mimics expert behavior won't be caught.

---

### 4.2 Method 2: Poison Classifier (β_PC)

**Idea**: Use a small, guaranteed-clean "anchor set" to train a binary classifier distinguishing clean from noisy trajectories. Inspired by the defense from the RLHF poisoning paper (splitting reward model data from SFT data).

**Steps**:
1. Reserve a small clean anchor set `D_anchor` (e.g., 10 trajectories) that are **guaranteed** to be expert quality. These are held out and never mixed with the full dataset.
   - In practice: the first 10 trajectories from the expert collection, before any poison mixing.
2. Generate "negative examples" (synthetic poison) for training the classifier:
   - Use randomly-acting or badly-behaving agents to produce synthetic bad trajectories.
   - These do NOT come from the poisoned datasets D1–D5.
3. Train a binary classifier `C: φ(τ) → {clean, poison}` using `D_anchor` (positive) + synthetic negatives.
4. Assign weights:
   ```
   w_i = P(clean | φ(τ_i))   (predicted probability from C)
   ```

**Key question**: Does the classifier generalize from synthetic poison to the actual poison in the dataset? This is a core empirical question and a potential limitation to discuss.

**Alternative if anchor set is unavailable**: Use the output of Method 1 (Outlier Detection) to pseudo-label a clean subset, then train the classifier iteratively.

---

### 4.3 Method 3: Reward Consistency (β_RC)

**Idea**: Use the learned reward function itself as a self-consistency signal. Trajectories that are inconsistent with the current reward estimate are likely poisoned.

**Steps**:
1. Run standard (unweighted) MaxEnt IRL on the full dataset `D` to get an initial reward estimate `r^(0)`.
   - Optionally: run it only on the top-k trajectories ranked by Method 1 to get a cleaner initialization.
2. Score each trajectory by its log-likelihood under `r^(0)`:
   ```
   score_i = log P(τ_i | r^(0)) = Σ_{t} r^(0)(s_t, a_t) - log Z
   ```
3. Convert scores to weights (softmax-style or rank-based):
   ```
   w_i = sigmoid(λ · (score_i - median(scores)))
   ```
4. Re-run weighted MaxEnt IRL with these weights to get `r^(1)`.
5. Repeat steps 2–4 for `K` iterations (typically K=3–5).

**This is an EM-style algorithm**: Expectation step (estimate weights from reward), Maximization step (re-learn reward from weighted demos).

**Convergence**: Monitor change in `θ` between iterations; stop when `||θ^(k) - θ^(k-1)|| < ε`.

**Strengths**: Self-correcting; can potentially handle even sophisticated adversarial poison.

**Weaknesses**: May converge to a bad local minimum if initial reward estimate is too corrupted (high poison levels). The 50% poison case is the critical stress test.

---

## 5. Weighted MaxEnt IRL: Integration

The modified objective is:

```
L(θ) = Σ_i  w_i · log P(τ_i | θ)
      = Σ_i  w_i · [Σ_t r_θ(s_t, a_t) - log Z(θ)]
```

Gradient:
```
∇_θ L = Σ_i w_i · f_i  -  (Σ_i w_i) · E_π[f]
```

Where `f_i` is the feature count vector for trajectory `τ_i`. This is a minimal change to the standard MaxEnt IRL gradient: the empirical feature expectation becomes a **weighted average** instead of a uniform average.

**Implementation note**: Weights `w_i` are recomputed before gradient updates:
- For β_OD and β_PC: weights are computed once before training.
- For β_RC: weights are recomputed each outer iteration (EM loop).

---

## 6. Experimental Design (Detailed)

### 6.1 Training Protocol

For each dataset D0–D5, and for each method (Baseline, β_OD, β_PC, β_RC):
1. Compute trajectory weights (or use uniform weights for baseline).
2. Run weighted MaxEnt IRL for `M` gradient steps (e.g., M=1000) with a fixed learning rate.
3. Extract the learned reward function `r_θ`.

### 6.2 RL Training with Learned Reward

- Use PPO (stable-baselines3) trained with `r_θ` as the reward signal for `E` environment steps (e.g., 500k steps).
- No access to ground-truth reward during RL training.

### 6.3 Evaluation

- Run the trained RL policy for 5 evaluation episodes using the **ground-truth** CarRacing reward.
- Report mean score across 5 episodes.
- Repeat the full pipeline with 3 random seeds; report mean ± std.

### 6.4 Metrics

| Metric | Description |
|--------|-------------|
| **Score@0%** | Performance with no poison (sanity check) |
| **Score@D_k** | Mean evaluation score at each poison level |
| **Degradation Curve** | Score vs. poison % graph (the main result) |
| **AUC** | Area under the degradation curve (single summary number) |
| **Reward Correlation** | Spearman rank correlation between `r_θ` and ground-truth reward on held-out states (optional diagnostic) |

### 6.5 Hyperparameters to Fix Before Running

| Parameter | Value | Notes |
|-----------|-------|-------|
| `N_total` | 200 | Total trajectories per dataset |
| Trajectory horizon `T` | 500 steps | One episode |
| IRL learning rate | 0.01 | To be tuned on D0 |
| IRL iterations `M` | 1000 | |
| EM iterations `K` (β_RC) | 5 | |
| Anomaly score temperature `λ` | 1.0 | To be tuned |
| RL training steps `E` | 500k | |
| Eval episodes | 5 | |
| Random seeds | 3 | |

---

## 7. Expected Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Continuous state space makes tabular MaxEnt IRL intractable | Use feature-based linear reward; discretize or sample state space for partition function |
| Partition function Z(θ) is intractable for large state spaces | Approximate via importance sampling or restrict to visited states |
| Poison at 50% may corrupt β_RC initialization | Warm-start β_RC with β_OD weights |
| Anchor set assumption for β_PC is unrealistic | Discuss as a limitation; test sensitivity to anchor set size |
| CarRacing is hard to get demonstrations for | Use pre-trained PPO agent; document the agent's score |

---

## 8. Open Questions to Resolve (Iteration Points)

1. **Poison definition**: Should poison trajectories be purely random, scripted (e.g., always swerve left), or adversarially crafted? The answer affects how hard each β method needs to work.

2. **Feature set**: What 5–8 features best capture the distinction between expert and poisoned behavior in CarRacing? Candidates: mean speed, steering variance, off-track percentage, tiles per step, brake frequency.

3. **Partition function approximation**: Which approximation is best for continuous CarRacing? Options: (a) restrict to visited state-action pairs, (b) discretize state space, (c) use the "neural" approach of Deep MaxEnt IRL.

4. **β_PC anchor set size**: How many guaranteed-clean trajectories are assumed available? Is this a realistic assumption for the problem framing?

5. **Combining methods**: Should we report each β independently, or also test a combined estimator (e.g., geometric mean of β_OD and β_RC)?

6. **Adversarial poison**: Do we include adversarially crafted poison (harder) or only unintentional/random poison for this study?

---

## 9. Division of Work (Suggested)

| Component | Owner |
|-----------|-------|
| Expert data collection + poisoning pipeline | TBD |
| Baseline MaxEnt IRL implementation | TBD |
| β_OD: Outlier Detection | TBD |
| β_PC: Poison Classifier | TBD |
| β_RC: Reward Consistency | TBD |
| RL training + evaluation harness | TBD |
| Visualization + paper writing | TBD |

---

## 10. Deliverables Checklist

- [ ] Fixed expert/poison data collection script
- [ ] Trajectory feature extractor
- [ ] Baseline MaxEnt IRL (unweighted)
- [ ] β_OD implementation + integration
- [ ] β_PC implementation + integration
- [ ] β_RC implementation + integration (EM loop)
- [ ] RL training pipeline (PPO on learned reward)
- [ ] Evaluation script (ground-truth scoring)
- [ ] Result graphs (degradation curves, 4 lines × 6 points)
- [ ] Final report
