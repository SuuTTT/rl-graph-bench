# Reproduction Rules — rl-graph-bench

> **Phase intent**: We are *reproducing* published RL algorithms on our benchmark suite.
> We are **not** inventing new methods, and we are **not** setting custom performance targets.

---

## Rule 1 — Targets must come directly from the source paper

A valid target is:
- An exact numerical result (e.g. NCut=0.33)
- On an exact dataset the paper used (e.g. Cora)
- At an exact k or split (e.g. k=4)
- Cited with table/figure number

**Never** back-calculate a target from `baseline × factor` and treat it as a paper result.
If the paper does not report a result on our benchmark graphs, there is no paper target for those graphs.

### Example of a VIOLATION (do not repeat)
> "NeuroCUT paper target on mini5 = 0.333 (= Spectral 0.406 × 0.82)"

This is **not** in the NeuroCUT paper. The paper reports NCut=0.33 on **Cora k=4**.
mini5 is our synthetic suite; it has no paper target.

### Valid targets from source papers

| Algo | Paper dataset | k | Metric | Paper value | Source |
|------|--------------|---|--------|------------|--------|
| NeuroCUT | Cora | 4 | NCut ↓ | 0.33 | KDD 2024, Table 3 |
| NeuroCUT | CiteSeer | 4 | NCut ↓ | 0.20 | KDD 2024, Table 3 |
| NeuroCUT | Harbin | 4 | NCut ↓ | 0.07 | KDD 2024, Table 3 |
| WRT | City Traffic | 4, n=100 | NCut ↓ | 0.060 | arXiv 2505.13986 |
| CLARE | SNAP Amazon | — | F1 ↑ | 0.773 | KDD 2022 |
| SLRL | SNAP Amazon | — | F-score ↑ | 0.878 | AAAI 2025 |
| AC2CD | BlogCatalog3 | — | NMI ↑ | 0.75 | KBS 2023 |

---

## Rule 2 — Evaluate on the paper's dataset, not a proxy

If the paper target is on Cora, run on Cora.  
mini5 (synthetic SBM graphs) is a **development/smoke-test suite** — useful for fast
iteration, not for claiming paper reproduction.

Do not write "we beat NeuroCUT" based on mini5 results. That claim requires Cora evaluation.

---

## Rule 3 — In-house benchmark datasets are first-class, but assist-only

Synthetic or curated suites (e.g. mini5, fixed17) are **legitimate benchmark datasets in
this repo**. Label them clearly as such (e.g. "mini5 SBM", "fixed17 synthetic") — never
as a paper dataset name.

Their role in the workflow:

| Role | Allowed? | Notes |
|------|----------|-------|
| Smoke-test that training runs without bugs | ✅ | |
| Hyperparameter search / A/B comparison | ✅ | Compare runs against each other, not against paper values |
| Auxiliary eval reported alongside paper-dataset results | ✅ | Label column "mini5" clearly |
| **Primary iteration target** (the number you optimise toward) | ❌ | Use paper datasets for that |
| Claiming paper reproduction | ❌ | Requires paper's own dataset |

**Key constraint**: the improvement-iteration loop (train → eval → decide next run) must be
driven by results on the **paper dataset**, not by progress on an in-house suite.
In-house metrics are assistive signal only — they may confirm a direction but must not be
the sole justification for continuing a training campaign.

---

## Rule 4 — Baselines must match the paper's baselines

When comparing to a paper result, also compare to the paper's listed baselines (e.g. Spectral,
GAP, DMon). Do not substitute a different baseline and claim equivalence.

---

## Rule 5 — "Beat the paper" means strictly less than on the same setup

A result **reproduces** a paper if:
- Same dataset, same split, same k
- Same metric definition (NCut normalisation, NMI variant, etc.)
- Equal or better value

A result merely **beats our synthetic baseline** and is reported as such.

---

## Rule 6 — Follow the reproduction decision tree

Every algo must move through this tree in order. Do not skip steps.

```
[1] Evaluate on paper dataset + paper metric
          |
          v
    result >= target?
    /            \
  YES             NO
   |               |
  [✅ Done]    gap <= 15%?
               /        \
             YES          NO
              |            |
        [continue      gap > 30% after
         training]      reasonable effort?
                         /            \
                       YES             NO
                        |               |
               [run original       [keep tuning]
                authors' code]
                        |
                 original also fails?
                 /               \
               YES                NO
                |                  |
          [flag result as     [our impl has
           over-reported]      a bug; debug]
```

**"Reasonable effort"** for each algo:
- NeuroCUT: ≥ 5000 training episodes on the paper dataset, ≥ 3 random seeds
- CLARE / SLRL: ≥ 3000 episodes, ≥ 3 seeds
- WRT: ≥ 10k episodes (ring/wedge env is sparse reward)

---

## What in-house benchmark results ARE useful for

- Smoke-testing that the algorithm trains without bugs
- Comparing hyperparameter choices **relative to each other** (not to paper numbers)
- Establishing a local performance trend before running expensive paper-dataset evaluations
- Checking that reward signals are positive and training is converging

mini5 NeuroCUT currently: **NCut=0.3534** (Phase-5 ppo_150, −12.9% vs Spectral).
This is an **assist metric** tracked in the changelog. The next iteration target is
NeuroCUT on **Cora k=4** (paper target NCut ≤ 0.33).

---

## Checklist before logging a "paper target met" result

- [ ] Evaluated on the exact dataset named in the paper
- [ ] k / split matches the paper's table
- [ ] Metric definition verified (same NCut normalisation, etc.)
- [ ] Result strictly ≤ (or ≥) the paper value, not just "close"
- [ ] Compared against paper's own baselines
- [ ] No derived target used
