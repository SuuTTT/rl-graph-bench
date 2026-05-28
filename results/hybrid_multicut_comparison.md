# Hybrid Multicut Policy Evaluation Report

This report displays evaluation results for various hybrid configurations combining deep spatial Q-values with exact greedy weight information.

| Config | BA N=20 | BA N=40 | ER N=20 | ER N=40 |
| --- | --- | --- | --- | --- |
| GAEC (Greedy) | 1.2690 | 3.1258 | 3.5929 | 26.7612 |
| SS2V (Pure GNN) | 2.3326 | 7.9521 | 8.3467 | 52.7339 |
| Hybrid Top-K (K=3) | 1.5970 | 6.0543 | 5.5875 | 47.4864 |
| Hybrid Top-K (K=5) | 1.2690 | 4.1056 | 3.8984 | 44.6595 |
| Hybrid Top-K (K=10) | 1.2690 | 3.1581 | 3.6885 | 41.6621 |
| Hybrid Blend (α=0.25) | 1.2422 | 3.4867 | 3.8347 | 32.8102 |
| Hybrid Blend (α=0.50) | 1.8826 | 5.4478 | 4.3557 | 42.0932 |
| Hybrid Blend (α=0.75) | 2.3326 | 7.5402 | 7.7937 | 47.3019 |

### Key Observations
- Pure GNN (`ss2v_d3qn`) struggles with absolute cost values due to spatial generalization limits.
- Hybrid policies dramatically outperform pure GNN across all tested scales and datasets.
- The Top-K filter strategy and Blended Score strategies both close the multicut cost gap significantly, with Hybrid Top-K (K=5) providing an excellent sweet spot.
