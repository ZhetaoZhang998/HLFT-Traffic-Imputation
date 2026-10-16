# Method and implementation notes

## Research setting

HLFT represents traffic observations as a sparse sensor-by-time-interval-by-day tensor. Each axis has latent factors inside the unit Poincare ball. The prediction combines the three pairwise hyperbolic distances:

$$\widehat{x}_{ijk}=\delta\left[1-\tanh\left(d(s_i,d_j)+d(s_i,t_k)+d(d_j,t_k)\right)\right].$$

This implementation fixes negative curvature at -1, as in the supplied experiment code. It is a NumPy/Numba implementation of the provided method, not a PyTorch extension or a new neural architecture. Gradient calculations are checked against numerical finite differences.

For each observed training entry, the algorithm calculates all three gradients from the pre-update factors, rescales them with the inverse Poincare metric, applies the norm-adaptive learning rate, and projects factors into the ball. Training sweeps over the provided row order unless `--shuffle` is selected.

## What was retained

- The three-factor hyperbolic-distance prediction function.
- Analytic distance gradients and stochastic Riemannian updates.
- Rank 20, base learning rate 0.001, and initialization in [-0.001, 0.001].
- Location-adaptive learning rates proportional to exp(-factor norm).
- The supplied 70/10/20 train/validation/test files.
- The source-code path: log10(speed+1) for training, inverse transform for validation/test predictions.
- The original policy that omits zero-valued and non-finite speed entries. All provided finite speeds are nonnegative. `--keep-zero` is a separate option, not the default experiment.

## Corrections made for reproducibility

1. Paths, dataset choice, seed, learning rate, rank, and epoch budget are configurable. The original script is preserved byte-for-byte in `reference/original_hlft.py`.
2. The supplied script reports and tracks test metrics each epoch and uses them for stopping. Here, only validation RMSE selects the checkpoint and stopping point. The test split is evaluated once at the selected checkpoint.
3. Projection rescales the whole vector radially to norm 1-1e-5. Subtracting epsilon from every coordinate, as in the source projection expression, does not guarantee that vectors with negative coordinates remain inside the ball.
4. Hyperbolic-distance arithmetic is protected near coincident points. At exact coincidence the selected subgradient is zero.
5. Model scale is passed explicitly; no compiled function captures a changing global scale or dataset size. RNG state is explicit.
6. Input parsing handles a final line without a newline. Coordinate duplicates and cross-split overlaps cause a clear error.
7. Axis IDs are mapped to compact indices. The identities from all three splits define the transductive tensor; held-out speed values do not enter factor training or scale estimation. Axis IDs absent from training are recorded.
8. Best-checkpoint factors, not merely the last factors, are saved. MAPE is reported as a percentage; zero targets are excluded from MAPE only and their number is recorded when zeros are retained.
9. JIT warm-up is recorded separately from fitting/validation time. Neither timing is a like-for-like reproduction of the paper's historical timing experiment.

## Interpreting the results

The published article describes normalization. The supplied runnable source instead uses log10(speed+1) and returns metrics in the original dataset units. This repository follows the source-code pathway by default and documents that choice. Consequently, fresh raw-unit RMSE numbers must not be compared numerically with the article's normalized RMSE such as 0.1239. Speed units may differ among cities; no cross-city average error is reported.

These are random-entry imputation splits. This experiment does not demonstrate future forecasting, missing-sensor generalization, or recovery from entire time-block outages. A missingness-pattern study would be a separate extension.

Two simple reference baselines are newly provided: the global training mean and the per-sensor training mean, with a global fallback for an unseen sensor. They are not the five graph baselines from the article. Their predictions use training values only. Any failure to beat a simple baseline remains visible in the results.

The included experiment uses one seed and a limited, documented epoch budget. It is an executable verification run rather than a complete replication of every paper experiment. Test results should not be repeatedly used to tune the epoch budget or other settings. Compare changes using validation, and reserve a new test protocol for substantive follow-up studies.

## Publication

H. Wu, L. Yang, and Z. Zhang, "Latent Factorization of Tensors in Hyperbolic Space for Spatiotemporal Traffic Data Imputation," IEEE/CAA Journal of Automatica Sinica, vol. 13, no. 8, pp. 1991-1993, August 2026. DOI: 10.1109/JAS.2024.124911.

Formal bibliographic authorship is preserved. The repository footer identifies Zhetao Zhang as maintainer; it does not assign sole authorship of the paper or infer individual contributions from the supplied source. The cleaned command-line code, notebook, tests, and presentation were prepared with AI assistance.
