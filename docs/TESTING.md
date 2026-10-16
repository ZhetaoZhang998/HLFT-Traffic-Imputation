# Verification record

Verified on 16 October 2026 with Python 3.12, NumPy 2.3.5, Numba 0.63.1, llvmlite 0.46.0, and Matplotlib 3.10.8.

- Twelve correctness checks passed, including numerical checks of the distance gradients and full prediction-loss gradient, projection, a descent step, input handling, split boundaries, zero-aware metrics, and a train-only sensor baseline.
- All twelve provided data files were parsed and audited. No coordinate duplicates or cross-split overlaps were found. Every compressed file was decompressed and compared byte-for-byte with the supplied text before packaging.
- The cleaned model ran on all eligible training entries for all four cities. Settings were fixed before examining test results: rank 20, seed 42, learning rate 0.001, initialization scale 0.001, maximum 60 epochs, patience 8, log10 target transform, and the source's zero-exclusion policy.
- Best validation checkpoints were selected at epochs 28, 60, 26, and 3 for Berlin, Guangzhou, New York, and Seattle respectively. Final test RMSE was approximately 7.2689, 5.5396, 11.1701, and 10.8705 in supplied data units.
- Saved checkpoint factors were finite and inside the Poincare ball. Configurations, data hashes, package versions, history, prediction samples, and checkpoints are included under results/verified/.
- The self-contained notebook was syntax-checked and its execution cell completed on a separate small synthetic tensor. Synthetic results are not mixed with the four-city empirical results.
- README links and image paths were checked. The figure-generation script was executed from the included result records, and the figures were visually reviewed.

This verifies the supplied-data workflow and numerical behavior. It is one seeded run under a limited epoch budget, not a replication of all published baseline comparisons or a new state-of-the-art claim.
