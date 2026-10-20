# Task Memory: icecube-neutrinos-in-deep-ice

## Task profile
- Type: Regression (predicting neutrino direction as azimuth/zenith angles from detector pulse sequences)
- Dataset: ~118.7M training events across 594 parquet files (~95 GB); variable-length pulse sequences (median 63 pulses, 95th pct ~221, max ~178K); 6 features per pulse (x, y, z, time, charge, auxiliary flag); 13.2M test events
- Metric: Mean Angular Error (radians), lower is better
- Medal thresholds: Bronze ≤1.01857, Silver < ~1.001, Gold < ~0.995 (approx)

## Best result
- Score: 1.00019 | Medal: Silver | Run: 20260306_171737
- Approach: Transformer encoder (6 layers, 192 dim, 6 heads, Flash Attention) with sinusoidal Fourier encoding of 6 raw pulse features at 16 frequency bands. vMF loss + auxiliary azimuth/zenith classification heads. Sequential IterableDataset I/O (20× throughput gain) was the critical enabler. Progressive tuning across 3 checkpoint resumptions (1.033→1.017→1.015).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260213_142241 | 1.0491 | none | First real run; Fourier + Transformer + vMF; only 25% data used |
| 20260215_105134 | ~1.097 | none | Vectorized pipeline (40× speedup) but cosine schedule exhausted LR too early |
| 20260215_183520 | ~1.097 | none | Same run, confirmed plateau from LR misconfiguration |
| 20260217_114353 | 1.1016 | none | Polars I/O fix; two-phase loss (vMF→angular); training incomplete |
| 20260221_192216 | 1.0273 | none | Scaled model 1.45M→8M params; fixed cosine LR bug; multi-task heads |
| 20260306_171737 | 1.00019 | silver | Sequential IterableDataset I/O; Flash Attention; progressive tuning |
| 20260308_120346 | 1.0354 | none | Batch-file-native pipeline (3,240 evt/s); cosine+angular loss replaced vMF |
| 20260325_123812 | 1.276 | none | Only 5/72 agent calls used; no tuning runs |
| 20260328_114511 | 1.0473 | none | 1D CNN beat Transformer (convergence speed); vMF instability |
| 20260330_134034 | 1.0287 | none | Multi-task Transformer; BatchGroupSampler; data diversity bottleneck |
| 20260419_113648 | 1.5346 | none | DynEdge GNN; catastrophic I/O thrashing (42 steps in 3.8h) |
| 20260420_092404 | 1.0164 | bronze | Stabilized vMF (softplus+clamp kappa); AttentionPool; d=384, 8 layers |

## Proven strategies
- **Transformer encoder dominates**: All top competition solutions and best runs used Transformers; GNNs/CNNs consistently underperform given time budgets
- **Fourier/sinusoidal feature encoding**: Log-spaced sin/cos pairs (16 frequencies per feature) consistently outperform raw normalization; negligible compute cost, large gradient quality improvement
- **Sequential batch-file I/O is mandatory**: Random per-event parquet access yields ~50-100 evt/s; sequential IterableDataset/BatchGroupSampler yields 1,000-3,240+ evt/s (10-20× gain). Single biggest lever across all runs
- **vMF loss + auxiliary classification heads**: Multi-task learning (direction regression + binned azimuth/zenith classification) consistently improves convergence over single-head regression
- **Priority pulse sampling**: Non-auxiliary pulses first, then by charge descending, truncated to 128-256. Covers ~87-98% of events untruncated; minimal information loss
- **Numerically stable vMF**: softplus(log_kappa) + clamp + log-sinh normalizer eliminates NaN divergence. Raw exp() or unbounded kappa is a recurring failure mode
- **AttentionPool > mean/max pooling**: Learnable query cross-attention for aggregating variable-length sequences consistently outperforms naive pooling
- **Model scaling matters**: 1.45M→8M params closed 0.14 rad gap (run 20260221); d_model=384, 8 layers justified when I/O is solved
- **Flash Attention on A100**: Enables longer sequences and larger batches; prerequisite for competitive runs
- **float32 loss computation**: Even with fp16 training, loss must be computed in float32 to avoid silent NaN corruption

## Failure patterns
- **I/O thrashing**: Random-access parquet loading is the #1 run killer; caused timeouts in runs 20260419 (42 steps in 3.8h), 20260217, 20260215, 20260308
- **vMF kappa saturation/NaN**: Raw exp() kappa or unbounded kappa diverges; occurred in runs 20260213, 20260308 (kappa→500, loss very negative, MAE stuck at 1.53), 20260420 (required reviser fix)
- **Cosine LR schedule misconfiguration**: Underestimating total_steps causes LR to bottom out early; confirmed in runs 20260215 (87% of training at min LR) and 20260221 (premature stagnation)
- **Coder timeout (1200s)**: Candidates consistently hit wall-clock limits during sanity checks or test inference on 13.2M events; batch-file indexing required
- **GNN/EdgeConv overhead**: Dynamic k-NN graph construction (O(n²)) is too slow per step; DynEdge candidates plateau or timeout in every run attempted
- **Test inference bottleneck**: 13.2M test events with per-event Dataset creation causes timeout; must batch-process parquet files directly for inference
- **Checkpoint LR restoration**: Resuming mid-epoch breaks optimizer state / LR scheduler; safer to start fresh epochs

## Feature engineering playbook
- **Fourier encoding**: sin/cos at 16 log-spaced frequencies (2⁰ to 2⁷ or 1.0 to 100.0) per continuous feature → 33 dims per input feature
- **Time normalization**: Per-event relative time (subtract event start time) + log1p(charge)
- **Pulse truncation**: Sort by auxiliary flag (non-aux first), then by charge descending; keep top 128-256 pulses
- **Attention masking**: Explicit masking of padded positions in attention/pooling; prevents padding bias
- **Azimuth rotation augmentation**: Rotate (x,y) coordinates and target azimuth simultaneously; exploits cylindrical symmetry of IceCube detector
- **Charge-weighted centroid**: Useful as global feature but insufficient alone (MLP baselines plateau at 1.05-1.15)
- **Sequence bucketing**: Group similar-length sequences to minimize padding waste; 2-3× throughput improvement

## Tuning insights
- **LR schedule**: Cosine annealing with accurate total_steps estimation is critical; plateau-aware or manual warmup-only for time-budgeted runs. Overestimate total_steps rather than underestimate
- **Two-phase loss**: vMF for early training (broad direction learning), then angular/cosine for refinement; or vMF throughout with stable kappa
- **Kappa bounds**: Clamp to [0.1, 50]; use softplus + 1e-4 floor, not raw exp()
- **Model sizing**: d_model=192-384, 6-8 layers, 6-8 heads covers the sweet spot; larger models justify if I/O is solved and training time permits
- **Batch size**: Constrained by GPU memory with Flash Attention; larger batches improve gradient stability for direction regression
- **Training coverage**: One full epoch ≈ 30+ hours; most runs see <1% of data. Bronze requires either full-dataset training or ensemble
- **Gradient clipping**: norm=1.0 standard; prevents vMF gradient explosions
- **Pre-norm > post-norm**: With stochastic depth for deep Transformers on this task

## Open questions
- **Ensemble potential**: Transformer + CNN/DynEdge architecturally orthogonal; late-fusion averaging of 3D unit vectors never fully tested but estimated ~0.01 MAE gain
- **Test-time augmentation**: Average predictions over azimuth rotations; reported ~0.005 MAE gain in Kaggle posts, never implemented
- **Full dataset training**: No run trained on >25% of data within time budget; solving I/O + longer training likely closes remaining gap to gold
- **Loss function ablation**: Is auxiliary classification head necessary, or does pure stabilized vMF suffice? Adding classification adds hyperparams
- **Sequence length scaling**: Fixed 128-256 vs adaptive padding; 99th percentile ~1177 pulses—are long events systematically mispredicted?
- **Hybrid GNN+Transformer**: 1st-place Kaggle solution used this; never successfully trained due to torch-geometric unavailability and timeout issues
- **Conformal calibration of kappa**: Post-hoc adjustment without retraining; proposed but never tested
- **HDF5/zarr preprocessing**: Converting 594 parquets to single chunked binary store could eliminate I/O bottleneck entirely; never attempted
