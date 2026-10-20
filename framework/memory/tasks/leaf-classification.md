# Task Memory: leaf-classification

## Task profile
- Type: Multi-class classification (99 leaf species)
- Dataset: 891 samples, 99 classes (~9 samples/class); 192 pre-extracted tabular features (64 margin + 64 shape + 64 texture histograms) + variable-size binary grayscale leaf silhouette images
- Metric: Multi-class log-loss (lower is better)
- Medal thresholds: Bronze ≈ 0.01526, Silver ≈ 0.00791 (estimated from run notes)

## Best result
- Score: 0.01157 | Medal: bronze | Run: 20260308_193041
- Approach: Two-level stacking ensemble (LightGBM, XGBoost, ExtraTrees, Logistic Regression as base learners on 249 engineered features) with L2-regularized LR meta-learner on 396 OOF meta-features. Multi-seed MLP variants (ConfigE achieving 0.00808 OOF) combined via temperature-scaled probability calibration per fold.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260210_010735 | N/A | — | Setup failure (PyTorch symbol error) |
| 20260210_012248 | N/A | — | Setup OK but designer never ran |
| 20260210_013600 | N/A | — | 4 designs drafted, no training executed |
| 20260210_014155 | N/A | — | 4 designs, run stopped after design phase |
| 20260210_055317 | 0.0520 | none | Deep NN on tabular features; removed label smoothing → 0.00185 OOF |
| 20260210_214144 | 0.01321 | bronze | XGBoost+LightGBM+MLP stacking ensemble |
| 20260222_231202 | 0.21277 | none | Regression: XGB/LGBM without calibration, MLP underperformed |
| 20260225_203307 | 0.0777 | none | ExtraTrees+LR stacking; LR calibration key |
| 20260226_181938 | 0.0268 | none | SVM+LR+RF stacking; LR alone hit 0.06 |
| 20260227_053624 | 0.01726 | none | Newton-CG LR+PCA dominated; multi-learner ensemble |
| 20260228_010134 | 0.09342 | none | Multi-seed MLP ensemble (OOF 0.00366 but test 0.09) |
| 20260228_022928 | 0.03159 | none | RBF SVM + isotonic calibration |
| 20260308_193041 | 0.01157 | bronze | Stacking + multi-seed MLP + temperature scaling |
| 20260309_194148 | 0.0267 | none | LR+LDA blend + image-derived geometric features |

## Proven strategies
- **Linear models dominate**: Logistic Regression (Newton-CG/lbfgs, C=10–50, elastic-net) consistently outperforms tree-based and deep learning methods on this 99-class, ~9-sample/class regime
- **Stacking with LR meta-learner**: Two-level stacking (diverse base learners → L2 LR meta-learner on OOF probabilities) is the most reliable architecture; directly optimizes log-loss
- **Multi-seed averaging**: 5–10 random seeds per fold dramatically reduces variance (e.g., OOF 0.0071 vs 0.0323 single-seed); cheaper than complex ensembling
- **Temperature scaling per fold**: Global temperature on averaged logits reduced log-loss ~30% vs raw softmax; superior to Platt/isotonic on tiny folds
- **StandardScaler per feature block**: Mandatory for SVM/LR; shape features span 10⁻⁴ vs margin/texture at 10⁻²
- **5-fold StratifiedKFold (random_state=42)**: Stable, reproducible; min class count of 6 limits effective folds to ~6
- **Per-group feature statistics**: Summary stats (mean, std, max, min, median, skew, kurtosis) over margin/shape/texture groups; 21 engineered features boost diversity without overfitting
- **Probability clipping [1e-7, 1-1e-7]**: Prevents log(0) evaluation errors

## Failure patterns
- **Tree-based models plateau badly**: LightGBM (0.18–2.49), XGBoost (0.32–4.59) consistently fail on 99-class extreme multiclass; poor calibration, memorize noise
- **CNN/image approaches always fail**: Binary grayscale silhouettes ≠ ImageNet RGB; ~9 images/class causes severe overfitting; killed by judge in every run attempted
- **Multimodal fusion infeasible**: Training time (100–300 min) exceeds budgets; complexity not justified by marginal gains over tabular-only
- **Label smoothing is a double-edged sword**: Can tank log-loss if features already encode high confidence (run 20260210_055317: removing it gave 10× improvement); mild values (0.028–0.05) work only with MLPs
- **Post-hoc calibration fragile**: CalibratedClassifierCV with isotonic/sigmoid unreliable on ~9 samples/class; meta-learner calibration preferred
- **SVM Platt scaling catastrophic for multiclass**: One-vs-rest SVM produced LL=2.67 in stacking; avoid for probability estimation
- **Coder timeouts at 1200s**: Complex stacking ensembles (5+ base learners × 10-fold CV) frequently timeout; keep to 2–3 base learners
- **OOF-to-test gap**: Multi-seed MLP achieved 0.00366 OOF but 0.09342 test (run 20260228_010134); overfitting on OOF metric possible

## Feature engineering playbook
- **StandardScaler per feature block** (margin/shape/texture separately): essential baseline
- **Per-group summary statistics**: mean, std, max, min, median, skewness, kurtosis across each 64-dim group → 21 new features
- **Per-group PCA (12 components each)**: 36 additional features capturing principal variation per group → total 249 features
- **Image-derived geometric features**: area_ratio, circularity, solidity, aspect_ratio via OpenCV (7–10 features); marginal gains when combined with tabular
- **What doesn't help**: Polynomial interactions, recursive feature elimination, aggressive PCA on full 192-dim, complex interaction terms — all add noise on this tiny dataset

## Tuning insights
- **LR hyperparameters**: C=10–50, solver=newton-cg or lbfgs, penalty=l2 or elastic-net; moderate regularization critical
- **SVM (when used)**: RBF kernel, C≈986, gamma≈0.0025 via 100-trial Optuna; isotonic calibration > Platt scaling
- **MLP architecture**: Group-specific embedding heads (64→64 BN+ReLU+Dropout per group), shared [512,512] layers; label_smoothing=0.028, mixup α=0.05, SWA starting epoch 200, patience-50 early stopping
- **Temperature scaling**: Grid search T ∈ [0.3, 3.0] step 0.02; apply to ensemble logits, not per-seed
- **Ensemble weighting**: scipy.optimize.minimize on OOF predictions; weighted averaging beats geometric/simple averaging and meta-learner training on this sample size
- **LightGBM (if used as base)**: Requires heavy regularization (min_child_samples high, subsample low, L1/L2); 50-round early stopping essential; alone insufficient
- **1-epoch metrics unreliable**: NN scores (4.79 → 0.0052 after full training) mislead early pruning; allow at least 10–20 epochs before kill decisions

## Open questions
- **ConfigE OOF gap**: Best single MLP config achieved 0.00808 OOF (near silver) but ensemble with other configs dragged to 0.01349; investigate config selection vs weighted ensemble
- **Per-class temperature**: Global T may be suboptimal; per-class or learned T could push below bronze
- **CLR transforms**: Centered log-ratio on compositional feature groups (margin/texture histograms) untested
- **Feature subset ensembling**: margin-only vs shape-only vs texture-only models as diversity lever, not yet tried
- **Why LR dominates so thoroughly**: Are the 192 features near-linearly separable? Feature importance / learning curve analysis needed
- **Class weighting**: Near-balanced (6–10 per class) but focal loss or mild upsampling of 6-sample classes unexplored
