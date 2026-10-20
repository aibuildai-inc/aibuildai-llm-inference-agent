# Pattern: video-understanding-action-detection

## When this applies
- Temporal segmentation/detection on multi-modal sequences (skeleton, audio, tracking, optional video)
- Small-to-medium labeled corpora (hundreds of sequences) with severe class imbalance or rare-event targets
- Per-frame/per-pair labels with strong group structure (session, game_play) and large val→test gaps
- Metrics that penalize sequence structure: Levenshtein, MCC (not raw accuracy)

## Proven patterns
- **Separate models per sub-problem** beats one unified model: P2P vs P2G in nfl-player-contact-detection (silver 0.7285); detect-then-classify proposer+classifier in multi-modal-gesture-recognition (bronze 0.2421)
- **GBDT (LightGBM/XGBoost) on engineered features > deep nets** on small/tabular-ish data: XGBoost GPU silver 0.7285 vs Bi-LSTM 0.6755, MLP 0.077; wav2vec2-frozen+LightGBM bronze 0.2421 vs MS-TCN 0.561 test
- **Group-aware CV is mandatory**: GroupKFold by game_play (nfl); session-grouped GroupKFold (gesture). Random splits silently leak play/session structure
- **Direct metric-targeted post-processing**: per-class MCC threshold search (+3–5% MCC, nfl); exactly-K=20 DP decoder with min-duration + bigram LM (gesture, non-negotiable)
- **Frozen pretrained embeddings** break plateaus cheaply: wav2vec2 XLS-R unblocked 4+ stuck skeleton+GBDT runs in gesture task

## Common pitfalls
- **End-to-end deep nets timeout or collapse**: 1D-CNN killed (nfl); CTC blew up to 25.65 Levenshtein (gesture run 20260419); HMM EM training exceeds 1200s
- **Video/CNN fusion rarely pays**: killed every NFL run (4–8h preprocessing, alignment noise); multimodal late-fusion BiLSTM (C4) lost to simpler C3+C5 stack in gesture
- **Submission format bugs destroy scores**: ID "300" vs "0300" (gesture); `.cat.codes` differing train/test (nfl)
- **Val→test gap is structural** (~2× in gesture: 0.0657→0.2421); over-trust of val drives wrong selection
- **Stacking with 5-fold OOF on heavy bases** times out; keep base count small
- **Data leak vectors**: test `_data.mat` Labels fields (gesture); always enforce `include_labels=False`

## Model selection guide
- Tabular per-step/per-pair targets → **LightGBM CPU baseline → XGBoost GPU `hist` with monotone constraints**
- Sequence-of-labels with structured output → **detect-then-classify + DP/Viterbi decoder**, never CTC
- Audio modality available → **frozen wav2vec2 XLS-R + GBDT**, not MFCC+GMM-HMM from scratch
- Multi-class temporal → **per-class GMM-HMM + bigram LM + exact-K DP**
- Avoid: MS-TCN, end-to-end Bi-LSTM/GRU, video CNN fusion, stacking >3 heavy bases

## Feature engineering patterns
- **Distance/proximity dominates**: Euclidean distance, closing speed, lag/lead at ±1,2,3,5; nearest-neighbor counts within radius (nfl silver)
- **Body/scene normalization**: hip-relative coords, torso-length scaling (gesture); team centroid + spread (nfl)
- **Cyclical encoding** for angles (sin/cos, period 360°) beats raw degrees for trees
- **Segment-summary stats** (mean/std/min/max/quantiles) over proposed boundaries > frame-level windows on small data
- **Monotone constraints** on physically-meaningful features (distance↓ → contact↑) inject domain priors
- **Modality priority**: Audio(pretrained) ≈ Skeleton/Tracking >> Depth/Video >> RGB

## Tuning patterns
- **Imbalance**: negative downsampling (10:1) + early stopping > scale_pos_weight tuning (nfl run 4)
- **Threshold optimization** per sub-model on fine grid (P2P ~0.49, P2G ~0.54); coarse 20-point grid leaves ~0.01 MCC
- **GBDT settings**: 500–1000 estimators, depth 6–8, GPU `hist`, early stopping (P2G stops at 15–32 iters)
- **Post-processing**: median filter 11–15fr, min duration ~8fr/300ms, merge gap ~5; Gaussian temporal smoothing for persistent events
- **Ensemble weights**: greedy search optimizing the **competition metric directly** (Levenshtein/MCC) on group-OOF, with temperature calibration before stacking
