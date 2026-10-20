# Task Memory: random-acts-of-pizza

## Task profile
- Type: Binary classification
- Dataset: ~2,878 Reddit posts requesting free pizza; tabular + short text (title + body); 24.8% positive class (75:25 imbalance); features include account metadata, karma, temporal info, subreddit history, and text fields
- Metric: AUC-ROC (higher is better)
- Medal thresholds: Bronze ≥ 0.6921, Silver ≈ 0.78, Gold unknown

## Best result
- Score: 0.78067 | Medal: Silver | Run: 20260228_091354
- Approach: XGBoost with custom sentiment lexicons (57 positive, 41 negative, urgency words), extended text readability features (vocabulary richness, type-token ratio), narrative structure indicators, TF-IDF→SVD(50), temporal cyclical encoding, and giver_known binary leakage feature. `scale_pos_weight` for class imbalance. 4 tuner rounds.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260210_061828 | 0.606 (1-epoch) | ungraded | TF-IDF + shallow NN; loss function bug (BCELoss instead of BCEWithLogitsLoss) |
| 20260210_064011 | 0.9977 (leaked) | ungraded | Ensemble meta-learner; `_at_retrieval` features caused data leakage; run aborted |
| 20260210_073418 | 0.64625 | none | XGBoost baseline; tuner plateau at ~0.65; BiLSTM redesign failed to produce artifact |
| 20260211_202754 | 0.7741 (CV) | ungraded | Stacking ensemble (LightGBM+XGB+LR+ET); giver_known feature discovered |
| 20260228_024128 | 0.77959 | Silver | LightGBM + 47 hand-crafted features + giver_known; 0.9124 CV |
| 20260228_091354 | 0.78067 | Silver | XGBoost + custom sentiment lexicons + narrative features; 0.7978 val AUC |

## Proven strategies
- **giver_username_if_known is the single most impactful feature**: 208/208 non-null giver entries in training are positive. Test set has 79 non-null values. Binary flag alone worth ~1-2% AUC. Must include.
- **Hand-crafted text features dominate embeddings at <3k samples**: char/word/sentence counts, punctuation ratios, exclamation marks, uppercase ratio, URL presence consistently outperform TF-IDF or transformer embeddings
- **TF-IDF → TruncatedSVD (50–100 components) is mandatory**: Raw TF-IDF (5000 features) causes severe overfitting; SVD stabilizes CV
- **Custom domain-specific sentiment lexicons beat generic NLP**: Curated word lists for gratitude, hardship, urgency, reciprocity (e.g., "student", "broke", "hungry", "rent", "lost job") outperform pretrained sentiment tools
- **Tree boosting models (LightGBM, XGBoost) dominate**: Consistently 0.77–0.78 AUC. XGBoost slightly edges LightGBM (~0.8% AUC) due to feature interaction handling
- **`scale_pos_weight` ≈ 3.0 for class imbalance**: Non-negotiable; all competitive runs used it
- **Stratified 5-fold CV is essential**: Random splits collapse to ~0.50 AUC; stratification critical with only ~714 positive samples
- **Temporal cyclical encoding (sin/cos)**: Avoids circular artifacts from hour/day-of-week features
- **Title-specific features add signal**: Brackets, question marks, formatting indicators in title are predictive

## Failure patterns
- **Data leakage via `_at_retrieval` columns**: `number_of_upvotes_of_request_at_retrieval`, `number_of_downvotes_of_request_at_retrieval` are post-request features. Including them inflates AUC to 0.99+ but collapses on test. Must exclude all `*_at_retrieval` fields.
- **Stacking/ensembles overfit on 2,878 samples**: Two-level stacking collapsed to AUC=1.0 on validation (memorization). Meta-learners see too little data. Simple weighted averaging also underperformed single models due to low base-learner diversity.
- **Transformer embeddings (DistilBERT, RoBERTa) consistently fail**: Frozen pretrained embeddings don't align with Reddit pizza domain. High-dimensional PCA (768→64) still overfits. Skip transformers entirely.
- **Complex PyTorch redesigns mid-run waste budget**: BiLSTM+Attention redesign consumed 36 coder turns and $1.54 with no deliverable. Stick to tree models and iterate on features.
- **Hyperparameter tuning alone plateaus at ~0.65 AUC (XGBoost) or ~0.79 (with good features)**: Tuner consistently reports "architectural changes needed" after 2 rounds. Gains come from feature engineering, not param sweeps.
- **Training AUC = 1.0 is a red flag**: XGBoost memorizes training data (train AUC 1.0 vs val 0.65). Indicates regularization insufficient or features too noisy—pivot to feature engineering.
- **Loss function implementation bugs**: Calculating `pos_weight` but applying unweighted `BCELoss()` instead of `BCEWithLogitsLoss`—silent failure that 1-epoch verification doesn't catch.

## Feature engineering playbook
- **Binary giver_known flag**: `1 if giver_username_if_known != null else 0` — strongest single feature
- **Text statistics**: char_count, word_count, sentence_count, avg_word_length, exclamation_count, question_mark_count, uppercase_ratio, URL_presence
- **Readability**: Flesch-Kincaid score, vocabulary richness (unique words / total words), type-token ratio, long-word proportion
- **Domain keyword indicators (binary)**: student, broke, hungry, rent, bills, lost job, unemployed, can't afford, paycheck, college, etc.
- **Custom sentiment scoring**: Curated positive list (~57 words: grateful, thanks, appreciate, blessing), negative list (~41 words: desperate, struggling, starving, behind on), urgency list
- **Narrative structure**: First-person pronoun density, story-telling cue presence
- **Temporal**: sin/cos encoding of hour, day-of-week; month, year, weekend flag
- **TF-IDF on concatenated title+body**: 1-3 grams, max 5000 features → SVD to 50 components
- **Engagement ratios**: karma per comment, account_age / post_count (proposed but not fully explored)
- **Avoid**: subreddit one-hot encoding (no signal), raw `_at_retrieval` fields (leakage), sentence-transformer embeddings (overfit)

## Tuning insights
- **LightGBM/XGBoost scale_pos_weight ≈ 3.0** matches the 75:25 imbalance ratio
- **SVD components: 50 optimal** (100 works but 50 more stable on this dataset size)
- **Early stopping at 50 rounds** prevents XGBoost overfitting
- **Tuner is effective for 1-2 rounds max**: First round typically gives +1-2% AUC; subsequent rounds show diminishing/negative returns
- **XGBoost slightly outperforms LightGBM** on final test scores despite similar CV performance
- **Random Forest and Extra Trees are weak base learners**: CV AUC ~0.52 standalone; not worth including in ensembles

## Open questions
- Can bigram/trigram keyword features ("lost job", "can't afford") push above 0.80 AUC? Only unigrams explored so far.
- Would focal loss (custom XGBoost objective) outperform `scale_pos_weight` for hard-negative mining?
- Threshold tuning on positive class: positive class recall was only 39.7% in early runs. Moving decision threshold may recover AUC without model changes.
- Blending (simple average of XGBoost + LightGBM predictions) vs stacking: stacking overfits, but does soft-voting average avoid this?
- Feature interaction terms (e.g., account_age × karma, request_length × sentiment_score) were proposed but never systematically tested.
- Is the 0.78 barrier a hard ceiling for this dataset, or can feature engineering alone break 0.80?
