# Task Memory: tensorflow-speech-recognition-challenge

## Task profile
- Type: Multi-class audio classification (keyword spotting, 10 target + unknown + silence ≈ 12 classes)
- Dataset: Short WAV clips (~1s), speaker-identified, background noise clips for silence synthesis; test distribution skewed vs training prior
- Metric: Accuracy (higher is better)
- Medal thresholds: Unknown; both runs finished below median

## Best result
- Score: 0.35316 | Medal: none | Run: 20260419_063528
- Approach: Small Conformer (4 layers, 144 dim, 4 heads) on 40-bin log-mel spectrograms (n_fft=512) with speaker-grouped 5-fold CV. Hit 98.71% val accuracy but test submission suffered 64-point drop due to class prior mismatch (predicted 64% unknown vs expected ~10%).

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260304_061241 | 0.02951 | none | Compact Conformer + pre-computed spectrograms, 2-fold CV; val 96.11% but near-zero test (likely leakage/label bug) |
| 20260419_063528 | 0.35316 | none | Log-mel + small Conformer, speaker-grouped 5-fold CV; val 98.71%, still collapsed on unknown class prior |

## Proven strategies
- Log-mel spectrograms (40 bins, n_fft=512) + Conformer or SE-ResNet consistently hits 97–99% val accuracy
- Pre-computing spectrograms eliminates CPU bottleneck (10–50× epoch speedup); required under 240-min wall budget
- Speaker-grouped CV splits prevent 15–20pt speaker-memorization leakage; random CV is misleading
- Silence synthesis from background noise clips + sampling unknown at training prior ~10% prevents boundary-class collapse
- TEST_MODE sanity checks catch pipeline bugs before burning full training budget
- Conformer beats SmallCNN by ~1.6pt val acc; SmallCNN remains competitive fallback

## Failure patterns
- **Massive val→test gap is the dominant failure** (96% val → 0.03 test; 99% val → 0.35 test). Root cause in run 1 never diagnosed; run 2 traced to class prior mismatch
- Class collapse / single-class predictions in multiple candidates (Cands 2/3/5/6 in run 1) — silent failure without per-class logging
- SincNet / learnable waveform filterbanks catastrophically unstable (3.6% val acc); needs careful init
- Complex architectures (Deep VGG + pseudo-labels, CRNN/TDNN + auxiliary CTC) repeatedly killed by judge for pipeline/spec issues
- 2-fold CV too aggressive for audio (fold variance 0.16 vs 0.96 observed)
- Test-time calibration patches added late (step 8–9) go unvalidated and ship broken

## Feature engineering playbook
- Default: 40-bin log-mel, n_fft=512, pre-computed to disk before training
- Build silence class from provided background noise clips (synthesis, not empty audio)
- Sample unknown class at training prior (~10%) to match boundary
- **Always emit and log submission class distribution** — compare against expected prior before submitting
- Speaker-stratified (GroupKFold by speaker ID) is mandatory, not optional

## Tuning insights
- Small Conformer (4 layers, 144 dim, 4 heads) sufficient; deeper nets don't help under wall budget
- Prefer 5-fold speaker-grouped over 2-fold; fold variance is the signal
- Log per-class accuracy during training, not just overall — catches collapse mid-run
- Test-time logit adjustment / prior calibration must be implemented early and validated by tuner, not bolted on in final reviser step
- Waveform-domain models (SincNet) not worth the risk vs log-mel baseline

## Open questions
- Root cause of run-1 val 96% → test 0.03 collapse (leakage, label encoding, or preprocessing asymmetry) never identified
- Actual test-set class distribution unknown; reviser's prior values were hand-tuned guesses
- Whether run-2's test-time logit adjustment was correctly applied or silently no-op'd
- Why SincNet fails so hard (3.6%) when log-mel variants hit 97%+ — init vs inherent instability
