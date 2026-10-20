# Pattern: transportation-mobility-forecasting

## When this applies
- Tabular regression on trip/ride records with geo coordinates and timestamps
- Large datasets (10M+ rows) requiring GPU-accelerated training
- Fare/duration/demand prediction with spatial-temporal structure
- Metric: RMSE or similar continuous error metric

## Proven patterns
- **LightGBM leaf-wise scales best on large geo-tabular data**: NYC taxi run 20260305_194201 achieved best 3.21 RMSE with LGBM vs 3.26 XGBoost on same features
- **Feature engineering > more data > HP tuning**: 10M rows + 45 features (3.26) beat 22M rows + 16 features (3.67); feature gains ~0.4 RMSE vs tuning <0.1
- **Domain-specific landmark features**: Airport distances (JFK/LGA/EWR) capture flat-rate routes critical for fare prediction
- **K-means spatial clustering (k=10)** complements haversine/Manhattan distances (run 20260210_221916)
- **GPU training mandatory** for 55M+ rows: LGBM device='gpu' or XGBoost gpu_hist enables <60min runs
- **Aggressive outlier filtering**: bbox + target clipping recovers ~0.2 RMSE; unfiltered ~2% invalid coords degrade models
- **Early stopping (50–100 rounds)** prevents overfit across all runs
- **Chunked CSV + float32 + parquet caching** prevents OOM on large trip logs

## Common pitfalls
- **Validation-submission gap is systemic**: runs 20260210_161554 (2.99→5.23) and 20260210_221916 (3.17→4.79) show ~50%+ degradation; audit val split and temporal drift
- **Temporal splits hurt when fare/tariff drifts**: random split outperforms temporal by 0.3–0.4 RMSE on NYC taxi (2009–2015 tariff creep)
- **Neural/MLP with embeddings underperforms trees**: best neural 3.79 vs tree 3.21 on NYC taxi even at 55M samples
- **Hybrid neural+GBDT redundant** when distance features are engineered (run 20260305_194201)
- **Wrong loss objective**: Huber/fair misaligns with RMSE metric; use direct RMSE
- **Under-using agent calls correlates with poor scores**: 4/10 calls → 6.51 RMSE on NYC taxi
- **HP tuner plateaus at ~8% above target after 2–3 iterations**; tuning alone cannot close gaps

## Model selection guide
- Rows >20M + tabular geo → **LightGBM leaf-wise (gpu)** first
- Need diversity for stacking → add **XGBoost gpu_hist** as second base
- Avoid **MLP/neural** unless trees are saturated and embeddings add new signal
- Avoid **regime-split models**: routing classifier errors outweigh gains

## Feature engineering patterns
- **Distance suite**: haversine (primary), Manhattan, straightness ratio, bearing
- **Landmark proximity**: euclidean to airports/hubs from both pickup and dropoff + binary flags
- **Temporal**: hour, DoW, month, year, sin/cos cyclical, rush-hour/weekend/holiday flags
- **Spatial binning**: lat/lon grid + k-means clusters (k=10 proven on NYC taxi)
- **Interactions**: distance × hour, hour × DoW capture demand/congestion
- **Target stats by zone/hour** (e.g., fare-per-mile historical) add signal but watch for leakage
- **Filter passenger/categorical extremes** [1, 6] range for taxi

## Tuning patterns
- **LGBM baseline**: lr=0.05, num_leaves=63, max_depth=8, 1000 trees, early_stop=50–100 (run 20260210_221916)
- **XGBoost progression**: max_depth 8→11, lr 0.025, 3000 iters helps internal but may not submit-generalize (run 20260210_161554)
- **Learning rate sweet spot**: 0.025–0.05
- **Diminishing returns after 2–3 tuner passes**; pivot to features/stacking instead
- **Use temporal CV as diagnostic** for validation-submission gap, even if random split trains better
