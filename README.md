# TM Trading V9.2 Core

## Architecture Overhaul
This is the pristine, consolidated V9.2 repository designed to strip away the legacy context drift and failed parameters of V7.x. 

### Key Lessons Learned from V7.x:
1. **D4/CVD Faux Alpha:** Annualized Sharpe on fixed TP/SL frameworks using D4 signal completely collapsed once realistic taker-fees and execution horizons were calculated. The event counts were fundamentally too low to survive transaction costs.
2. **Parameter Drift:** We found catastrophic divergence between Claude/Codex/Human states (e.g. 0.3% vs 3.0% stop-losses). This repository serves as the definitive **Source of Truth** to prevent assistant hallucinations.
3. **Storage Efficiency:** Duplicating raw HFT data to hot caches broke the NVMe. V9.2 strictly enforces a Cold-Storage (Seagate) -> Hot-Cache (NVMe) Tier 2 pipeline.

## V9.2 Roadmap Implementation
The transition to V9.2 is built on the following pillars:

- **Tier-2 Caching:** `scripts/v92_tier2_cache_builder.py` transforms massive zipped Spot trades into lightning-fast, backtest-ready Polars Volume Bars (with Volume Delta/VWAP).
- **Microstructure OFI:** `features/microstructure_ofi.py` processes raw Binance Futures L2 orderbook diff streams into clean Order Flow Imbalance.
- **Contextual Filters:** Strict *No-Leakage* implementation of Swing High/Lows, ATR Regimes, and ADR Exhaustion.
- **Dynamic Exits:** Replacing static TP/SL with ATR trailing, CVD reversals, and Time Stops.

## Layout
* `docs/`: Source of truth roadmaps and schematics.
* `scripts/`: Pipelines for caching, extraction, and diagnostics.
* `features/`: The core signal algorithms (OFI, Market Structure).
* `exits/`: Dynamic exit state-machines.
* `data/hft/`: Manifests and schemas for the institutional data pipeline.
