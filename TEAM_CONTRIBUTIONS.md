# AI Institutional Trading Bot — Team Contributions Report

> **Project**: Institutional-Grade AI Cryptocurrency Trading Bot  
> **Version**: 3.0 Autonomous | **Date**: May 2026  
> **Institution**: VIT-AP University, B.Sc. Data Science  

---

## Team Members

| # | Name | Roll Number | Primary Domain |
|---|------|-------------|----------------|
| 1 | **Sagnik Bhowmick** | 23BSD7045 | ML Pipeline & Backend Engineering |
| 2 | **Mantri Krishna Sri Inesh** | 23BSD7023 | Frontend Architecture & Trading UI |
| 3 | **Sunkavalli LSVP SeshaSai** | 23BSD7019 | Strategy Engine & Risk Management |

---

## Project Overview

This project implements an institutional-grade AI trading bot that fuses deep learning with advanced financial mathematics to trade cryptocurrency futures. The system comprises **7,600+ lines of production code** across TypeScript and Python, implementing **9 trading strategies**, **15+ technical indicators**, and a **3-layered risk management framework**. Key innovations include Fractional Differencing (d=0.4), Triple Barrier Method labeling, TCN deep learning, TDA regime detection, and Walk-Forward Analysis with plateau scoring.

---

# PHASE 1: Data Engineering, ML Pipeline & Backend Infrastructure

## Contributor: Sagnik Bhowmick (23BSD7045)

### 1.1 Dataset Construction — `build_dataset.py` (241 lines)

Designed and implemented the complete data acquisition and feature engineering pipeline that forms the foundation of the ML system.

**Key Deliverables:**
- Built the pipeline to fetch 500 hours of historical OHLCV data from Binance Futures API
- Integrated derivatives-market microstructure data: Open Interest and Funding Rates from dedicated API endpoints (`/fapi/v1/openInterest`, `/fapi/v1/fundingRate`)
- Engineered **18 feature columns** including returns, rolling volatility (20-period), Bollinger Bands (upper, lower, width), EMA-12/26, ADX-14, VWMA-20, and fractionally differenced close price
- Implemented data integrity measures: forward-fill for sparse funding/OI data, NaN dropping post-computation, and timestamp-indexed CSV output

### 1.2 Fractional Differencing — `frac_diff.py` (134 lines)

Implemented Marcos López de Prado's Fixed-Width Window Fractional Differencing (FFD) — the most mathematically sophisticated component of the system.

**Key Deliverables:**
- `get_weights_ffd(d, thres)` — Generates FFD weights using the recursive formula: w_k = -w_{k-1} × (d - k + 1) / k, with early termination when |w_k| < threshold
- `apply_frac_diff(series, d)` — Applies fractional differencing via convolution with the weight vector
- `find_optimal_d(series)` — Binary search algorithm testing d from 0.05 to 1.0, using ADF test (p < 0.05) to find minimum d achieving stationarity
- Validated that d=0.4 retains ~60% of price memory while achieving stationarity — resolving the fundamental stationarity-vs-memory trade-off

### 1.3 Triple Barrier Method — `tbm_labeler.py` (92 lines)

Implemented meta-labeling from *Advances in Financial Machine Learning*.

**Key Deliverables:**
- Upper barrier at +1.5% (take-profit), lower barrier at -1.0% (stop-loss), vertical barrier at 24 periods (time expiry)
- First-barrier-touched logic: label = 1 (WIN) if upper hit first, label = 0 (LOSS) if lower or time expiry hit first
- Look-ahead bias prevention by dropping the last `time_limit` rows from the dataset

### 1.4 TCN Model Training — `train_tcn.py` (289 lines)

Designed and trained the Temporal Convolutional Network architecture.

**Key Deliverables:**
- 3-layer residual block architecture with exponentially increasing dilations (1, 2, 4), achieving a receptive field of 15 timesteps
- Causal convolution enforcement via asymmetric left-only padding: `padding = (kernel_size - 1) × dilation`
- Batch normalization, ReLU activation, and 0.2 dropout per block
- Global average pooling → Linear → Sigmoid for probability output
- **Platt Scaling calibration**: Logistic regression fitted on raw model outputs to convert sigmoid scores into well-calibrated probabilities, critical for the 0.65 trade-entry threshold
- Training: Adam optimizer, lr=1e-3, BCEWithLogitsLoss, 100 epochs with early stopping, 20% validation split

### 1.5 FastAPI Backend — `api.py` (202 lines)

Built the REST API server for real-time ML inference.

**Key Deliverables:**
- `POST /predict` endpoint: fetches live Binance data, engineers 18 features, normalizes via rolling Z-score (window=20), and runs XGBoost inference
- Rolling Z-score normalization: `(df - rolling_mean) / (rolling_std + 1e-8)` — adaptive to regime changes
- Model loading with graceful fallback: primary from `institutional_xgboost_model.json`, fallback to HOLD signal
- Signal generation: probability ≥ 0.65 → LONG, ≤ 0.35 → SHORT, else HOLD
- CORS middleware for frontend communication, health check endpoint

### 1.6 Advanced Training Pipeline

Extended the ML pipeline with additional model architectures and validation:

- **`train_tbm_model.py` / `train_tbm_model_v2.py`** — XGBoost model training with TBM labels, per-symbol and regime-based thresholds
- **`train_global_model.py`** — Multi-asset global model training across top liquid crypto pairs
- **`train_institutional_model.py`** — Institutional-grade model with enhanced feature sets
- **`train_meta_labeler.py`** — Meta-labeling ensemble combining TCN + XGBoost predictions
- **`build_global_dataset.py`** — Global dataset construction across multiple symbols
- **`purged_kfold.py`** — Purged K-Fold cross-validation preventing temporal data leakage
- **`run_full_pipeline.py`** — End-to-end orchestrator: data → features → labels → train → calibrate → validate
- **`feature_ablation.py`** — Systematic feature importance analysis via ablation studies
- **`walk_forward_simulator.py`** — Python-side walk-forward validation with expanding window

### 1.7 Live Trading Bot — `live_bot.py` (31,870 bytes)

Implemented the production live trading system:

- **`live_bot.py`** — Full live trading loop with real-time signal generation, order execution, and position management
- **`live_features.py`** — Real-time feature computation matching the training pipeline exactly
- **`live_orders.py`** — Order execution layer with Binance Futures API integration
- **`live_state.py`** — Persistent state management via SQLite (`live_bot_state.db`)
- **`live_bot_monitor.py`** — Health monitoring and alerting for the live bot process

### Summary of Sagnik's Contributions

| Area | Files | Total Lines |
|------|-------|-------------|
| Data Pipeline | `build_dataset.py`, `build_global_dataset.py`, `build_institutional_dataset.py` | ~700+ |
| ML Core | `frac_diff.py`, `tbm_labeler.py`, `train_tcn.py`, `train_model.py` | ~800+ |
| Advanced Training | `train_tbm_model_v2.py`, `train_meta_labeler.py`, `purged_kfold.py`, `feature_ablation.py` | ~1,200+ |
| Backend API | `api.py` | ~200 |
| Live Bot | `live_bot.py`, `live_features.py`, `live_orders.py`, `live_state.py` | ~1,400+ |
| **Total** | **15+ Python files** | **~4,300+ lines** |

---

# PHASE 2: Frontend Architecture, Dashboard & User Experience

## Contributor: Mantri Krishna Sri Inesh (23BSD7023)

### 2.1 Application Architecture — Next.js 16 / React 19

Designed and implemented the complete frontend application using cutting-edge web technologies.

**Key Deliverables:**
- App Router architecture with server-side rendering, API route proxying, and component-based design
- Full TypeScript type safety across the entire frontend codebase
- Dark theme design system: background `#121212`, surface `#1A1A1A`, accent `#1DB954`

### 2.2 Landing Page — `app/page.tsx` (203 lines)

Created the brand identity and first-impression experience.

**Key Deliverables:**
- Animated SVG background with upward-trending graph using `stroke-dasharray` animation
- Ambient glow effect: 800px Gaussian blur in brand green (#1DB954)
- Feature grid showcasing 6 key innovations: TDA Regime Detection, Global TCN Models, Triple Barrier Labeling, HRP Risk Management, Fractional Differencing, Agentic Orchestration
- Research Core section with team member attribution
- Engine Hardening metrics with status indicators

### 2.3 Institutional Trading Dashboard — `binance-trading-dashboard.tsx` (~4,200 lines)

Built the flagship component — the largest single module in the entire system.

**Key Deliverables:**
- Professional-grade candlestick charting via Lightweight Charts v5.1.0 with MA-7/25/99 overlays
- Real-time chart updates via WebSocket with live candle rendering
- Volume histogram with institutional-style visualization
- Position management panels: entry/exit prices, PnL tracking, SL/TP levels
- Strategy selection controls with confidence indicators
- Performance analytics: equity curves, win rate charts, PnL distribution
- Asset selection workflow with top-30 liquid crypto pairs

### 2.4 ML Analytics Panel — `ml-analytics.tsx` (416 lines)

**Key Deliverables:**
- Real-time probability display from the ensemble model
- Market regime indicator with visual classification
- Active strategy and confidence level displays
- Historical prediction accuracy visualization using Recharts
- Probability distribution charts for calibration assessment

### 2.5 Strategy Configurator — `strategy-configurator.tsx` (408 lines)

**Key Deliverables:**
- Granular parameter controls: SL/TP percentages, risk-per-trade limits, max concurrent positions
- Strategy-specific parameter tuning: RSI thresholds, BB multipliers, grid spacing
- Real-time parameter application to active trading sessions
- WFA trigger integration for parameter validation

### 2.6 Supporting Dashboard Components

| Component | Lines | Contribution |
|-----------|-------|-------------|
| `dashboard.tsx` | 256 | Portfolio overview with KPI cards |
| `financial-chart.tsx` | 285 | TradingView-style charts with Lightweight Charts |
| `global-trade-config.tsx` | 239 | System-wide trade settings panel |
| `price-chart-pro.tsx` | 236 | Advanced multi-MA charting |
| `price-chart-crypto.tsx` | 206 | Crypto-specific chart component |
| `bot-settings.tsx` | 208 | Bot parameter controls UI |
| `coin-table.tsx` | 183 | Asset listing with live stats |
| `trading-interface.tsx` | 137 | Order entry panel |
| `crypto-search.tsx` | 135 | Fuzzy search across coins |
| `trading-pair-selector.tsx` | 126 | Pair selection dropdown |
| `app-header.tsx` | 67 | Navigation header |
| `trade-history.tsx` | 60 | Recent trade log |
| `theme-provider.tsx` | — | Dark/light theme switching |

### 2.7 Page Routes & Navigation

**Key Deliverables:**
- `app/dashboard/` — Main trading dashboard page
- `app/trade/` — Active trading interface
- `app/coins/` — Coin listing and search
- `app/crypto/` — Individual crypto detail pages
- `app/settings/` — Bot configuration panel
- `app/layout.tsx` — Root layout with theme provider and metadata
- `app/globals.css` — Global styles and design tokens

### 2.8 API Routes & Data Services

**Key Deliverables:**
- `app/api/coins/[id]/history/route.ts` — CoinGecko proxy endpoint to avoid CORS issues
- `lib/binance-websocket.ts` (431 lines) — Singleton WebSocket manager with kline, trade, and multi-ticker streams, CoinGecko fallback, MA pre-computation (MA7/25/99), auto-reconnect with exponential backoff
- `lib/binance-client.ts` (423 lines) — Full authenticated Binance API client with HMAC-SHA256 signing, LOT_SIZE/TICK_SIZE rounding, market/limit/stop orders, testnet support
- `lib/coingecko-service.ts` — CoinGecko API integration for fallback data
- `lib/coin-cache.ts` — Client-side data caching layer
- `lib/mock-data.ts` — Synthetic data generation for development/fallback
- `lib/use-live-market.ts` — Custom React hook for live market data

### 2.9 UI Component Library Integration

**Key Deliverables:**
- Integrated 25+ Radix UI primitives for accessibility: Accordion, Dialog, Dropdown, Tabs, Tooltip, etc.
- Tailwind CSS 3.4.17 with `tailwind-merge` for class deduplication
- `class-variance-authority` for component variant management
- `tailwindcss-animate` for animation utilities
- Recharts 2.15.4 for equity curves, PnL distribution, and performance charts

### Summary of Inesh's Contributions

| Area | Files | Total Lines |
|------|-------|-------------|
| App Pages & Layout | `page.tsx`, `layout.tsx`, route pages | ~500+ |
| Trading Dashboard | `binance-trading-dashboard.tsx` | ~4,200 |
| Analytics & Config | `ml-analytics.tsx`, `strategy-configurator.tsx` | ~824 |
| Dashboard Components | 12 additional components | ~2,100+ |
| Data Services | WebSocket, Binance client, CoinGecko, cache | ~1,300+ |
| UI Library | Radix UI integration, theme, styles | ~500+ |
| **Total** | **25+ TSX/TS files** | **~9,400+ lines** |

---

# PHASE 3: Trading Strategies, Risk Management & Intelligent Orchestration

## Contributor: Sunkavalli LSVP SeshaSai (23BSD7019)

### 3.1 Trading Strategies — `trading-strategies.ts` (586 lines)

Designed and implemented the complete 9-algorithm strategy suite.

**Key Deliverables:**

**1. Moving Average Crossover Strategy**
- Golden Cross detection (SMA20 > SMA50 > SMA200) → BUY at 0.85 confidence
- Death Cross detection → SELL at 0.85 confidence
- Partial alignment detection (within 1%) at 0.60 confidence

**2. MACD Strategy**
- Histogram flip detection for bullish/bearish crossovers at 0.75 confidence
- Momentum divergence signals at 0.65 confidence
- Stateful tracking of previous histogram values

**3. Mean Reversion Strategy (Bollinger + RSI)**
- Price < BB Lower AND RSI < 30 → BUY at 0.85 confidence
- Extreme RSI (<20, >80) fallback signals at 0.70 confidence

**4. Grid Trading Strategy**
- Configurable N-level grid with percentage spacing
- Buy/sell cascade logic for ranging markets
- Optimal for ADX < 20, low BB width conditions

**5. RSI Divergence Strategy**
- Bullish divergence: price Lower Low + RSI Higher Low → BUY at 0.85
- Bearish divergence: price Higher High + RSI Lower High → SELL at 0.85
- Historical extreme tracking with real-time updates

**6. Bollinger Breakout Strategy** — Volatility expansion signals at 0.80 confidence

**7. VWAP Trend Strategy** — Institutional flow detection via price-VWAP relationship with 0.2% buffer

**8. Ichimoku Cloud Strategy** — Full 5-line system (Tenkan/Kijun/Senkou A&B) at 0.90 confidence for full alignment

**9. Pivot Point Reversal Strategy** — Rolling support/resistance from 50-candle data with 0.2% tolerance

### 3.2 Strategy Engine — `strategy-engine.ts` (447 lines)

**Key Deliverables:**
- Multi-Signal Ensemble (Voting System): aggregates 4 sub-strategies with democratic voting requiring ≥2 agreement
- Confidence formula: `min(0.95, 0.50 + votes × 0.20)` averaged with individual strategy confidences
- Top-K asset selection via composite scoring
- Kelly Criterion position sizing: `f* = (winRate × avgWin - (1-winRate) × avgLoss) / avgWin` with safety multiplier
- Multi-condition exit logic: SL, TP, time-in-trade, VaR breach, probability drop

### 3.3 Risk Management System — `risk-management.ts` (394 lines)

Implemented the 3-layered defense-in-depth risk architecture.

**Key Deliverables:**

**Layer 1 — Position-Level Controls:**
- Fixed-risk position sizing: `positionSize = (balance × maxRiskPerTrade) / |entryPrice - stopLoss|`
- ATR-based dynamic stop-loss placement
- Risk-reward ratio enforcement

**Layer 2 — Portfolio-Level Controls:**
- Maximum concurrent positions limit
- Total exposure limits
- Correlation-based diversification
- Maximum allocation per asset

**Layer 3 — Account-Level Circuit Breakers:**
- Maximum drawdown: 10% from peak → halt all trading
- Daily loss limit: 5% of balance → stop new positions
- Consecutive losses: 3 in a row → 15-minute cooldown
- Position count limiter → queue new signals

**Dynamic Risk Scaling ("Gas Pedal"):**
- Confidence > 0.85 → 1.5x risk multiplier (aggressive)
- Confidence 0.65–0.85 → 1.0x (standard)
- Confidence < 0.65 → REJECT trade entirely

**Trade Journal:** Complete trade recording with symbol, action, entry/exit prices, quantity, PnL, timestamp

### 3.4 Trading Engine — `trading-engine.ts` (393 lines)

**Key Deliverables:**
- Real-time execution loop: WebSocket kline → price update → strategy evaluation → risk validation → paper trade execution
- Position lifecycle management: entry, monitoring, SL/TP checking, trailing stops, exit
- WebSocket integration for live candlestick and trade streams
- Auto-reconnect with 3-second delay and exponential backoff (max 30s)

### 3.5 AI Configuration Engine — `ai-config-engine.ts` (1,336 lines)

Built the largest and most complex module — the central decision-making orchestrator.

**Key Deliverables:**
- `detectMarketRegime()` — Classifies into STRONG_TREND, WEAK_TREND, RANGING, VOLATILE, BREAKOUT using ADX/BB/ATR
- `analyzeMultiTimeframe()` — Confluence scoring across 1m, 5m, 15m, 1h, 4h
- `detectSMCPatterns()` — Smart Money Concepts: Order Blocks, Fair Value Gaps, Liquidity Sweeps
- `calculateSentimentDivergence()` — Hidden divergences between price action and volume/RSI
- `runPaperBacktest()` — Strategy simulation on last 100 candles before deployment
- `generateConfiguration()` — Final trading parameter output (strategy, leverage, SL/TP, position size, confidence threshold)

**Regime → Strategy Mapping:**

| Regime | Strategy | Risk Multiplier |
|--------|----------|----------------|
| STRONG_TREND | Trend Following (MA Crossover) | 1.2x |
| RANGING | Grid Trading / Mean Reversion | 1.0x |
| VOLATILE | Mean Reversion | 0.5x |
| BREAKOUT | Multi-Signal Ensemble | 1.5x |

### 3.6 ML Models Module — `ml-models.ts` (451 lines)

**Key Deliverables:**
- TypeScript port of the Python feature extraction pipeline
- Triple Barrier Labeling implementation (TS version) with configurable TP/SL/time parameters
- Simplified TCN prediction simulation for browser-side inference
- TDA Persistence Landscapes: `computePersistenceLandscape()` returning loop count (H1 Betti), components (H0), max persistence
- Ensemble output with probability, confidence, regime, features, and TDA score

### 3.7 Technical Indicators — `technical-indicators.ts` (339 lines)

**Key Deliverables:**
- Complete indicator suite: SMA (20/50/200), EMA (12/26), MACD (12/26/9), RSI (14), Bollinger Bands (20/2σ), ATR (14), VWAP (50), Ichimoku (9/26/52), ADX (14 with Wilder's smoothing)
- Rolling window architecture with 500-candle buffer
- ATR fallback for flat candles using close-to-close absolute changes

### 3.8 Market Analyzer — `market-analyzer.ts` (169 lines)

**Key Deliverables:**
- Composite trend scoring (-5 to +5) across 5 indicators with ADX amplifier
- Volatility classification: HIGH/NORMAL/LOW using dual BB-width + ATR% metrics
- Sentiment engine: 5-state mood from EXTREME_FEAR to EXTREME_GREED using RSI + SMA distance
- Strategy selection matrix mapping 10+ condition combinations to optimal strategies

### 3.9 Supporting Modules

**Math Utilities — `math-utils.ts` (322 lines):**
- Statistical: mean, std, correlation, covariance matrix
- Financial: VaR (95%), CVaR, Sharpe Ratio (√252 annualization)
- TypeScript port of Fractional Differencing with simplified ADF test
- Hierarchical Risk Parity: inverse-volatility weighting

**Walk-Forward Analyzer — `walk-forward-analyzer.ts` (502 lines):**
- 80/20 in-sample/out-of-sample split
- 41 parameter combinations across 5 strategy families
- Plateau scoring: neighbor-averaged OOS performance for robustness
- Composite ranking: `Score = OOS_PnL% + PlateauScore × 0.5`

**Agentic Orchestrator — `agentic-orchestrator.ts` (227 lines):**
- Regime-based strategy selection across 6 market states
- Anti-whipsaw: 5-minute linger rule + 75% hysteresis threshold
- Penalty system: 3 consecutive losses → halve confidence, <40% → force safety strategy
- Volatility scaling: EXTREME_FEAR → Risk ×0.7 / TP ×2.0

**Gemini AI Advisor — `gemini-advisor.ts` (122 lines):**
- Gemini 2.5 Flash integration via `@google/generative-ai`
- Output guardrails: price target stripping, probability claim removal, action validation, 800-char truncation, disclaimer prefix
- Local rule-based fallback when API unavailable

**Backtester — `backtester.ts` (301 lines):**
- Full simulation with slippage (0.05%) and commission (0.1%)
- Metrics: Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate
- Balance history for equity curve analysis

### Summary of SeshaSai's Contributions

| Area | Files | Total Lines |
|------|-------|-------------|
| Trading Strategies | `trading-strategies.ts` | 586 |
| Strategy Engine | `strategy-engine.ts` | 447 |
| Risk Management | `risk-management.ts` | 394 |
| Trading Engine | `trading-engine.ts` | 393 |
| AI Config Engine | `ai-config-engine.ts` | 1,336 |
| ML Models (TS) | `ml-models.ts` | 451 |
| Technical Indicators | `technical-indicators.ts` | 339 |
| Market Analyzer | `market-analyzer.ts` | 169 |
| Math Utilities | `math-utils.ts` | 322 |
| Walk-Forward Analyzer | `walk-forward-analyzer.ts` | 502 |
| Agentic Orchestrator | `agentic-orchestrator.ts` | 227 |
| Gemini Advisor | `gemini-advisor.ts` | 122 |
| Backtester | `backtester.ts` | 301 |
| **Total** | **13 TypeScript files** | **~5,589 lines** |

---

# Consolidated Summary

## Overall Contribution Breakdown

| Phase | Contributor | Domain | Files | Lines |
|-------|------------|--------|-------|-------|
| **Phase 1** | Sagnik Bhowmick (23BSD7045) | ML Pipeline & Backend | 15+ Python files | ~4,300+ |
| **Phase 2** | Mantri Krishna Sri Inesh (23BSD7023) | Frontend & Dashboard | 25+ TSX/TS files | ~9,400+ |
| **Phase 3** | Sunkavalli LSVP SeshaSai (23BSD7019) | Strategy & Risk Engine | 13 TypeScript files | ~5,589 |
| | | **Grand Total** | **50+ files** | **~19,000+ lines** |

## Technology Ownership

```
Sagnik Bhowmick          Mantri Krishna Sri Inesh       Sunkavalli LSVP SeshaSai
─────────────────        ────────────────────────       ────────────────────────
Python / FastAPI         Next.js 16 / React 19          TypeScript Core Library
PyTorch (TCN)            Lightweight Charts              9 Trading Strategies
XGBoost                  Recharts                        Risk Management (3-layer)
Fractional Differencing  Radix UI + Tailwind CSS         AI Config Engine (1,336 LOC)
Triple Barrier Method    WebSocket Integration           Walk-Forward Analyzer
Platt Scaling            CoinGecko/Binance Services      Agentic Orchestrator
Live Trading Bot         Dashboard UI (~4,200 LOC)       Technical Indicators (15+)
Model Training Pipeline  API Routes & Data Layer         Math Utilities & HRP
```

## Key Innovation Ownership

| Innovation | Primary Contributor |
|-----------|-------------------|
| Fractional Differencing (d=0.4) | Sagnik Bhowmick |
| Triple Barrier Method Labeling | Sagnik Bhowmick |
| TCN Architecture + Platt Scaling | Sagnik Bhowmick |
| Live Trading Bot Infrastructure | Sagnik Bhowmick |
| Institutional Trading Dashboard | Mantri Krishna Sri Inesh |
| Real-time WebSocket Charts | Mantri Krishna Sri Inesh |
| ML Analytics Visualization | Mantri Krishna Sri Inesh |
| Multi-source Data Resilience | Mantri Krishna Sri Inesh |
| 9-Algorithm Strategy Ensemble | Sunkavalli LSVP SeshaSai |
| 3-Layer Risk Framework + Circuit Breakers | Sunkavalli LSVP SeshaSai |
| Smart Money Concepts Detection | Sunkavalli LSVP SeshaSai |
| Walk-Forward Analysis + Plateau Scoring | Sunkavalli LSVP SeshaSai |
| Agentic Orchestrator (Anti-Whipsaw) | Sunkavalli LSVP SeshaSai |
| Gemini AI Advisory with Guardrails | Sunkavalli LSVP SeshaSai |

---

> **End of Team Contributions Report**  
> Covers all 50+ source files across Python and TypeScript  
> Generated: May 2026
