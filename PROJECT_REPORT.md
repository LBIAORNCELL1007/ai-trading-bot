# AI Institutional Trading Bot â€” Comprehensive Project Report

> **Version**: 3.0 Autonomous | **Date**: May 2026  
> **Authors**: Sagnik Bhowmick (23BSD7045), Mantri Krishna Sri Inesh (23BSD7023), Sunkavalli LSVP SeshaSai (23BSD7019)  
> **Institution**: VIT-AP University, B.Sc. Data Science  

---

## Table of Contents â€” Part 1

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Python ML Pipeline](#4-python-ml-pipeline)
5. [Backend API (FastAPI)](#5-backend-api-fastapi)
6. [AI Configuration Engine](#6-ai-configuration-engine)

---

## 1. Executive Summary

### 1.1 Project Vision

This project implements an **institutional-grade AI trading bot** that fuses cutting-edge deep learning with advanced financial mathematics to trade cryptocurrency futures on Binance. Unlike retail bots that rely on simple technical indicator crossovers, this system employs:

- **Temporal Convolutional Networks (TCN)** for non-linear time-series forecasting
- **Topological Data Analysis (TDA)** via persistence landscapes for regime detection
- **Fractional Differencing** (d=0.4) for stationarity while preserving long-memory dependencies
- **Triple Barrier Method (TBM)** for meta-labeling trade outcomes
- **Hierarchical Risk Parity (HRP)** for scientific portfolio allocation
- **Walk-Forward Analysis (WFA)** for overfitting-resistant strategy validation

The system is built as a full-stack monorepo: a **Python/FastAPI** backend handles ML model training and inference, while a **Next.js 16 / React 19** frontend provides a real-time trading dashboard with WebSocket price feeds, interactive charting, and autonomous strategy orchestration.

### 1.2 Key Metrics

| Metric | Value |
|---|---|
| Total Source Files | ~50+ (TypeScript, Python, TSX) |
| Lines of Code (lib/) | ~7,500+ lines |
| Python ML Pipeline | ~750+ lines |
| Trading Strategies | 9 distinct algorithms |
| Technical Indicators | 15+ (SMA, EMA, RSI, MACD, BB, ATR, ADX, VWAP, Ichimoku) |
| Supported Assets | Top 30 liquid crypto pairs |
| Risk Controls | Circuit breakers, VaR, drawdown limits, dynamic sizing |
| Execution Mode | Paper Trading (Live Simulation) |

### 1.3 Differentiating Factors

| Feature | Retail Bots | This System |
|---|---|---|
| Signal Generation | Single indicator | 9-strategy ensemble with voting |
| Stationarity | Raw prices / returns | Fractional differencing (d=0.4) |
| Labeling | Fixed thresholds | Triple Barrier Method |
| Risk | Fixed lot size | Kelly criterion + HRP + VaR |
| Validation | In-sample only | Walk-Forward Analysis with plateau scoring |
| Regime | None | TDA persistence + ADX/ATR classification |
| AI Advisory | None | Gemini 2.5 Flash with guardrails |

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        FRONTEND (Next.js 16 / React 19)            â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚ Dashboard â”‚  â”‚  Charts  â”‚  â”‚ ML Panel  â”‚  â”‚ Strategy Config  â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚        â”‚            â”‚              â”‚                 â”‚              â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚              Core Library Layer (lib/)                        â”‚  â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚  â”‚
â”‚  â”‚  â”‚ AI Config      â”‚  â”‚ Trading       â”‚  â”‚ Risk Management  â”‚ â”‚  â”‚
â”‚  â”‚  â”‚ Engine         â”‚  â”‚ Engine        â”‚  â”‚ (RiskManager)    â”‚ â”‚  â”‚
â”‚  â”‚  â”‚ (1336 lines)   â”‚  â”‚ (393 lines)   â”‚  â”‚ (394 lines)      â”‚ â”‚  â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚  â”‚
â”‚  â”‚          â”‚                  â”‚                    â”‚            â”‚  â”‚
â”‚  â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚  â”‚
â”‚  â”‚  â”‚        Supporting Modules                               â”‚ â”‚  â”‚
â”‚  â”‚  â”‚  â€¢ Strategy Engine    â€¢ ML Models    â€¢ Math Utils       â”‚ â”‚  â”‚
â”‚  â”‚  â”‚  â€¢ Market Analyzer    â€¢ Backtester   â€¢ WFA Analyzer     â”‚ â”‚  â”‚
â”‚  â”‚  â”‚  â€¢ Agentic Orch.      â€¢ Gemini AI    â€¢ Tech Indicators  â”‚ â”‚  â”‚
â”‚  â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                              â”‚                                     â”‚
â”‚                    Binance WebSocket + REST API                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     BACKEND (Python / FastAPI)                      â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”              â”‚
â”‚  â”‚ api.py       â”‚  â”‚ train_tcn.py â”‚  â”‚ build_      â”‚              â”‚
â”‚  â”‚ (Inference)  â”‚  â”‚ (Training)   â”‚  â”‚ dataset.py  â”‚              â”‚
â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤              â”‚
â”‚  â”‚ â€¢ XGBoost    â”‚  â”‚ â€¢ TCN Arch   â”‚  â”‚ â€¢ OHLCV     â”‚              â”‚
â”‚  â”‚ â€¢ Z-Score    â”‚  â”‚ â€¢ Platt Cal  â”‚  â”‚ â€¢ 18 Feats  â”‚              â”‚
â”‚  â”‚ â€¢ FracDiff   â”‚  â”‚ â€¢ Residual   â”‚  â”‚ â€¢ OI + FR   â”‚              â”‚
â”‚  â”‚ â€¢ /predict   â”‚  â”‚ â€¢ Blocks     â”‚  â”‚ â€¢ FracDiff   â”‚              â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜              â”‚
â”‚                                                                    â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                               â”‚
â”‚  â”‚ frac_diff.py â”‚  â”‚ tbm_labeler  â”‚                               â”‚
â”‚  â”‚ (Marcos LdP) â”‚  â”‚ .py (Labels) â”‚                               â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 2.2 Data Flow

```
Binance Futures API
    â”‚
    â”œâ”€â”€â–º build_dataset.py â”€â”€â–º alpha_dataset.csv
    â”‚                              â”‚
    â”‚                         tbm_labeler.py â”€â”€â–º labeled_alpha_dataset.csv
    â”‚                              â”‚
    â”‚                         train_tcn.py â”€â”€â–º institutional_tcn_model.pth
    â”‚                                          institutional_xgboost_model.json
    â”‚
    â”œâ”€â”€â–º api.py (/predict) â”€â”€â–º { probability, signal, confidence }
    â”‚
    â””â”€â”€â–º BinanceWebSocket â”€â”€â–º Real-time Klines â”€â”€â–º TradingEngine
                                                       â”‚
                                              StrategyEngine.evaluate()
                                                       â”‚
                                              RiskManager.validate()
                                                       â”‚
                                              Execute / Reject Trade
```

---

## 3. Technology Stack

### 3.1 Frontend

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16.1.6 | React framework with App Router |
| React | 19.2.0 | UI rendering |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 3.4.17 | Utility-first styling |
| Radix UI | Various | Accessible component primitives |
| Recharts | 2.15.4 | Data visualization |
| Lightweight Charts | 5.1.0 | TradingView-style candlestick charts |
| Lucide React | 0.454.0 | Icon library |
| Google Generative AI | 0.24.1 | Gemini 2.5 Flash integration |

### 3.2 Backend

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | ML pipeline language |
| FastAPI | Latest | REST API framework |
| PyTorch | â‰¥2.0.0 | TCN neural network training |
| XGBoost | Latest | Gradient boosted ensemble model |
| Pandas | â‰¥2.0.0 | Data manipulation |
| NumPy | â‰¥1.24.0 | Numerical computing |
| scikit-learn | â‰¥1.3.0 | ML utilities and preprocessing |
| Optuna | Latest | Hyperparameter optimization |
| statsmodels | Latest | ADF test for stationarity |
| ccxt | Latest | Exchange API abstraction |

### 3.3 External Services

| Service | Usage |
|---|---|
| Binance Futures API | OHLCV, Open Interest, Funding Rates |
| Binance WebSocket | Real-time price streaming |
| CoinGecko API | Historical price data fallback |
| Google Gemini 2.5 Flash | AI-powered market advisory |

---

## 4. Python ML Pipeline

### 4.1 Data Acquisition â€” `build_dataset.py` (241 lines)

This script constructs the feature-engineered dataset used for model training. It fetches 500 hours of historical data from Binance Futures and computes 18 feature columns.

#### 4.1.1 Data Sources

```python
# Three API endpoints are hit:
/fapi/v1/klines          # OHLCV candlestick data (500 periods)
/fapi/v1/openInterest    # Current open interest
/fapi/v1/fundingRate     # Perpetual funding rate
```

#### 4.1.2 Feature Engineering Pipeline

| # | Feature | Formula / Method | Purpose |
|---|---|---|---|
| 1 | `open` | Raw OHLCV | Price structure |
| 2 | `high` | Raw OHLCV | Intrabar volatility |
| 3 | `low` | Raw OHLCV | Intrabar volatility |
| 4 | `close` | Raw OHLCV | Primary price signal |
| 5 | `volume` | Raw OHLCV | Participation metric |
| 6 | `open_interest` | Binance API | Market positioning |
| 7 | `funding_rate` | Binance API | Sentiment proxy |
| 8 | `returns` | `close.pct_change()` | Price momentum |
| 9 | `volatility` | `returns.rolling(20).std()` | Risk metric |
| 10 | `bb_upper` | SMA20 + 2Ïƒ | Overbought level |
| 11 | `bb_lower` | SMA20 - 2Ïƒ | Oversold level |
| 12 | `bb_width` | (Upper - Lower) / SMA20 | Volatility squeeze |
| 13 | `vwma_20` | Volume-weighted MA | Institutional flow |
| 14 | `ema_12` | Exponential MA (12) | Short-term trend |
| 15 | `ema_26` | Exponential MA (26) | Medium-term trend |
| 16 | `adx_14` | Wilder's ADX (14) | Trend strength |
| 17 | `frac_diff_close` | FFD with d=0.4 | Stationary price |
| 18 | `tbm_label` | Triple Barrier Method | Trade outcome |

#### 4.1.3 Data Integrity Measures

- **Forward-fill** (`ffill`) applied to funding rates and open interest to handle missing values
- **NaN dropping** after feature computation to ensure clean training data
- Output saved as `alpha_dataset.csv` with timestamp index

### 4.2 Fractional Differencing â€” `frac_diff.py` (134 lines)

Implementation of Marcos LÃ³pez de Prado's Fixed-Width Window Fractional Differencing (FFD).

#### 4.2.1 Mathematical Foundation

Standard differencing (d=1) achieves stationarity but destroys memory:
```
Î”Â¹ xâ‚œ = xâ‚œ - xâ‚œâ‚‹â‚    (stationary but memoryless)
```

Fractional differencing with d=0.4 preserves long-range dependencies:
```
Î”áµˆ xâ‚œ = Î£â‚– wâ‚– Â· xâ‚œâ‚‹â‚–   where wâ‚– = -wâ‚–â‚‹â‚ Â· (d - k + 1) / k
```

#### 4.2.2 Key Functions

| Function | Purpose |
|---|---|
| `get_weights_ffd(d, thres)` | Generate FFD weights until |wâ‚–| < threshold |
| `apply_frac_diff(series, d)` | Apply fractional differencing to a pandas Series |
| `find_optimal_d(series)` | Binary search for minimum d achieving stationarity (ADF p < 0.05) |

#### 4.2.3 Stationarity Verification

The module uses the Augmented Dickey-Fuller test from `statsmodels`:
```python
adf_result = adfuller(diff_series.dropna())
p_value = float(adf_result[1])
# Accept if p_value < 0.05 (reject null hypothesis of unit root)
```

### 4.3 Triple Barrier Method â€” `tbm_labeler.py` (92 lines)

Implements the meta-labeling technique from *Advances in Financial Machine Learning*.

#### 4.3.1 Barrier Configuration

| Barrier | Type | Default | Description |
|---|---|---|---|
| Upper | Take Profit | +1.5% | Win condition |
| Lower | Stop Loss | -1.0% | Loss condition |
| Vertical | Time Expiry | 24 periods | Maximum holding time |

#### 4.3.2 Labeling Logic

```
For each candle i:
  Look ahead up to 24 periods
  If returns >= +1.5%  â†’ Label = 1 (WIN)
  If returns <= -1.0%  â†’ Label = 0 (LOSS)
  If time expires       â†’ Label = 0 (LOSS by time decay)
```

The last `time_limit` rows are dropped to prevent look-ahead bias.

### 4.4 TCN Training â€” `train_tcn.py` (289 lines)

#### 4.4.1 Architecture

```
Input (18 features Ã— sequence_length)
    â”‚
    â”œâ”€â”€ Residual Block 1 (channels=64, kernel=3, dilation=1)
    â”‚   â”œâ”€â”€ CausalConv1d â†’ BatchNorm â†’ ReLU â†’ Dropout(0.2)
    â”‚   â”œâ”€â”€ CausalConv1d â†’ BatchNorm â†’ ReLU â†’ Dropout(0.2)
    â”‚   â””â”€â”€ Skip Connection (1Ã—1 conv if channel mismatch)
    â”‚
    â”œâ”€â”€ Residual Block 2 (channels=64, kernel=3, dilation=2)
    â”‚   â””â”€â”€ Same structure, doubled receptive field
    â”‚
    â”œâ”€â”€ Residual Block 3 (channels=64, kernel=3, dilation=4)
    â”‚   â””â”€â”€ Same structure, quadrupled receptive field
    â”‚
    â”œâ”€â”€ Global Average Pooling
    â”‚
    â””â”€â”€ Linear â†’ Sigmoid â†’ P(win)
```

#### 4.4.2 Causal Convolution

Causal padding ensures the model cannot see future data:
```python
self.padding = (kernel_size - 1) * dilation  # Left-pad only
x = F.pad(x, (self.padding, 0))              # Zero-pad left side
```

#### 4.4.3 Platt Scaling Calibration

Post-training probability calibration using logistic regression:
```python
calibrator = LogisticRegression()
calibrator.fit(raw_logits, true_labels)
calibrated_prob = calibrator.predict_proba(new_logits)[:, 1]
```

This converts raw sigmoid outputs into well-calibrated probabilities, critical for the 0.65 confidence threshold used in production.

#### 4.4.4 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Loss Function | BCEWithLogitsLoss |
| Epochs | 100 (early stopping) |
| Batch Size | 32 |
| Dropout | 0.2 |
| Validation Split | 20% |

---

## 5. Backend API â€” `api.py` (202 lines)

### 5.1 Endpoint: `POST /predict`

The FastAPI server exposes a single prediction endpoint that:

1. **Fetches** 500 hours of live Binance Futures data (OHLCV + OI + Funding Rate)
2. **Engineers** the same 18 features used in training
3. **Normalizes** using rolling Z-score (window=20)
4. **Infers** using the loaded XGBoost model
5. **Returns** a structured prediction response

### 5.2 Normalization: Rolling Z-Score

```python
def rolling_zscore(df, window=20):
    mean = df.rolling(window=window).mean()
    std  = df.rolling(window=window).std()
    return (df - mean) / (std + 1e-8)  # epsilon to prevent division by zero
```

This adaptive normalization adjusts to regime changes rather than using static scaling.

### 5.3 Model Loading Strategy

```python
# Priority 1: Load from file
model = xgb.Booster()
model.load_model("institutional_xgboost_model.json")

# Priority 2: Fallback to basic prediction
if model is None:
    return {"signal": "HOLD", "confidence": 0.0}
```

### 5.4 Response Schema

```json
{
  "symbol": "BTCUSDT",
  "probability": 0.73,
  "signal": "LONG",
  "confidence": 0.73,
  "timestamp": "2026-05-01T12:00:00Z",
  "features_used": 18
}
```

The **0.65 threshold** determines signal generation:
- `probability >= 0.65` â†’ LONG signal
- `probability <= 0.35` â†’ SHORT signal
- Otherwise â†’ HOLD

---

## 6. AI Configuration Engine â€” `ai-config-engine.ts` (1,336 lines)

This is the **largest and most complex module** in the system. It serves as the central decision-making orchestrator on the frontend.

### 6.1 Core Responsibilities

| Function | Description |
|---|---|
| `detectMarketRegime()` | Classifies market into STRONG_TREND, RANGING, VOLATILE, etc. |
| `analyzeMultiTimeframe()` | Confluence scoring across 1m, 5m, 15m, 1h, 4h |
| `detectSMCPatterns()` | Smart Money Concepts: order blocks, FVGs, liquidity sweeps |
| `calculateSentimentDivergence()` | Compares price action vs. volume/RSI for hidden divergences |
| `runPaperBacktest()` | Simulates strategy on last 100 candles before deployment |
| `generateConfiguration()` | Outputs final trading parameters |

### 6.2 Market Regime Classification

```typescript
enum MarketRegime {
  STRONG_TREND    // ADX > 30, clear directional movement
  WEAK_TREND      // ADX 20-30, directional but uncertain
  RANGING         // ADX < 20, BB width < 2%
  VOLATILE        // ATR > 1.5% of price
  BREAKOUT        // BB squeeze followed by expansion
}
```

The regime determines which strategy gets activated:

| Regime | Strategy | Risk Multiplier |
|---|---|---|
| STRONG_TREND | Trend Following (MA Crossover) | 1.2x |
| RANGING | Grid Trading / Mean Reversion | 1.0x |
| VOLATILE | Mean Reversion (fade extremes) | 0.5x |
| BREAKOUT | Multi-Signal Ensemble | 1.5x |

### 6.3 Smart Money Concepts (SMC)

The engine detects three institutional patterns:

**Order Blocks**: Areas where institutional players have placed large orders
```typescript
// Detected when: Strong directional candle followed by
// price returning to the origin zone
orderBlock = {
  type: 'BULLISH' | 'BEARISH',
  zone: [highPrice, lowPrice],
  strength: volumeAtZone / averageVolume
}
```

**Fair Value Gaps (FVGs)**: Price imbalances where candle bodies don't overlap
```typescript
// Three-candle pattern where candle2.low > candle0.high (bullish)
// Indicates unfilled institutional orders
```

**Liquidity Sweeps**: Stop-loss hunts where price briefly breaches a level then reverses

### 6.4 Configuration Output

The engine produces a typed configuration object consumed by the Trading Engine:

```typescript
interface TradingConfig {
  strategy: string;
  leverage: number;        // 1x-10x based on regime
  stopLossPercent: number;  // Dynamic based on ATR
  takeProfitPercent: number;// Dynamic based on volatility
  positionSizePercent: number; // Kelly-derived
  maxConcurrentPositions: number;
  confidenceThreshold: number; // Minimum signal strength
}
```

---

*Continued in Part 2...*
# AI Institutional Trading Bot â€” Project Report (Part 2)

## Table of Contents â€” Part 2

7. [Trading Engine](#7-trading-engine)
8. [Strategy Engine & Algorithms](#8-strategy-engine--algorithms)
9. [Risk Management System](#9-risk-management-system)
10. [ML Models Module](#10-ml-models-module)
11. [Technical Indicators Engine](#11-technical-indicators-engine)
12. [Market Analyzer](#12-market-analyzer)
13. [Supporting Modules](#13-supporting-modules)

---

## 7. Trading Engine â€” `trading-engine.ts` (393 lines)

### 7.1 Overview

The Trading Engine is the real-time execution loop. It connects to Binance WebSocket streams, receives live candlestick data, evaluates strategies, validates through risk management, and executes paper trades.

### 7.2 Execution Loop

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚          WebSocket Kline Event          â”‚
â”‚         (every closed candle)           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
       â”Œâ”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”
       â”‚ Update Price   â”‚
       â”‚ Feed to All    â”‚
       â”‚ Sub-Strategies â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
       â”Œâ”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       â”‚ StrategyEngine        â”‚
       â”‚ .evaluateEntry()      â”‚
       â”‚ Returns: signal,      â”‚
       â”‚ confidence, reasoning â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
       â”Œâ”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       â”‚ RiskManager           â”‚
       â”‚ .canOpenPosition()    â”‚
       â”‚ .calculatePosition    â”‚
       â”‚  SizeFixedRisk()      â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
     â”‚  confidence > 0.65 â”‚â”€â”€Noâ”€â”€â–º SKIP
     â”‚  risk validated?   â”‚
     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚ Yes
       â”Œâ”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”
       â”‚ Execute Paper  â”‚
       â”‚ Trade          â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
       â”Œâ”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
       â”‚ Monitor Position      â”‚
       â”‚ â€¢ Check SL/TP         â”‚
       â”‚ â€¢ Evaluate exits      â”‚
       â”‚ â€¢ Trail stops         â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 7.3 Position Management

The engine tracks active positions with full lifecycle management:

```typescript
interface ActivePosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  quantity: number;
  entryTime: number;
  takeProfitPrice: number;
  stopLossPrice: number;
  trailingStop?: number;
}
```

### 7.4 WebSocket Integration

Price updates flow through the `BinanceWebSocket` singleton:
- **Kline streams**: `symbol@kline_interval` for candlestick data
- **Trade streams**: `symbol@trade` for real-time tick prices
- **Multi-ticker**: Combined streams for portfolio-wide monitoring
- **Auto-reconnect**: 3-second reconnection with exponential backoff (max 30s)

---

## 8. Strategy Engine & Algorithms â€” `strategy-engine.ts` (447 lines) + `trading-strategies.ts` (586 lines)

### 8.1 Strategy Hierarchy

```
TradingStrategy (Base Class)
â”œâ”€â”€ MovingAverageCrossoverStrategy
â”œâ”€â”€ MACDStrategy
â”œâ”€â”€ MeanReversionStrategy
â”œâ”€â”€ GridTradingStrategy
â”œâ”€â”€ RSIDivergenceStrategy
â”œâ”€â”€ BollingerBreakoutStrategy
â”œâ”€â”€ VWAPTrendStrategy
â”œâ”€â”€ IchimokuStrategy
â”œâ”€â”€ PivotReversalStrategy
â””â”€â”€ MultiSignalStrategy (Ensemble)
```

### 8.2 Strategy Details

#### 8.2.1 Moving Average Crossover
- **Signal**: Golden Cross (SMA20 > SMA50 > SMA200) â†’ BUY
- **Signal**: Death Cross (SMA20 < SMA50 < SMA200) â†’ SELL
- **Confidence**: 0.85 for full alignment, 0.60 for partial

#### 8.2.2 MACD Strategy
- **Signal**: Histogram flip positive (bullish crossover) â†’ BUY
- **Signal**: Histogram flip negative (bearish crossover) â†’ SELL
- **Divergence**: Increasing histogram momentum amplifies signal
- **Confidence**: 0.75 for crossovers, 0.65 for divergence

#### 8.2.3 Mean Reversion (Bollinger + RSI)
- **BUY**: Price < BB Lower AND RSI < 30 (confidence 0.85)
- **SELL**: Price > BB Upper AND RSI > 70 (confidence 0.85)
- **Extreme**: RSI < 20 â†’ BUY (0.70), RSI > 80 â†’ SELL (0.70)

#### 8.2.4 Grid Trading
- Creates N grid levels around a base price
- Each grid has buy/sell trigger prices spaced by `gridPercentage`
- Fills cascade: buy at lower grid, sell at upper grid
- Optimal for sideways/ranging markets (ADX < 20)

#### 8.2.5 RSI Divergence
- **Bullish**: Price makes Lower Low, RSI makes Higher Low (RSI < 30)
- **Bearish**: Price makes Higher High, RSI makes Lower High (RSI > 70)
- Confidence: 0.85 for confirmed divergence

#### 8.2.6 Bollinger Breakout
- **BUY**: Price breaks above Upper Band (volatility expansion)
- **SELL**: Price breaks below Lower Band
- Confidence: 0.80

#### 8.2.7 VWAP Trend
- **BUY**: Price > VWAP Ã— 1.002 (above with 0.2% buffer)
- **SELL**: Price < VWAP Ã— 0.998
- Institutional bias detection via volume-weighted average price

#### 8.2.8 Ichimoku Cloud
- **Strong BUY**: Price > Cloud Top AND Tenkan > Kijun (confidence 0.90)
- **Strong SELL**: Price < Cloud Bottom AND Tenkan < Kijun (confidence 0.90)
- Requires minimum 52 periods of data

#### 8.2.9 Pivot Point Reversal
- Calculates rolling pivot from 50-candle high/low/close
- **BUY**: Price within 0.2% of Support S1
- **SELL**: Price within 0.2% of Resistance R1

### 8.3 Multi-Signal Ensemble (Voting System)

The `MultiSignalStrategy` aggregates four sub-strategies:
```
Votes = [MA_Crossover, MACD, Mean_Reversion, Grid]

If buyVotes >= 2 AND buyVotes > sellVotes:
    action = BUY
    confidence = min(0.95, 0.5 + buyVotes Ã— 0.2)

If sellVotes >= 2 AND sellVotes > buyVotes:
    action = SELL

Final confidence = (vote_confidence + avg_individual_confidence) / 2
```

### 8.4 Strategy Engine â€” Advanced Features

#### 8.4.1 Top-K Asset Selection
Ranks the asset universe by composite score and selects top K for active trading.

#### 8.4.2 Kelly Criterion Position Sizing
```typescript
kellyFraction = (winRate Ã— avgWin - (1 - winRate) Ã— avgLoss) / avgWin
positionSize = balance Ã— kellyFraction Ã— safetyMultiplier
```

#### 8.4.3 Multi-Condition Exit Logic
Positions are closed when ANY of these trigger:
1. Hard stop-loss hit
2. Take-profit reached
3. Time-in-trade exceeds maximum (prevents stale positions)
4. VaR breach detected
5. Probability drops below threshold on re-evaluation

---

## 9. Risk Management System â€” `risk-management.ts` (394 lines)

### 9.1 Architecture

The `RiskManager` class implements a multi-layered safety system:

```
Layer 1: Position-Level Controls
â”œâ”€â”€ Fixed-risk position sizing
â”œâ”€â”€ ATR-based stop-loss placement
â””â”€â”€ Risk-reward ratio enforcement

Layer 2: Portfolio-Level Controls
â”œâ”€â”€ Maximum concurrent positions
â”œâ”€â”€ Total exposure limits
â”œâ”€â”€ Correlation-based diversification
â””â”€â”€ Maximum allocation per asset

Layer 3: Account-Level Controls (Circuit Breakers)
â”œâ”€â”€ Maximum drawdown limit (10%)
â”œâ”€â”€ Daily loss limit
â”œâ”€â”€ Consecutive loss cooldown
â””â”€â”€ Equity curve monitoring
```

### 9.2 Dynamic Position Sizing ("Gas Pedal")

```typescript
calculatePositionSizeFixedRisk(price: number, stopLoss: number): number {
  const riskAmount = this.accountBalance * this.maxRiskPerTrade;
  const riskPerUnit = Math.abs(price - stopLoss);
  return riskPerUnit > 0 ? riskAmount / riskPerUnit : 0;
}
```

The risk is dynamically scaled by confidence:
```
If confidence > 0.85: riskMultiplier = 1.5x (aggressive)
If confidence 0.65-0.85: riskMultiplier = 1.0x (standard)
If confidence < 0.65: REJECT trade entirely
```

### 9.3 Circuit Breakers

| Breaker | Threshold | Action |
|---|---|---|
| Max Drawdown | 10% from peak | Halt all trading |
| Daily Loss | 5% of balance | Stop opening new positions |
| Consecutive Losses | 3 in a row | 15-minute cooldown |
| Position Count | MAX_POSITIONS | Queue new signals |

### 9.4 Risk-Reward Calculation

```typescript
setRiskReward(entryPrice: number, riskPercent: number, rewardMultiplier: number) {
  const stopLoss = entryPrice * (1 - riskPercent);
  const riskDistance = entryPrice - stopLoss;
  const takeProfit = entryPrice + (riskDistance * rewardMultiplier);
  return { stopLoss, takeProfit };
}
```

### 9.5 Trade History & Analytics

The RiskManager maintains a complete trade journal:
```typescript
interface TradeRecord {
  symbol: string;
  action: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  timestamp: number;
}
```

---

## 10. ML Models Module â€” `ml-models.ts` (451 lines)

### 10.1 Feature Extraction

The frontend ML module mirrors the Python pipeline's feature engineering:

```typescript
extractFeatures(klines: BinanceKline[]): number[][] {
  // Computes: returns, volatility, RSI, MACD,
  // Bollinger width, volume ratio, OBV slope,
  // price momentum, mean reversion distance
}
```

### 10.2 Triple Barrier Labeling (TypeScript Port)

```typescript
applyTripleBarrier(
  klines: BinanceKline[],
  tpPercent: number = 0.015,   // +1.5% take profit
  slPercent: number = -0.01,    // -1.0% stop loss
  timeLimit: number = 24        // 24-bar expiry
): number[]
```

### 10.3 TCN Prediction Simulation

Since the full PyTorch TCN cannot run in the browser, the frontend implements a **simplified TCN-like prediction** using:

1. Feature extraction from the last N candles
2. Weighted combination with exponential decay
3. TDA persistence score integration
4. Probability calibration via sigmoid scaling

### 10.4 TDA Persistence Landscapes

Topological Data Analysis extracts "shape" features from price time series:

```typescript
computePersistenceLandscape(prices: number[]): {
  loops: number;        // H1 Betti numbers (cyclical patterns)
  components: number;   // H0 connected components
  maxPersistence: number; // Longest-lived topological feature
}
```

- **High loop count**: Market is cycling (ranging) â†’ favor mean reversion
- **Low loop count**: Market is trending â†’ favor trend following
- **Persistence voids**: Potential regime change imminent

### 10.5 Ensemble Output

```typescript
interface EnsemblePrediction {
  probability: number;     // 0.0 - 1.0
  confidence: number;      // Calibrated confidence
  regime: string;          // Current market regime
  features: number[];      // Raw feature vector
  tdaScore: number;        // Topological persistence score
}
```

---

## 11. Technical Indicators Engine â€” `technical-indicators.ts` (339 lines)

### 11.1 Indicator Suite

| Indicator | Method | Parameters |
|---|---|---|
| SMA | Simple Moving Average | 20, 50, 200 |
| EMA | Exponential Moving Average | 12, 26 |
| MACD | Moving Average Convergence Divergence | 12/26/9 |
| RSI | Relative Strength Index | 14-period |
| Bollinger Bands | Mean Â± 2Ïƒ | 20-period |
| ATR | Average True Range | 14-period |
| VWAP | Volume-Weighted Average Price | 50-period rolling |
| Ichimoku | Cloud (Tenkan/Kijun/Senkou) | 9/26/52 |
| ADX | Average Directional Index | 14-period (Wilder's smoothing) |

### 11.2 Rolling Window Architecture

```typescript
class TechnicalIndicators {
  private prices: number[] = [];   // Close prices
  private volumes: number[] = [];  // Volume data
  private highs: number[] = [];    // High prices
  private lows: number[] = [];     // Low prices
  private maxSize: number = 500;   // Rolling buffer

  addCandle(close, volume, high, low) {
    // Push new data, shift oldest if > maxSize
  }
}
```

### 11.3 ADX Implementation (Wilder's Method)

The ADX calculation follows J. Welles Wilder's original method:

1. Calculate True Range (TR), +DM, -DM for each bar
2. Apply Wilder's smoothing: `smoothed = prev - (prev/period) + current`
3. Compute +DI and -DI directional indicators
4. Calculate DX = |+DI - -DI| / (+DI + -DI) Ã— 100
5. ADX = Simple average of last N DX values

### 11.4 ATR Fallback

When high/low data equals close (flat candles), ATR falls to zero. The system implements a fallback:
```typescript
if (atr === 0 && prices.length > period) {
  // Use close-to-close absolute changes instead
  return sumAbsChanges / period;
}
```

---

## 12. Market Analyzer â€” `market-analyzer.ts` (169 lines)

### 12.1 Trend Scoring System

The analyzer uses a composite scoring approach (-5 to +5):

| Signal | Bullish (+1) | Bearish (-1) |
|---|---|---|
| SMA 50/200 | SMA50 > SMA200 | SMA50 < SMA200 |
| SMA 20/50 | SMA20 > SMA50 | SMA20 < SMA50 |
| VWAP | Price > VWAP | Price < VWAP |
| MACD | Histogram > 0 | Histogram < 0 |
| Ichimoku | Price > Cloud | Price < Cloud |

**ADX Amplifier**:
- ADX > 25: Score â‰¥ 3 â†’ STRONG_UPTREND, Score â‰¤ -3 â†’ STRONG_DOWNTREND
- ADX < 25: Score â‰¥ 4 â†’ WEAK_UPTREND (early breakout), else SIDEWAYS

### 12.2 Volatility Classification

| State | Condition |
|---|---|
| HIGH | BB Width > 5% OR ATR% > 1.5% |
| NORMAL | Between LOW and HIGH |
| LOW | BB Width < 1.5% OR ATR% < 0.5% |

### 12.3 Sentiment Engine (Market Mood)

Combines RSI and distance-to-SMA50 for mood detection:

| Mood | Condition |
|---|---|
| EXTREME_GREED | RSI > 70 AND price > SMA50 + 5% |
| GREED | RSI > 60 AND price > SMA50 + 2% |
| NEUTRAL | Default |
| FEAR | RSI < 40 AND price < SMA50 - 2% |
| EXTREME_FEAR | RSI < 30 AND price < SMA50 - 5% |

### 12.4 Strategy Selection Matrix

| Condition | Strategy | Confidence |
|---|---|---|
| EXTREME_FEAR | RSI Divergence | 0.90 |
| EXTREME_GREED + Uptrend | Bollinger Breakout | 0.90 |
| Strong Uptrend + High Vol + RSI > 80 | RSI Divergence | 0.75 |
| Strong Uptrend + High Vol | Bollinger Breakout | 0.85 |
| Strong Uptrend + Cloud Bullish | Ichimoku | 0.90 |
| Strong Uptrend + Normal Vol | VWAP Trend | 0.85 |
| Downtrend + RSI < 20 | RSI Divergence | 0.75 |
| Downtrend + High Vol | Bollinger Breakout | 0.85 |
| Sideways + Low Vol | Grid Trading | 0.85 |
| Sideways + High Vol | Mean Reversion | 0.80 |

---

## 13. Supporting Modules

### 13.1 Math Utilities â€” `math-utils.ts` (322 lines)

#### Statistical Functions
- `mean()`, `std()` (with configurable ddof)
- `correlation()` (Pearson correlation coefficient)
- `covarianceMatrix()` (NÃ—N covariance for portfolio analysis)

#### Financial Metrics
- `calculateVaR()` â€” Value at Risk at 95% confidence
- `calculateCVaR()` â€” Conditional VaR (Expected Shortfall)
- `calculateSharpeRatio()` â€” Annualized risk-adjusted returns (âˆš252 scaling)

#### Fractional Differencing (TypeScript Port)
- `fractionalDifference()` â€” Mirrors the Python implementation with d=0.4
- `adfTest()` â€” Simplified Augmented Dickey-Fuller test

#### Regime Detection
```typescript
interface RegimeState {
  volatilityRegime: "low" | "medium" | "high";
  trendRegime: "uptrend" | "downtrend" | "sideways";
  volatilityPercentile: number;  // 0-100
  trendDeviation: number;        // % from SMA
}
```

#### Hierarchical Risk Parity (HRP)
Simplified inverse-volatility weighting for multi-asset allocation:
```typescript
weights[i] = (1 / volatilities[i]) / (n / totalVol)
// Normalized to sum = 1.0
```

### 13.2 Walk-Forward Analyzer â€” `walk-forward-analyzer.ts` (502 lines)

#### Design
Prevents overfitting by splitting data into in-sample (80%) and out-of-sample (20%), then ranking parameter combinations by OOS performance.

#### Strategies Tested
1. Mean Reversion (BB std dev: 1.5 to 3.0)
2. MA Crossover (fast: 5/10/15/20, slow: 30/50/100)
3. RSI Reversal (OB: 65-80, OS: 20-35)
4. MACD (4 configurations)
5. ATR Breakout (multiplier: 1.0 to 3.0)

#### Plateau Scoring
Measures robustness by averaging a result's OOS performance with its neighbors' performance. High plateau score = stable across parameter variations.

#### Composite Ranking
```
Score = OOS_PnL% + PlateauScore Ã— 0.5
// Favors strategies that are BOTH profitable AND stable
```

### 13.3 Agentic Orchestrator â€” `agentic-orchestrator.ts` (227 lines)

> **Note**: Currently marked `@deprecated` â€” not yet integrated into TradingEngine.

#### Regime â†’ Strategy Mapping
| Regime | Strategy | Risk Mult |
|---|---|---|
| BULL_TREND | MA Crossover | 1.2x |
| BEAR_TREND | MACD | 1.0x |
| RANGING | Grid | 1.0x |
| VOLATILE | Mean Reversion | 0.5x |
| BREAKOUT | Multi-Signal | 1.5x |
| UNCERTAIN | Multi-Signal | 0.5x |

#### Anti-Whipsaw Mechanisms
1. **Linger Rule**: 5-minute minimum lock after each strategy switch
2. **Hysteresis**: Requires â‰¥75% confidence to justify switching cost
3. **Penalty Box**: 3 consecutive losses halves confidence; forces safety strategy at <40%

#### Volatility Scaling
- EXTREME_FEAR: Risk Ã—0.7, Take Profit Ã—2.0
- EXTREME_GREED: Risk Ã—1.2, Take Profit Ã—1.5

### 13.4 Gemini AI Advisor â€” `gemini-advisor.ts` (122 lines)

#### Integration
Uses Google Gemini 2.5 Flash via `@google/generative-ai` SDK.

#### Prompt Engineering
Sends structured market data (RSI, ADX, ATR, SMAs, BB Width) and automated analysis to Gemini, requesting:
1. Strategy confirmation
2. Risk identification
3. Actionable recommendation (< 100 words)

#### Guardrails
- **Price target stripping**: Removes hallucinated specific price predictions
- **Probability claim removal**: Strips "90% chance" type statements
- **Action validation**: Ensures output contains BUY/SELL/HOLD/WAIT
- **Truncation**: Caps response at 800 characters
- **Disclaimer prefix**: Always prepends "Not Financial Advice" warning

#### Fallback
When the API is unavailable, generates local rule-based advice using trend + RSI + ADX.

### 13.5 Backtester â€” `backtester.ts` (301 lines)

#### Features
- Full simulation with Multi-Signal strategy
- Slippage modeling: 0.05% on entry and exit
- Commission: 0.1% per trade
- Portfolio metrics: Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate
- Balance history tracking for equity curve analysis

#### Trade Statistics
```typescript
{
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
}
```

### 13.6 Binance Client â€” `binance-client.ts` (423 lines)

Full-featured authenticated Binance API client:
- **HMAC-SHA256 signing** for authenticated endpoints
- **LOT_SIZE and TICK_SIZE rounding** via exchange info cache
- **Order types**: Market, Limit, Stop Loss, Take Profit
- **WebSocket subscriptions** with exponential backoff reconnection
- **Testnet support** via constructor flag

### 13.7 Binance WebSocket â€” `binance-websocket.ts` (431 lines)

Singleton WebSocket manager with:
- Kline subscription per symbol/interval
- Trade stream for real-time prices
- Multi-ticker aggregation for portfolio view
- CoinGecko fallback with Binance history as secondary fallback
- Moving average pre-computation (MA7, MA25, MA99)

---

*Continued in Part 3...*
# AI Institutional Trading Bot â€” Project Report (Part 3)

## Table of Contents â€” Part 3

14. [Frontend Architecture & UI](#14-frontend-architecture--ui)
15. [API Routes](#15-api-routes)
16. [Configuration & Deployment](#16-configuration--deployment)
17. [Mathematical Foundations](#17-mathematical-foundations)
18. [Code Metrics & Quality](#18-code-metrics--quality)
19. [Known Issues & Technical Debt](#19-known-issues--technical-debt)
20. [Future Roadmap](#20-future-roadmap)
21. [Appendix A: Complete File Inventory](#21-appendix-a-complete-file-inventory)
22. [Appendix B: Algorithm Reference](#22-appendix-b-algorithm-reference)
23. [Appendix C: Glossary](#23-appendix-c-glossary)

---

## 14. Frontend Architecture & UI

### 14.1 App Router Structure

```
app/
â”œâ”€â”€ page.tsx              # Landing page (hero + feature cards)
â”œâ”€â”€ layout.tsx            # Root layout with theme provider
â”œâ”€â”€ globals.css           # Global styles
â”œâ”€â”€ dashboard/            # Main trading dashboard
â”œâ”€â”€ trade/                # Active trading interface
â”œâ”€â”€ coins/                # Coin listing & search
â”œâ”€â”€ crypto/               # Individual crypto detail pages
â”œâ”€â”€ settings/             # Bot configuration panel
â””â”€â”€ api/                  # Next.js API routes
    â””â”€â”€ coins/[id]/       # CoinGecko proxy endpoints
```

### 14.2 Component Architecture

| Component | File | Lines | Purpose |
|---|---|---|---|
| `BinanceTradingDashboard` | `binance-trading-dashboard.tsx` | ~4,200 | Main institutional dashboard |
| `MLAnalytics` | `ml-analytics.tsx` | 416 | ML model performance panel |
| `StrategyConfigurator` | `strategy-configurator.tsx` | 408 | Strategy parameter tuning |
| `Dashboard` | `dashboard.tsx` | 256 | Portfolio overview |
| `FinancialChart` | `financial-chart.tsx` | 285 | TradingView-style charts |
| `GlobalTradeConfig` | `global-trade-config.tsx` | 239 | System-wide trade settings |
| `PriceChartPro` | `price-chart-pro.tsx` | 236 | Advanced charting with MAs |
| `PriceChartCrypto` | `price-chart-crypto.tsx` | 206 | Crypto-specific chart |
| `BotSettings` | `bot-settings.tsx` | 208 | Bot parameter controls |
| `CoinTable` | `coin-table.tsx` | 183 | Asset listing with stats |
| `TradingInterface` | `trading-interface.tsx` | 137 | Order entry panel |
| `CryptoSearch` | `crypto-search.tsx` | 135 | Fuzzy search across coins |
| `TradingPairSelector` | `trading-pair-selector.tsx` | 126 | Pair selection dropdown |
| `AppHeader` | `app-header.tsx` | 67 | Navigation header |
| `TradeHistory` | `trade-history.tsx` | 60 | Recent trade log |

### 14.3 Landing Page Design

The landing page (`app/page.tsx`, 203 lines) features:
- **Animated SVG background**: An upward-trending graph with `stroke-dasharray` animation
- **Ambient glow effects**: CSS blur with brand color (#1DB954)
- **Feature grid**: 6 cards highlighting TDA, TCN, TBM, HRP, FracDiff, Agentic Orchestration
- **Research Core section**: Team member attribution
- **Engine Hardening metrics**: Status indicators and stat boxes

### 14.4 Design System

| Property | Value |
|---|---|
| Background | `#121212` (deep black) |
| Surface | `#1A1A1A` (card background) |
| Brand/Accent | `#1DB954` (Spotify green) |
| Text Primary | `#EAEAEA` |
| Text Secondary | `gray-400` / `gray-500` |
| Border | `white/5` (5% white opacity) |
| Corner Radius | `2xl` (rounded-2xl = 16px) |
| Font | System sans-serif stack |

### 14.5 Charting Library

The system uses **Lightweight Charts v5.1.0** (TradingView's open-source library) for:
- Candlestick rendering
- Multi-series overlays (MA7, MA25, MA99)
- Volume histogram
- Real-time candle updates via WebSocket

**Recharts 2.15.4** is used for supplementary visualizations:
- Equity curves
- PnL distribution
- Win rate bar charts
- Strategy performance comparisons

### 14.6 UI Component Library

Built on **Radix UI** primitives for accessibility:
- Accordion, Alert Dialog, Avatar, Checkbox
- Collapsible, Context Menu, Dialog, Dropdown
- Hover Card, Label, Menubar, Navigation Menu
- Popover, Progress, Radio Group, Scroll Area
- Select, Separator, Slider, Switch, Tabs
- Toast (via Sonner), Toggle, Tooltip

Styling via:
- **Tailwind CSS 3.4.17** with `tailwind-merge` for class deduplication
- **class-variance-authority** for component variant management
- **tailwindcss-animate** for animation utilities

---

## 15. API Routes

### 15.1 Next.js API Routes

```
app/api/
â””â”€â”€ coins/
    â””â”€â”€ [id]/
        â””â”€â”€ history/
            â””â”€â”€ route.ts    # GET /api/coins/:id/history?days=max
```

This route proxies CoinGecko's market chart API to avoid CORS issues:
```typescript
// Fetches from: https://api.coingecko.com/api/v3/coins/{id}/market_chart
// Returns: { data: [{ timestamp, price }] }
```

### 15.2 Python API Endpoints

```
FastAPI Server (api.py)
â”œâ”€â”€ POST /predict          # ML inference endpoint
â”‚   Request:  { symbol: "BTCUSDT" }
â”‚   Response: { probability, signal, confidence, timestamp }
â”‚
â”œâ”€â”€ GET /health            # Service health check
â”‚   Response: { status: "ok", model_loaded: true }
â”‚
â””â”€â”€ CORS middleware enabled for frontend communication
```

---

## 16. Configuration & Deployment

### 16.1 Environment Variables

```env
# .env.example
NEXT_PUBLIC_GEMINI_API_KEY="your-gemini-api-key-here"

# Additional required (not in .env.example):
BINANCE_API_KEY="..."          # For authenticated endpoints
BINANCE_API_SECRET="..."       # HMAC signing
```

### 16.2 Package Configuration

```json
{
  "name": "my-v0-project",
  "version": "0.1.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint ."
  }
}
```

> **Note**: The package name `my-v0-project` is boilerplate from v0.dev scaffolding. Should be renamed to reflect the project identity.

### 16.3 Python Dependencies

```
torch>=2.0.0          # TCN neural network
scikit-learn>=1.3.0   # Calibration, preprocessing
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical operations
optuna                # Hyperparameter optimization
xgboost               # Gradient boosted trees
fastapi               # REST API server
uvicorn               # ASGI server
ccxt                  # Exchange abstraction
statsmodels           # ADF stationarity test
```

### 16.4 Development Workflow

```bash
# Frontend
npm install
npm run dev          # â†’ http://localhost:3000

# Backend
pip install -r requirements.txt
python build_dataset.py    # Generate training data
python tbm_labeler.py      # Apply Triple Barrier labels
python train_tcn.py        # Train TCN + XGBoost
uvicorn api:app --reload   # Start inference API
```

---

## 17. Mathematical Foundations

### 17.1 Fractional Differencing

**Problem**: Raw prices are non-stationary (unit root). Standard differencing (d=1) makes them stationary but destroys predictive memory.

**Solution**: Use fractional d âˆˆ (0, 1) to balance stationarity and memory preservation.

**Weight Formula**:
```
wâ‚€ = 1
wâ‚– = -wâ‚–â‚‹â‚ Ã— (d - k + 1) / k    for k â‰¥ 1
```

**Application**:
```
Xáµˆâ‚œ = Î£áµ¢ wáµ¢ Ã— Xâ‚œâ‚‹áµ¢    (convolution with fractional weights)
```

**Optimal d = 0.4**: Achieves ADF p-value < 0.05 while retaining ~60% of price memory.

### 17.2 Triple Barrier Method

Three simultaneous barriers create a labeling scheme that models real trading outcomes:

```
                    â”Œâ”€â”€â”€ Upper Barrier (TP: +1.5%) â”€â”€â”€ Label = 1 (WIN)
                    â”‚
Entry Price â”€â”€â”€â”€â”€â”€â”€â”€â”¤
                    â”‚
                    â”œâ”€â”€â”€ Lower Barrier (SL: -1.0%) â”€â”€â”€ Label = 0 (LOSS)
                    â”‚
                    â””â”€â”€â”€ Vertical Barrier (24 bars) â”€â”€ Label = 0 (TIME DECAY)
```

The first barrier touched determines the label. This is superior to fixed-threshold labeling because it accounts for holding period and asymmetric risk-reward.

### 17.3 Temporal Convolutional Networks (TCN)

**Key Properties**:
- **Causal convolutions**: Output at time t depends only on inputs at time â‰¤ t
- **Dilated convolutions**: Exponentially increasing receptive field (1, 2, 4, 8...)
- **Residual connections**: Gradient flow through skip connections
- **Parallelizable**: Unlike RNNs, entire sequence processes in parallel

**Receptive Field**: For L layers, kernel size k, and dilation factor d:
```
RF = 1 + Î£áµ¢ (k-1) Ã— dáµ¢ = 1 + (k-1) Ã— (2á´¸ - 1)
```

With 3 layers, k=3: RF = 1 + 2 Ã— 7 = **15 timesteps**

### 17.4 Platt Scaling

Post-hoc calibration that maps raw model outputs to true probabilities:

```
P(y=1|f(x)) = 1 / (1 + exp(A Ã— f(x) + B))
```

Parameters A and B are fit via logistic regression on a held-out calibration set. This ensures that when the model outputs P=0.7, approximately 70% of those predictions are actually wins.

### 17.5 Value at Risk (VaR)

**Historical VaR at 95% confidence**:
```
VaRâ‚‰â‚… = -Percentileâ‚…(returns)
```

Interpretation: "There is a 5% chance of losing more than VaRâ‚‰â‚… in a single period."

**Conditional VaR (CVaR)** = Expected loss given that loss exceeds VaR:
```
CVaRâ‚‰â‚… = -E[returns | returns < -VaRâ‚‰â‚…]
```

### 17.6 Kelly Criterion

Optimal bet sizing for geometric growth:
```
f* = (p Ã— b - q) / b

where:
  p = win probability
  q = 1 - p (loss probability)
  b = average win / average loss
```

The system applies a **safety multiplier** (typically 0.25-0.5) to the raw Kelly fraction to reduce variance.

### 17.7 Sharpe Ratio

Annualized risk-adjusted return:
```
Sharpe = (E[R] - Rf) / Ïƒ(R) Ã— âˆš252

where:
  E[R] = expected daily return
  Rf   = risk-free rate (0.02 annual)
  Ïƒ(R) = standard deviation of daily returns
  252  = trading days per year
```

### 17.8 Hierarchical Risk Parity (HRP)

Simplified inverse-volatility allocation:
```
wáµ¢ = (1/Ïƒáµ¢) / Î£â±¼ (1/Ïƒâ±¼)
```

Assets with lower volatility receive larger allocations, promoting diversification without requiring covariance matrix inversion.

---

## 18. Code Metrics & Quality

### 18.1 Lines of Code by Module

| Module | File | Lines | Language |
|---|---|---|---|
| AI Config Engine | `lib/ai-config-engine.ts` | 1,336 | TypeScript |
| Trading Strategies | `lib/trading-strategies.ts` | 586 | TypeScript |
| Walk-Forward Analyzer | `lib/walk-forward-analyzer.ts` | 502 | TypeScript |
| ML Models | `lib/ml-models.ts` | 451 | TypeScript |
| Strategy Engine | `lib/strategy-engine.ts` | 447 | TypeScript |
| Binance WebSocket | `lib/binance-websocket.ts` | 431 | TypeScript |
| Binance Client | `lib/binance-client.ts` | 423 | TypeScript |
| Risk Management | `lib/risk-management.ts` | 394 | TypeScript |
| Trading Engine | `lib/trading-engine.ts` | 393 | TypeScript |
| Technical Indicators | `lib/technical-indicators.ts` | 339 | TypeScript |
| Math Utilities | `lib/math-utils.ts` | 322 | TypeScript |
| Backtester | `lib/backtester.ts` | 301 | TypeScript |
| Train TCN | `train_tcn.py` | 289 | Python |
| Build Dataset | `build_dataset.py` | 241 | Python |
| Agentic Orchestrator | `lib/agentic-orchestrator.ts` | 227 | TypeScript |
| Landing Page | `app/page.tsx` | 203 | TSX |
| API Server | `api.py` | 202 | Python |
| Market Analyzer | `lib/market-analyzer.ts` | 169 | TypeScript |
| Frac Diff | `frac_diff.py` | 134 | Python |
| Gemini Advisor | `lib/gemini-advisor.ts` | 122 | TypeScript |
| TBM Labeler | `tbm_labeler.py` | 92 | Python |
| **TOTAL (lib/ only)** | | **~5,900** | TypeScript |
| **TOTAL (Python)** | | **~960** | Python |
| **GRAND TOTAL** | | **~7,600+** | Mixed |

### 18.2 Complexity Analysis

| Metric | Value |
|---|---|
| Distinct Trading Strategies | 9 + 1 ensemble |
| Technical Indicators | 15 unique |
| Market Regimes Classified | 6 |
| Risk Safety Layers | 3 (position, portfolio, account) |
| API Integrations | 4 (Binance REST, Binance WS, CoinGecko, Gemini) |
| Data Flow Fallbacks | 3-tier (Primary â†’ CoinGecko â†’ Mock) |

### 18.3 Type Safety

The codebase is fully typed in TypeScript with explicit interfaces for all major data structures:
- `BinanceKline`, `BinanceTicker`, `IndicatorData`
- `StrategySignal`, `Position`, `TradeRecord`
- `BacktestResult`, `WFAReport`, `EnsemblePrediction`
- `MarketRegime`, `StrategicDecision`, `TradingConfig`

---

## 19. Known Issues & Technical Debt

### 19.1 Critical

| ID | Issue | Impact | Mitigation |
|---|---|---|---|
| TD-1 | Agentic Orchestrator is `@deprecated` and not wired into TradingEngine | Strategy selection logic is not applied in production | Wire `decide()` into the main trading loop |
| TD-2 | Package name is `my-v0-project` | Unprofessional; confusing in deployment | Rename in `package.json` |
| TD-3 | Model file dependency (`institutional_xgboost_model.json`) not guaranteed | API falls back to HOLD if file missing | Add model versioning and S3/GCS storage |

### 19.2 Moderate

| ID | Issue | Impact | Mitigation |
|---|---|---|---|
| TD-4 | `requirements.txt` lists `scikit-learn` twice | Redundant dependency | Remove duplicate |
| TD-5 | Forward-fill on OI/funding rate may mask data gaps | Potential feature leakage | Add gap detection alerts |
| TD-6 | TCN runs only in Python; frontend uses simplified simulation | Prediction quality lower on frontend | Add WebSocket bridge to Python backend |
| TD-7 | No authentication on `/predict` endpoint | Security risk in production | Add API key middleware |
| TD-8 | `.env.example` only lists Gemini key | Missing Binance credentials documentation | Complete `.env.example` |

### 19.3 Low Priority

| ID | Issue | Impact |
|---|---|---|
| TD-9 | No unit tests | Regression risk |
| TD-10 | No CI/CD pipeline | Manual deployment |
| TD-11 | Binance WebSocket uses global error suppression | May hide connection issues |
| TD-12 | Some Radix UI components imported but unused | Bundle size |

---

## 20. Future Roadmap

### Phase 1: Production Hardening
- [ ] Wire Agentic Orchestrator into TradingEngine
- [ ] Add API authentication (JWT or API key)
- [ ] Implement WebSocket bridge for TCN inference
- [ ] Complete environment variable documentation
- [ ] Add unit and integration tests

### Phase 2: ML Pipeline Enhancement
- [ ] Integrate Optuna hyperparameter tuning into training pipeline
- [ ] Add walk-forward cross-validation to Python trainer
- [ ] Implement online learning for model updates
- [ ] Add feature importance visualization

### Phase 3: Dashboard Enhancement
- [ ] Integrate ADX, VWMA, FracDiff metrics into dashboard
- [ ] Add real-time equity curve charting
- [ ] Implement trade journal with export functionality
- [ ] Add mobile-responsive trading interface

### Phase 4: Live Trading
- [ ] Paper trading â†’ Live transition with safety switches
- [ ] Multi-exchange support (Bybit, OKX via ccxt)
- [ ] Portfolio rebalancing automation
- [ ] Telegram/Discord alerting integration

---

## 21. Appendix A: Complete File Inventory

### Root Directory
| File | Purpose |
|---|---|
| `api.py` | FastAPI prediction server |
| `build_dataset.py` | Feature engineering pipeline |
| `train_tcn.py` | TCN model training |
| `frac_diff.py` | Fractional differencing |
| `tbm_labeler.py` | Triple Barrier labeling |
| `requirements.txt` | Python dependencies |
| `package.json` | Node.js dependencies |
| `.env.example` | Environment template |
| `TODO.md` | Task tracking |

### lib/ Directory
| File | Lines | Primary Export |
|---|---|---|
| `ai-config-engine.ts` | 1,336 | `AIConfigEngine` class |
| `trading-strategies.ts` | 586 | 9 strategy classes + `MultiSignalStrategy` |
| `walk-forward-analyzer.ts` | 502 | `WalkForwardAnalyzer` class |
| `ml-models.ts` | 451 | Feature extraction + TDA + ensemble |
| `strategy-engine.ts` | 447 | `StrategyEngine` class |
| `binance-websocket.ts` | 431 | `binanceWS` singleton |
| `binance-client.ts` | 423 | `BinanceClient` class |
| `risk-management.ts` | 394 | `RiskManager` class |
| `trading-engine.ts` | 393 | `TradingEngine` class |
| `technical-indicators.ts` | 339 | `TechnicalIndicators` class |
| `math-utils.ts` | 322 | Statistical & financial utilities |
| `backtester.ts` | 301 | `Backtester` class |
| `agentic-orchestrator.ts` | 227 | `AgenticOrchestrator` class |
| `market-analyzer.ts` | 169 | `MarketAnalyzer` class |
| `gemini-advisor.ts` | 122 | `GeminiAdvisor` class |

### components/ Directory
| File | Size | Purpose |
|---|---|---|
| `binance-trading-dashboard.tsx` | 157KB | Full institutional dashboard |
| `ml-analytics.tsx` | 15KB | ML performance panel |
| `strategy-configurator.tsx` | 15KB | Strategy parameter UI |
| `financial-chart.tsx` | 11KB | TradingView charts |
| `dashboard.tsx` | 10KB | Portfolio overview |
| `global-trade-config.tsx` | 9KB | System-wide settings |
| `price-chart-pro.tsx` | 9KB | Advanced charting |
| `bot-settings.tsx` | 8KB | Bot controls |
| `price-chart-crypto.tsx` | 8KB | Crypto charts |
| `coin-table.tsx` | 7KB | Asset listing |
| `trading-interface.tsx` | 5KB | Order entry |
| `crypto-search.tsx` | 5KB | Search |
| `trading-pair-selector.tsx` | 5KB | Pair selector |
| `app-header.tsx` | 3KB | Navigation |
| `trade-history.tsx` | 2KB | Trade log |
| `price-chart.tsx` | 1KB | Simple chart |
| `theme-provider.tsx` | 0.3KB | Dark/light theme |
| `ui/` | dir | Radix UI primitives |

---

## 22. Appendix B: Algorithm Reference

### B.1 RSI (Relative Strength Index)

```
RSI = 100 - 100 / (1 + RS)
RS  = Average Gain / Average Loss    (over N periods)
```

### B.2 MACD (Moving Average Convergence Divergence)

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

### B.3 Bollinger Bands

```
Middle = SMA(20)
Upper  = SMA(20) + 2 Ã— Ïƒ(20)
Lower  = SMA(20) - 2 Ã— Ïƒ(20)
Width  = (Upper - Lower) / Middle
```

### B.4 ATR (Average True Range)

```
TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
ATR = SMA(TR, 14)
```

### B.5 ADX (Average Directional Index)

```
+DM = High - PrevHigh    (if positive and > -DM, else 0)
-DM = PrevLow - Low      (if positive and > +DM, else 0)

Smoothed via Wilder's method:
  Smooth = Prev - (Prev / N) + Current

+DI = (Smooth +DM / Smooth TR) Ã— 100
-DI = (Smooth -DM / Smooth TR) Ã— 100

DX  = |+DI - -DI| / (+DI + -DI) Ã— 100
ADX = SMA(DX, 14)
```

### B.6 Ichimoku Cloud

```
Tenkan-sen  = (Highest High(9) + Lowest Low(9)) / 2
Kijun-sen   = (Highest High(26) + Lowest Low(26)) / 2
Senkou A    = (Tenkan + Kijun) / 2
Senkou B    = (Highest High(52) + Lowest Low(52)) / 2
```

### B.7 VWAP (Volume-Weighted Average Price)

```
VWAP = Î£(Typical Price Ã— Volume) / Î£(Volume)
Typical Price = (High + Low + Close) / 3
```

---

## 23. Appendix C: Glossary

| Term | Definition |
|---|---|
| **ADF Test** | Augmented Dickey-Fuller test for stationarity |
| **ATR** | Average True Range â€” volatility measure |
| **CVaR** | Conditional Value at Risk (Expected Shortfall) |
| **EMA** | Exponential Moving Average |
| **FFD** | Fixed-width Window Fractional Differencing |
| **FVG** | Fair Value Gap â€” institutional price imbalance |
| **HRP** | Hierarchical Risk Parity â€” portfolio allocation |
| **Kelly Criterion** | Optimal bet sizing formula |
| **MACD** | Moving Average Convergence Divergence |
| **OI** | Open Interest â€” total outstanding futures contracts |
| **Platt Scaling** | Post-hoc probability calibration |
| **RSI** | Relative Strength Index |
| **SMA** | Simple Moving Average |
| **SMC** | Smart Money Concepts â€” institutional trading patterns |
| **TBM** | Triple Barrier Method â€” ML trade labeling |
| **TCN** | Temporal Convolutional Network |
| **TDA** | Topological Data Analysis |
| **VaR** | Value at Risk â€” maximum expected loss at confidence level |
| **VWAP** | Volume-Weighted Average Price |
| **WFA** | Walk-Forward Analysis â€” overfitting prevention |

---

> **End of Report**  
> Total coverage: 23 sections across 3 parts  
> Generated from live codebase analysis â€” May 2026
