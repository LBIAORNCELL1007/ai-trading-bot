/**
 * Machine Learning Models for Trading
 * Includes TCN, TDA, and prediction engines
 */

import {
  calculateTechnicalIndicators,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  detectRegime,
  mean,
  std,
} from "./math-utils"
import { calibrateProbabilities } from "./strategy-engine"

// ===== FEATURE ENGINEERING =====

export interface TradeFeatures {
  price: number
  returns: number
  volatility: number
  rsi: number
  macd: number
  bollingerPosition: number
  trendStrength: number
  volumeProfile: number
  priceLevel: number // percentile of price range
  regimeIndicator: number
}

export function extractFeatures(prices: number[], volumes: number[] = []): TradeFeatures {
  if (prices.length === 0) {
    return {
      price: 0,
      returns: 0,
      volatility: 0,
      rsi: 50,
      macd: 0,
      bollingerPosition: 0.5,
      trendStrength: 0,
      volumeProfile: 0.5,
      priceLevel: 0.5,
      regimeIndicator: 0,
    }
  }

  const currentPrice = prices[prices.length - 1]
  const previousPrice = prices.length > 1 ? prices[prices.length - 2] : currentPrice
  const returns = previousPrice !== 0 ? (currentPrice - previousPrice) / previousPrice : 0

  // Calculate volatility
  const recentPrices = prices.slice(-20)
  const recentReturns = recentPrices.slice(1).map((p, i) => (p - recentPrices[i]) / recentPrices[i])
  const volatility = std(recentReturns)

  // Technical indicators
  const indicators = calculateTechnicalIndicators(prices)
  const rsi = indicators.rsi
  const macd = indicators.macd.histogram
  const bb = indicators.bollingerBands

  // Bollinger Band position (0-1, where 0.5 is middle)
  const bbRange = bb.upper - bb.lower
  const bollingerPosition =
    bbRange > 0 ? (currentPrice - bb.lower) / bbRange : 0.5

  // Trend strength (based on MACD)
  const trendStrength = Math.tanh(Math.abs(indicators.macd.histogram) * 10)

  // Volume profile (simplified)
  const volumeProfile = volumes.length > 0 ? mean(volumes.slice(-20)) / (Math.max(...volumes) || 1) : 0.5

  // Price level (percentile in recent range)
  const minPrice = Math.min(...recentPrices)
  const maxPrice = Math.max(...recentPrices)
  const priceRange = maxPrice - minPrice
  const priceLevel =
    priceRange > 0 ? (currentPrice - minPrice) / priceRange : 0.5

  // Regime indicator (0-1)
  const regime = detectRegime(prices)
  const regimeIndicator =
    (regime.volatilityPercentile / 100) * 0.5 + (regime.trendDeviation > 0 ? 0.5 : 0)

  return {
    price: currentPrice,
    returns,
    volatility,
    rsi: rsi / 100, // Normalize to 0-1
    macd: Math.tanh(macd), // Normalize
    bollingerPosition,
    trendStrength,
    volumeProfile,
    priceLevel,
    regimeIndicator,
  }
}

// ===== TRIPLE BARRIER METHOD =====

export interface TripleBarrierLabels {
  labels: Array<0 | 1 | -1> // 0 = neutral, 1 = profit, -1 = loss
  barriers: Array<{
    upper: number
    lower: number
    vertical: number
  }>
  times: number[]
}

export function tripleBarrierLabeling(
  prices: number[],
  profitTarget = 0.03, // 3% take profit
  stopLoss = 0.02, // 2% stop loss
  timeHorizon = 20 // 20 periods max hold
): TripleBarrierLabels {
  const labels: Array<0 | 1 | -1> = []
  const barriers: Array<{
    upper: number
    lower: number
    vertical: number
  }> = []
  const times: number[] = []

  for (let i = 0; i < prices.length - timeHorizon; i++) {
    const entryPrice = prices[i]
    const upper = entryPrice * (1 + profitTarget)
    const lower = entryPrice * (1 - stopLoss)
    const vertical = i + timeHorizon

    barriers.push({ upper, lower, vertical })

    // Find which barrier is hit first
    let label: 0 | 1 | -1 = 0
    let hitTime = vertical

    for (let j = i + 1; j <= Math.min(i + timeHorizon, prices.length - 1); j++) {
      if (prices[j] >= upper) {
        label = 1 // Profit
        hitTime = j
        break
      } else if (prices[j] <= lower) {
        label = -1 // Loss
        hitTime = j
        break
      }
    }

    // Neutral if time barrier hit
    if (label === 0) {
      const endIdx = Math.min(vertical, prices.length - 1)
      label = prices[endIdx] > entryPrice ? 1 : -1
    }

    labels.push(label)
    times.push(hitTime - i)
  }

  return { labels, barriers, times }
}

// ===== TEMPORAL CONVOLUTIONAL NETWORK (TCN) SIMPLIFIED =====

export class SimpleTCN {
  private weights: number[][] = []
  private biases: number[] = []
  private history: number[] = []

  constructor(inputSize: number, outputSize: number) {
    // Initialize random weights
    this.weights = Array(outputSize)
      .fill(null)
      .map(() =>
        Array(inputSize)
          .fill(null)
          .map(() => (Math.random() - 0.5) * 2)
      )
    this.biases = Array(outputSize)
      .fill(null)
      .map(() => Math.random() - 0.5)
  }

  /**
   * Dilated causal convolution
   * Dilation expands receptive field exponentially
   */
  private dilation(input: number[], dilation: number): number[] {
    const dilated: number[] = []
    for (let i = 0; i < input.length; i++) {
      if (i % dilation === 0) {
        dilated.push(input[i])
      }
    }
    return dilated
  }

  /**
   * ReLU activation
   */
  private relu(x: number): number {
    return Math.max(0, x)
  }

  /**
   * Sigmoid activation
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500)))
  }

  /**
   * Forward pass
   */
  forward(input: number[]): number[] {
    const output: number[] = []

    for (let i = 0; i < this.weights.length; i++) {
      let sum = this.biases[i]
      for (let j = 0; j < Math.min(input.length, this.weights[i].length); j++) {
        sum += input[j] * this.weights[i][j]
      }
      output.push(this.sigmoid(sum))
    }

    return output
  }

  /**
   * Train on historical data (simplified gradient descent)
   */
  train(inputs: number[][], targets: number[], epochs = 10, learningRate = 0.01) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < inputs.length; i++) {
        const output = this.forward(inputs[i])

        // Update all output neurons
        for (let k = 0; k < this.weights.length; k++) {
          const error = targets[i] - output[k]
          for (let j = 0; j < this.weights[k].length; j++) {
            this.weights[k][j] += learningRate * error * inputs[i][j]
          }
          this.biases[k] += learningRate * error
        }
      }
    }
  }

  /**
   * Predict probability of upward movement
   */
  predict(features: number[]): number {
    const output = this.forward(features)
    return output[0]
  }
}

// ===== TOPOLOGICAL DATA ANALYSIS (TDA) SIMPLIFIED =====

export interface PersistenceLandscape {
  landscape: number[]
  entropy: number
  cyclicStrength: number
}

export function calculatePersistenceLandscape(prices: number[]): PersistenceLandscape {
  if (prices.length < 10) {
    return {
      landscape: Array(100).fill(0),
      entropy: 0,
      cyclicStrength: 0,
    }
  }

  // Simplified: use price differences as persistence features
  const diffs = prices.slice(1).map((p, i) => Math.abs(p - prices[i]))
  const landscape: number[] = []

  // Create 100-point landscape
  for (let i = 0; i < 100; i++) {
    const idx = Math.floor((i / 100) * diffs.length)
    landscape.push(diffs[idx] || 0)
  }

  // Calculate entropy (measure of disorder)
  const normalized = landscape.map((l) => l / (Math.max(...landscape) || 1))
  const entropy = -normalized.reduce((sum, p) => {
    return sum + (p > 0 ? p * Math.log(p) : 0)
  }, 0)

  // Calculate cyclic strength (persistent homology H1)
  let cyclicStrength = 0
  for (let i = 0; i < prices.length - 2; i++) {
    const angle1 = Math.atan2(prices[i + 1] - prices[i], i + 1 - i)
    const angle2 = Math.atan2(prices[i + 2] - prices[i + 1], i + 2 - (i + 1))
    const angleDiff = Math.abs(angle1 - angle2)
    cyclicStrength += angleDiff
  }
  cyclicStrength /= prices.length

  return { landscape, entropy, cyclicStrength }
}

// ===== ENSEMBLE PREDICTION =====

export interface ModelPrediction {
  probability: number // 0-1, probability of upward move
  confidence: number // 0-1, how confident we are
  signals: {
    tcn: number
    tda: number
    technicalIndicators: number
  }
}

export function predictDirection(
  prices: number[],
  volumes: number[] = [],
  tcnModel?: SimpleTCN,
  /**
   * Optional probability calibrator (isotonic-regression mapping fitted on
   * a held-out validation set via `calibrateProbabilities`).  If provided,
   * the ensemble's raw probability is mapped to an empirically-calibrated
   * win-probability before being returned.  Without calibration, `probability`
   * is the *raw model output* and should NEVER be compared against fixed
   * thresholds like 0.55 — see `optimizeThreshold` for data-driven tuning.
   */
  calibrator?: (p: number) => number
): ModelPrediction {
  if (prices.length < 10) {
    return {
      probability: 0.5,
      confidence: 0,
      signals: { tcn: 0.5, tda: 0.5, technicalIndicators: 0.5 },
    }
  }

  // TCN prediction
  const features = extractFeatures(prices, volumes)
  const featureVector = [
    features.rsi,
    features.macd,
    features.bollingerPosition,
    features.trendStrength,
    features.volatility,
    features.regimeIndicator,
  ]

  const tcnPred = tcnModel
    ? tcnModel.predict(featureVector)
    : 0.5 + (features.rsi - 0.5) * 0.3 // Fallback to simple RSI signal

  // TDA prediction
  const tdaAnalysis = calculatePersistenceLandscape(prices)
  const tdaPred = Math.min(1, tdaAnalysis.entropy / 2) // Higher entropy = more volatile = higher chance of reversal

  // Technical indicators signal
  const rsiSignal = features.rsi > 0.7 ? 0.3 : features.rsi < 0.3 ? 0.7 : 0.5
  const macdSignal = features.macd > 0 ? 0.6 : 0.4
  const bbSignal = features.bollingerPosition > 0.7 ? 0.3 : features.bollingerPosition < 0.3 ? 0.7 : 0.5
  const techSignal = (rsiSignal + macdSignal + bbSignal) / 3

  // Ensemble: weighted average
  const rawProbability = tcnPred * 0.4 + tdaPred * 0.3 + techSignal * 0.3
  const probability = calibrator
    ? Math.min(1, Math.max(0, calibrator(Math.min(1, Math.max(0, rawProbability)))))
    : Math.min(1, Math.max(0, rawProbability))
  const confidence = Math.min(
    1,
    Math.abs(probability - 0.5) * 2 +
      (features.volatility < 0.05 ? 0.2 : 0) // Higher confidence in low vol
  )

  return {
    probability,
    confidence,
    signals: {
      tcn: tcnPred,
      tda: tdaPred,
      technicalIndicators: techSignal,
    },
  }
}

// ===== BACKTESTING =====

export interface BacktestResult {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  totalTrades: number
}

export function backtest(
  prices: number[],
  predictions: ModelPrediction[],
  threshold = 0.55
): BacktestResult {
  let cash = 10000
  let position = 0
  const trades: Array<{ type: "buy" | "sell"; price: number; time: number }> = []
  const equity: number[] = [10000]

  for (let i = 0; i < Math.min(prices.length, predictions.length); i++) {
    const currentPrice = prices[i]
    const prediction = predictions[i]

    if (prediction.probability > threshold && position === 0) {
      // Buy signal
      position = cash / currentPrice
      cash = 0
      trades.push({ type: "buy", price: currentPrice, time: i })
    } else if (prediction.probability < 1 - threshold && position > 0) {
      // Sell signal
      cash = position * currentPrice
      position = 0
      trades.push({ type: "sell", price: currentPrice, time: i })
    }

    const currentEquity = cash + position * currentPrice
    equity.push(currentEquity)
  }

  // Close position at end
  if (position > 0) {
    cash = position * prices[prices.length - 1]
  }
  const finalEquity = cash

  // Calculate metrics
  const totalReturn = (finalEquity - 10000) / 10000
  const returns = equity.slice(1).map((e, i) => (e - equity[i]) / equity[i])
  const avgReturn = mean(returns) * 252
  const stdReturn = std(returns) * Math.sqrt(252)
  const sharpeRatio = stdReturn > 0 ? (avgReturn - 0.02) / stdReturn : 0

  let maxDrawdown = 0
  let peak = equity[0]
  for (let i = 1; i < equity.length; i++) {
    if (equity[i] > peak) peak = equity[i]
    const drawdown = (peak - equity[i]) / peak
    if (drawdown > maxDrawdown) maxDrawdown = drawdown
  }

  // Count actual winning round-trips (sell price > preceding buy price)
  let winningTrades = 0
  for (let i = 0; i < trades.length - 1; i++) {
    if (trades[i].type === "buy" && trades[i + 1].type === "sell") {
      if (trades[i + 1].price > trades[i].price) winningTrades++
    }
  }
  const totalRoundTrips = Math.floor(trades.filter((t) => t.type === "sell").length)
  const winRate = totalRoundTrips > 0 ? winningTrades / totalRoundTrips : 0

  return {
    totalReturn,
    sharpeRatio,
    maxDrawdown,
    winRate: Math.min(1, Math.max(0, winRate)),
    totalTrades: trades.length,
  }
}

// ===== CALIBRATION + THRESHOLD OPTIMIZATION =====

/**
 * Fit an isotonic-regression calibrator on a validation set.
 *
 * Use this once on held-out data BEFORE deploying.  The returned function
 * maps raw ensemble probabilities to empirically-calibrated win probabilities.
 *
 * @param rawProbs   raw ensemble probabilities from `predictDirection` (uncalibrated)
 * @param outcomes   binary outcomes (1 = price went up over horizon, 0 = down)
 * @param bins       number of binning buckets for PAVA (default 15)
 */
export function fitCalibrator(
  rawProbs: number[],
  outcomes: number[],
  bins = 15
): (p: number) => number {
  return calibrateProbabilities(rawProbs, outcomes, bins)
}

/**
 * Find the entry-threshold that maximises Sharpe ratio (with a minimum-trade
 * gate) on a validation set.  Replaces hard-coded 0.55 / 0.45 thresholds.
 *
 * The threshold is symmetric: if optimal is τ then BUY when prob > τ and
 * SELL when prob < 1 − τ.
 *
 * @param prices       validation-set price series (aligned with predictions)
 * @param predictions  validation-set predictions (aligned with prices)
 * @param minTrades    minimum number of round-trips required to trust a
 *                     threshold (default 10) — avoids picking a threshold that
 *                     fires once and gets lucky
 * @param grid         threshold candidates (default 0.50 → 0.75 step 0.01)
 */
export function optimizeThreshold(
  prices: number[],
  predictions: ModelPrediction[],
  minTrades = 10,
  grid: number[] = Array.from({ length: 26 }, (_, i) => 0.5 + i * 0.01)
): { threshold: number; sharpe: number; result: BacktestResult } {
  let bestThreshold = 0.55
  let bestSharpe = -Infinity
  let bestResult: BacktestResult = {
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winRate: 0,
    totalTrades: 0,
  }

  for (const t of grid) {
    const r = backtest(prices, predictions, t)
    if (r.totalTrades < minTrades) continue
    // Combined score: Sharpe primary, with a small win-rate tiebreaker
    const score = r.sharpeRatio + r.winRate * 0.1
    if (score > bestSharpe) {
      bestSharpe = score
      bestThreshold = t
      bestResult = r
    }
  }

  return { threshold: bestThreshold, sharpe: bestResult.sharpeRatio, result: bestResult }
}
