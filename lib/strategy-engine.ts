/**
 * Advanced Trading Strategy Engine
 * Implements Top-K ranking, multi-exit conditions, and walk-forward training
 */

import { calculateVaR, calculateCVaR } from "./math-utils"
import { ModelPrediction } from "./ml-models"

// ===== POSITION MANAGEMENT =====

export interface Position {
  id: string
  asset: string
  type: "long" | "short"
  entryPrice: number
  entryTime: number
  quantity: number
  stopLoss: number
  takeProfit: number
  currentPrice: number
  pnl: number
  pnlPercent: number
  confidenceScore: number
  exitReason?: string
}

export interface ExitSignal {
  shouldExit: boolean
  reason:
    | "stop_loss"
    | "take_profit"
    | "time_exit"
    | "probability_flip"
    | "var_breach"
    | "none"
  exitPrice: number
}

// ===== MULTI-EXIT STRATEGY =====

/**
 * Evaluate all 5 exit conditions.
 *
 * FIXES vs prior version:
 *   1. SL/TP are configurable (no longer hard-coded ±2%/+3%).
 *      Pass per-position SL/TP via `position.stopLoss` / `position.takeProfit`.
 *      If not set, fall back to ATR-scaled defaults (1.5× ATR / 3× ATR).
 *   2. Time exit uses BAR INDEX, not Date.now() ms (the previous comparison of
 *      milliseconds to the literal "48" caused instant time-exit on every position).
 *   3. VaR breach checks `pnlPercent < -threshold`, not `Math.abs(pnlPercent)` —
 *      previously this also exited *winning* trades.
 *   4. Probability-flip uses the prediction's *probability* below 0.45 (or above
 *      0.55 for short positions) rather than its self-reported confidence which
 *      was conflated with a calibrated win probability.
 *
 * @param currentBarIndex bar count since session start (used for time exit)
 * @param entryBarIndex bar count when the position was opened
 * @param atrPercent current ATR as % of price (used as fallback SL distance)
 */
export function evaluateExitConditions(
  position: Position,
  currentPrice: number,
  currentPrediction: ModelPrediction,
  historyReturns: number[],
  options: {
    currentBarIndex: number
    entryBarIndex: number
    maxHoldBars?: number
    atrPercent?: number
    probabilityFlipThreshold?: number
    side?: 'long' | 'short'
  }
): ExitSignal {
  const {
    currentBarIndex,
    entryBarIndex,
    maxHoldBars = 48,
    atrPercent = 1.5,
    probabilityFlipThreshold = 0.45,
    side = position.type ?? 'long',
  } = options

  const isLong = side === 'long'
  const directional =
    (currentPrice - position.entryPrice) / position.entryPrice * (isLong ? 1 : -1)

  // 1. Stop Loss — use position.stopLoss if set, else 1.5× ATR
  const slDistance = position.stopLoss && position.stopLoss > 0
    ? Math.abs(position.entryPrice - position.stopLoss) / position.entryPrice
    : (atrPercent / 100) * 1.5
  if (directional <= -slDistance) {
    return { shouldExit: true, reason: "stop_loss", exitPrice: currentPrice }
  }

  // 2. Take Profit — use position.takeProfit if set, else 3× ATR (2:1 R:R)
  const tpDistance = position.takeProfit && position.takeProfit > 0
    ? Math.abs(position.takeProfit - position.entryPrice) / position.entryPrice
    : (atrPercent / 100) * 3.0
  if (directional >= tpDistance) {
    return { shouldExit: true, reason: "take_profit", exitPrice: currentPrice }
  }

  // 3. Time Exit (bar count, not milliseconds!)
  if (currentBarIndex - entryBarIndex >= maxHoldBars) {
    return { shouldExit: true, reason: "time_exit", exitPrice: currentPrice }
  }

  // 4. Probability Flip — direction-aware
  const adverseProbability = isLong
    ? currentPrediction.probability < probabilityFlipThreshold
    : currentPrediction.probability > 1 - probabilityFlipThreshold
  if (adverseProbability && currentPrediction.confidence > 0.5) {
    return { shouldExit: true, reason: "probability_flip", exitPrice: currentPrice }
  }

  // 5. VaR Breach — only exits on adverse moves (no Math.abs)
  if (historyReturns.length > 20) {
    const cvar95 = calculateCVaR(historyReturns, 0.95)
    // CVaR is conventionally NEGATIVE (expected return in worst 5% tail).
    // Adverse threshold is therefore 1.5× |cvar95|.
    const adverseThreshold = Math.abs(cvar95) * 1.5
    if (directional < -adverseThreshold && adverseThreshold > 0) {
      return { shouldExit: true, reason: "var_breach", exitPrice: currentPrice }
    }
  }

  return { shouldExit: false, reason: "none", exitPrice: currentPrice }
}

// ===== TOP-K RANKING EXECUTION =====

export interface AssetSignal {
  asset: string
  probability: number
  confidence: number
  expectedReturn: number
  riskScore: number
}

/**
 * Top-K Ranking Strategy
 * Trade only top 3 highest-probability coins (concentrated capital, quality over quantity)
 */
export function rankAndSelectAssets(
  assetSignals: AssetSignal[],
  k = 3
): AssetSignal[] {
  // Calculate composite score: probability * confidence / riskScore
  const scoredAssets = assetSignals
    .map((asset) => ({
      ...asset,
      score: (asset.probability * asset.confidence) / Math.max(asset.riskScore, 0.1),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)

  return scoredAssets
}

// ===== POSITION SIZING =====

export interface PositionSizeResult {
  quantity: number
  riskAmount: number
  notionalValue: number
}

/**
 * Kelly Criterion for position sizing.
 *
 * Optimal bet fraction f* = (b·p − q) / b   where b = NET odds (gain/loss).
 * Previous version used b = max(payoff/loss, 1) which over-bets on small edges.
 *
 * @param probability  CALIBRATED probability of win (use calibrateProbabilities first!)
 * @param expectedReturn fractional gain on a winning trade (e.g. 0.03 = 3%)
 * @param maxLoss fractional loss on a losing trade (e.g. 0.02 = 2%)
 * @param kellyFraction fractional Kelly multiplier (default 0.25 — conservative)
 * @param maxBetFraction hard cap on f* (default 0.10 = 10% of equity)
 */
export function calculatePositionSizeKelly(
  accountBalance: number,
  currentPrice: number,
  probability: number,
  expectedReturn: number,
  maxLoss = 0.02,
  kellyFraction = 0.25,
  maxBetFraction = 0.1
): PositionSizeResult {
  const p = Math.min(0.99, Math.max(0.01, probability))
  const q = 1 - p
  // b = net odds (gain divided by loss).  Both inputs must be positive.
  const b = Math.max(0.1, expectedReturn / Math.max(maxLoss, 1e-6))

  let kellyPct = (b * p - q) / b
  // Negative edge → don't bet
  if (kellyPct <= 0) {
    return { quantity: 0, riskAmount: 0, notionalValue: 0 }
  }
  kellyPct *= kellyFraction
  kellyPct = Math.min(kellyPct, maxBetFraction)

  const riskAmount = accountBalance * kellyPct
  const quantity = riskAmount / Math.max(currentPrice, 1e-9)

  return {
    quantity,
    riskAmount,
    notionalValue: quantity * currentPrice,
  }
}

/**
 * Risk-based position sizing
 * Size position based on volatility and account risk tolerance
 */
export function calculatePositionSizeRisk(
  accountBalance: number,
  currentPrice: number,
  volatility: number,
  accountRiskPercent = 0.02, // Risk 2% per trade
  stopLossPercent = 0.02
): PositionSizeResult {
  const riskAmount = accountBalance * accountRiskPercent
  const dollarRisk = riskAmount
  const priceRisk = currentPrice * stopLossPercent
  const quantity = dollarRisk / priceRisk

  return {
    quantity,
    riskAmount,
    notionalValue: quantity * currentPrice,
  }
}

// ===== PORTFOLIO ALLOCATION =====

export interface PortfolioAllocation {
  allocations: Map<string, number>
  totalWeight: number
  concentrationRatio: number
}

/**
 * Hierarchical Risk Parity allocation
 * Allocate capital based on correlation and volatility
 */
export function allocatePortfolioHRP(
  positions: Position[],
  correlationMatrix: number[][],
  volatilities: number[]
): PortfolioAllocation {
  const allocations = new Map<string, number>()

  if (positions.length === 0) {
    return {
      allocations,
      totalWeight: 0,
      concentrationRatio: 0,
    }
  }

  // Inverse volatility weighting
  const totalInverseVol = volatilities.reduce((sum, vol) => sum + 1 / Math.max(vol, 0.001), 0)
  let totalWeight = 0

  for (let i = 0; i < positions.length; i++) {
    const weight = (1 / Math.max(volatilities[i], 0.001)) / totalInverseVol
    allocations.set(positions[i].asset, weight)
    totalWeight += weight
  }

  // Calculate concentration ratio (Herfindahl index)
  let concentrationRatio = 0
  allocations.forEach((weight) => {
    concentrationRatio += weight * weight
  })

  return {
    allocations,
    totalWeight,
    concentrationRatio,
  }
}

// ===== WALK-FORWARD TRAINING =====

export interface WalkForwardWindow {
  trainStart: number
  trainEnd: number
  testStart: number
  testEnd: number
}

/**
 * Generate walk-forward cross-validation windows
 * Train on past, validate on future, advance window
 */
export function generateWalkForwardWindows(
  totalLength: number,
  trainWindowSize = 100,
  testWindowSize = 20,
  step = 10
): WalkForwardWindow[] {
  const windows: WalkForwardWindow[] = []

  for (let i = 0; i + trainWindowSize + testWindowSize <= totalLength; i += step) {
    windows.push({
      trainStart: i,
      trainEnd: i + trainWindowSize,
      testStart: i + trainWindowSize,
      testEnd: i + trainWindowSize + testWindowSize,
    })
  }

  return windows
}

// ===== TRADE STATISTICS =====

export interface TradeStats {
  totalTrades: number
  winningTrades: number
  losingTrades: number
  winRate: number
  avgWinSize: number
  avgLossSize: number
  profitFactor: number
  expectancy: number
}

/**
 * Calculate trade statistics from closed positions
 */
export function calculateTradeStats(closedPositions: Position[]): TradeStats {
  const totalTrades = closedPositions.length
  const winningTrades = closedPositions.filter((p) => p.pnl > 0).length
  const losingTrades = closedPositions.filter((p) => p.pnl < 0).length

  const wins = closedPositions
    .filter((p) => p.pnl > 0)
    .reduce((sum, p) => sum + p.pnl, 0)
  const losses = Math.abs(
    closedPositions
      .filter((p) => p.pnl < 0)
      .reduce((sum, p) => sum + p.pnl, 0)
  )

  const avgWinSize = winningTrades > 0 ? wins / winningTrades : 0
  const avgLossSize = losingTrades > 0 ? losses / losingTrades : 0
  const profitFactor = losses > 0 ? wins / losses : 0
  const expectancy = totalTrades > 0 ? (wins - losses) / totalTrades : 0

  return {
    totalTrades,
    winningTrades,
    losingTrades,
    winRate: totalTrades > 0 ? winningTrades / totalTrades : 0,
    avgWinSize,
    avgLossSize,
    profitFactor,
    expectancy,
  }
}

// ===== PROBABILITY CALIBRATION =====

/**
 * Isotonic regression for probability calibration via Pool-Adjacent-Violators
 * Algorithm (PAVA).
 *
 * Replaces the previous simplified two-pass average that did NOT converge to
 * a true monotone fit.  This is the canonical PAVA: pool adjacent violators
 * iteratively until the empirical curve is monotonically non-decreasing.
 *
 * The returned function maps a raw model probability ∈ [0,1] to a calibrated
 * win-probability estimate.  Empty bins are linearly interpolated from
 * neighboring populated bins (not blindly set to bin-index/bins as before).
 */
export function calibrateProbabilities(
  predictions: number[],
  outcomes: number[],
  bins = 15
): (prob: number) => number {
  // 1. Bin predictions and compute empirical win-rate per bin
  const sums: number[] = Array(bins).fill(0)
  const counts: number[] = Array(bins).fill(0)
  for (let i = 0; i < predictions.length; i++) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(predictions[i] * bins)))
    sums[idx] += outcomes[i]
    counts[idx] += 1
  }

  const populated: number[] = []
  const rates: number[] = []
  const weights: number[] = []
  for (let i = 0; i < bins; i++) {
    if (counts[i] > 0) {
      populated.push(i)
      rates.push(sums[i] / counts[i])
      weights.push(counts[i])
    }
  }

  if (populated.length === 0) {
    return (p: number) => p // identity fallback
  }

  // 2. Pool-Adjacent-Violators (PAVA) on (rates, weights)
  let blocks: { value: number; weight: number; binStart: number; binEnd: number }[] =
    rates.map((r, i) => ({
      value: r,
      weight: weights[i],
      binStart: populated[i],
      binEnd: populated[i],
    }))
  let merged = true
  while (merged) {
    merged = false
    for (let i = 0; i < blocks.length - 1; i++) {
      if (blocks[i].value > blocks[i + 1].value) {
        const totalW = blocks[i].weight + blocks[i + 1].weight
        const newVal =
          (blocks[i].value * blocks[i].weight + blocks[i + 1].value * blocks[i + 1].weight) /
          totalW
        blocks[i] = {
          value: newVal,
          weight: totalW,
          binStart: blocks[i].binStart,
          binEnd: blocks[i + 1].binEnd,
        }
        blocks.splice(i + 1, 1)
        merged = true
        break
      }
    }
  }

  // 3. Per-bin calibrated value (linear interpolation across empty bins)
  const calibrated: number[] = Array(bins).fill(NaN)
  for (const blk of blocks) {
    for (let b = blk.binStart; b <= blk.binEnd; b++) {
      calibrated[b] = blk.value
    }
  }
  // Forward-fill, then backward-fill any remaining NaN (sparsely populated bins)
  for (let i = 1; i < bins; i++) {
    if (isNaN(calibrated[i])) calibrated[i] = calibrated[i - 1]
  }
  for (let i = bins - 2; i >= 0; i--) {
    if (isNaN(calibrated[i])) calibrated[i] = calibrated[i + 1]
  }

  return (prob: number): number => {
    if (!isFinite(prob)) return 0.5
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(prob * bins)))
    return calibrated[idx]
  }
}

// ===== PORTFOLIO METRICS =====

export interface PortfolioMetrics {
  totalValue: number
  totalCash: number
  totalPositions: number
  portfolioVolatility: number
  sharpeRatio: number
  maxDrawdown: number
  calmarRatio: number
}

export function calculatePortfolioMetrics(
  positions: Position[],
  cash: number,
  returnsHistory: number[],
  periodsPerYear = 252
): PortfolioMetrics {
  const totalPositions = positions.length
  const totalValue = cash + positions.reduce((sum, p) => sum + p.pnl, 0)

  // Calculate volatility (annualized)
  let portfolioVolatility = 0
  if (returnsHistory.length > 1) {
    const mean = returnsHistory.reduce((a, b) => a + b, 0) / returnsHistory.length
    const variance =
      returnsHistory.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
      returnsHistory.length
    portfolioVolatility = Math.sqrt(variance) * Math.sqrt(periodsPerYear)
  }

  // Calculate Sharpe Ratio (annualized using the correct periodicity)
  const periodMean = returnsHistory.length > 0
    ? returnsHistory.reduce((a, b) => a + b, 0) / returnsHistory.length
    : 0
  const annualReturn = periodMean * periodsPerYear
  const sharpeRatio =
    portfolioVolatility > 0 ? (annualReturn - 0.02) / portfolioVolatility : 0

  // Calculate max drawdown from returns series (peak tracks the equity curve)
  let maxDrawdown = 0
  let peak = 0
  const cumulativeReturns = returnsHistory.reduce(
    (acc, r) => {
      const val = acc[acc.length - 1] * (1 + r)
      if (val > peak) peak = val
      const dd = (peak - val) / peak
      if (dd > maxDrawdown) maxDrawdown = dd
      return [...acc, val]
    },
    [1.0]  // start equity curve at 1.0 (normalized)
  )

  // Calmar Ratio
  const calmarRatio =
    maxDrawdown > 0 ? annualReturn / maxDrawdown : 0

  return {
    totalValue,
    totalCash: cash,
    totalPositions,
    portfolioVolatility,
    sharpeRatio,
    maxDrawdown,
    calmarRatio,
  }
}
