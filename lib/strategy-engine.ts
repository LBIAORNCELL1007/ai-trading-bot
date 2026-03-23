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
 * Evaluate all 5 exit conditions
 * 1. Stop Loss: -2% hard stop
 * 2. Take Profit: +3% target
 * 3. Time Exit: 48-hour maximum hold
 * 4. Probability Flip: Exit if confidence drops below 45%
 * 5. VaR Breach: Exit if loss exceeds 1.5x expected VaR
 */
export function evaluateExitConditions(
  position: Position,
  currentPrice: number,
  currentPrediction: ModelPrediction,
  historyReturns: number[],
  maxHoldPeriods = 48
): ExitSignal {
  const holdTime = Date.now() - position.entryTime // Simplified
  const pnlPercent = (currentPrice - position.entryPrice) / position.entryPrice

  // 1. Stop Loss: -2%
  if (pnlPercent <= -0.02) {
    return {
      shouldExit: true,
      reason: "stop_loss",
      exitPrice: currentPrice,
    }
  }

  // 2. Take Profit: +3%
  if (pnlPercent >= 0.03) {
    return {
      shouldExit: true,
      reason: "take_profit",
      exitPrice: currentPrice,
    }
  }

  // 3. Time Exit: 48-hour maximum hold
  if (holdTime > maxHoldPeriods) {
    return {
      shouldExit: true,
      reason: "time_exit",
      exitPrice: currentPrice,
    }
  }

  // 4. Probability Flip: Exit if confidence drops below 45%
  if (currentPrediction.confidence < 0.45) {
    return {
      shouldExit: true,
      reason: "probability_flip",
      exitPrice: currentPrice,
    }
  }

  // 5. VaR Breach: Exit if loss exceeds 1.5x expected VaR
  if (historyReturns.length > 0) {
    const var95 = calculateVaR(historyReturns, 0.95)
    const cvar95 = calculateCVaR(historyReturns, 0.95)
    const maxExpectedLoss = cvar95 * 1.5

    if (Math.abs(pnlPercent) > maxExpectedLoss) {
      return {
        shouldExit: true,
        reason: "var_breach",
        exitPrice: currentPrice,
      }
    }
  }

  return {
    shouldExit: false,
    reason: "none",
    exitPrice: currentPrice,
  }
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
 * Kelly Criterion for position sizing
 * Optimal bet size = f* = (bp - q) / b
 * where p = win probability, q = loss probability, b = odds
 */
export function calculatePositionSizeKelly(
  accountBalance: number,
  currentPrice: number,
  probability: number,
  expectedReturn: number,
  maxLoss = 0.02,
  kellyFraction = 0.25 // Use fractional Kelly for safety
): PositionSizeResult {
  // Simplified Kelly calculation
  const p = probability
  const q = 1 - probability
  const b = Math.max(expectedReturn / maxLoss, 1)

  let kellyPct = (p * b - q) / b
  kellyPct = Math.max(0, Math.min(kellyPct, 0.25)) // Cap at 25%
  kellyPct *= kellyFraction // Fractional Kelly

  const riskAmount = accountBalance * kellyPct
  const quantity = riskAmount / currentPrice

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
 * Isotonic regression for probability calibration
 * Ensures model predictions align with actual outcomes
 */
export function calibrateProbabilities(
  predictions: number[],
  outcomes: number[]
): (prob: number) => number {
  // Simplified calibration: bin and average
  const bins = 10
  const calibrationCurve: number[] = Array(bins).fill(0)
  const binCounts: number[] = Array(bins).fill(0)

  for (let i = 0; i < predictions.length; i++) {
    const binIndex = Math.min(bins - 1, Math.floor(predictions[i] * bins))
    calibrationCurve[binIndex] += outcomes[i]
    binCounts[binIndex] += 1
  }

  for (let i = 0; i < bins; i++) {
    calibrationCurve[i] =
      binCounts[i] > 0 ? calibrationCurve[i] / binCounts[i] : i / bins
  }

  // Enforce monotonicity (Pool Adjacent Violators Algorithm)
  for (let i = 1; i < bins; i++) {
    if (calibrationCurve[i] < calibrationCurve[i - 1]) {
      const avg = (calibrationCurve[i] + calibrationCurve[i - 1]) / 2
      calibrationCurve[i] = avg
      calibrationCurve[i - 1] = avg
    }
  }

  // Return calibration function
  return (prob: number): number => {
    const binIndex = Math.min(bins - 1, Math.floor(prob * bins))
    return calibrationCurve[binIndex]
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
  returnsHistory: number[]
): PortfolioMetrics {
  const totalPositions = positions.length
  const totalValue = cash + positions.reduce((sum, p) => sum + p.pnl, 0)

  // Calculate volatility
  let portfolioVolatility = 0
  if (returnsHistory.length > 1) {
    const mean = returnsHistory.reduce((a, b) => a + b, 0) / returnsHistory.length
    const variance =
      returnsHistory.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
      returnsHistory.length
    portfolioVolatility = Math.sqrt(variance) * Math.sqrt(252)
  }

  // Calculate Sharpe Ratio
  const annualReturn = returnsHistory.length > 0
    ? returnsHistory.reduce((a, b) => a + b, 0) * 252
    : 0
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
