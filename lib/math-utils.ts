/**
 * Advanced Mathematical and Financial Utilities
 * Implements core calculations for ML-based trading
 */

// ===== STATISTICAL UTILITIES =====

export function mean(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

export function std(values: number[], ddof = 1): number {
  if (values.length === 0) return 0
  const m = mean(values)
  const variance = values.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / (values.length - ddof)
  return Math.sqrt(variance)
}

export function correlation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0
  const mx = mean(x)
  const my = mean(y)
  const sx = std(x)
  const sy = std(y)
  if (sx === 0 || sy === 0) return 0

  const covariance = x.reduce((acc, xi, i) => acc + (xi - mx) * (y[i] - my), 0) / (x.length - 1)
  return covariance / (sx * sy)
}

// ===== FRACTIONAL DIFFERENCING =====

/**
 * Fractional differencing to achieve stationarity while preserving memory
 * d = 0.4 is optimal (between returns d=1 and prices d=0)
 */
export function fractionalDifference(prices: number[], d = 0.4): number[] {
  if (prices.length === 0) return []

  const weights: number[] = [1]
  for (let k = 1; k < prices.length; k++) {
    const weight = (-d * (k - 1 - d)) / (k * (k + 1)) * weights[k - 1]
    weights.push(weight)
  }

  const result: number[] = []
  for (let i = 0; i < prices.length; i++) {
    let diff = 0
    for (let j = 0; j <= i && j < weights.length; j++) {
      diff += weights[j] * prices[i - j]
    }
    result.push(diff)
  }

  return result
}

/**
 * Augmented Dickey-Fuller test for stationarity
 * Returns p-value (lower is more stationary)
 */
export function adfTest(timeSeries: number[]): number {
  if (timeSeries.length < 3) return 1.0

  const n = timeSeries.length
  const y = timeSeries
  const x = timeSeries.slice(0, n - 1)

  // Simple regression: y = alpha + beta * x
  const xMean = mean(x)
  const yMean = mean(y.slice(1))

  const beta =
    x.reduce((acc, xi, i) => acc + (xi - xMean) * (y[i + 1] - yMean), 0) /
    x.reduce((acc, xi) => acc + Math.pow(xi - xMean, 2), 0)

  // t-statistic approximation
  const residuals = x.map((xi, i) => y[i + 1] - (yMean + beta * (xi - xMean)))
  const residualStd = std(residuals)
  const tStat = (beta - 1) / (residualStd / Math.sqrt(x.reduce((acc, xi) => acc + Math.pow(xi - xMean, 2), 0)))

  // Approximate p-value
  const pValue = Math.max(0.001, Math.min(0.999, 0.5 * (1 + Math.tanh(-Math.abs(tStat) / 1.5))))
  return pValue
}

// ===== REGIME DETECTION =====

export interface RegimeState {
  volatilityRegime: "low" | "medium" | "high"
  trendRegime: "uptrend" | "downtrend" | "sideways"
  volatilityPercentile: number
  trendDeviation: number
}

export function detectRegime(prices: number[], windowSize = 20): RegimeState {
  if (prices.length < windowSize) {
    return {
      volatilityRegime: "medium",
      trendRegime: "sideways",
      volatilityPercentile: 50,
      trendDeviation: 0,
    }
  }

  const recentPrices = prices.slice(-windowSize)
  const returns = recentPrices.slice(1).map((p, i) => (p - recentPrices[i]) / recentPrices[i])
  const volatility = std(returns)

  // Calculate volatility regime (percentile ranking)
  const allReturns = []
  for (let i = Math.max(0, prices.length - 100); i < prices.length - 1; i++) {
    allReturns.push((prices[i + 1] - prices[i]) / prices[i])
  }
  const volatilityPercentile =
    allReturns.filter((r) => std(allReturns.slice(-20)) > volatility).length / allReturns.length
  const volatilityRegime =
    volatilityPercentile < 0.33 ? "low" : volatilityPercentile < 0.67 ? "medium" : "high"

  // Calculate trend regime
  const sma = mean(recentPrices)
  const currentPrice = recentPrices[recentPrices.length - 1]
  const trendDeviation = (currentPrice - sma) / sma

  const trendRegime =
    trendDeviation > 0.02 ? "uptrend" : trendDeviation < -0.02 ? "downtrend" : "sideways"

  return {
    volatilityRegime,
    trendRegime,
    volatilityPercentile: volatilityPercentile * 100,
    trendDeviation,
  }
}

// ===== TECHNICAL INDICATORS =====

export interface TechnicalIndicators {
  rsi: number
  macd: { value: number; signal: number; histogram: number }
  bollingerBands: { upper: number; middle: number; lower: number }
  atr: number
}

/**
 * Relative Strength Index (RSI)
 * 0-30 oversold, 70-100 overbought
 */
export function calculateRSI(prices: number[], period = 14): number {
  if (prices.length < period + 1) return 50

  const changes = prices.slice(1).map((p, i) => p - prices[i])
  const gains = changes.filter((c) => c > 0)
  const losses = changes.filter((c) => c < 0).map((c) => Math.abs(c))

  const avgGain = mean(gains.slice(-period))
  const avgLoss = mean(losses.slice(-period))

  if (avgLoss === 0) return 100
  const rs = avgGain / avgLoss
  return 100 - 100 / (1 + rs)
}

/**
 * MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(prices: number[], fast = 12, slow = 26, signal = 9) {
  const ema = (data: number[], period: number) => {
    if (data.length < period) return data[data.length - 1]
    const multiplier = 2 / (period + 1)
    let emaVal = mean(data.slice(0, period))
    for (let i = period; i < data.length; i++) {
      emaVal = data[i] * multiplier + emaVal * (1 - multiplier)
    }
    return emaVal
  }

  const fastEMA = ema(prices, fast)
  const slowEMA = ema(prices, slow)
  const macdLine = fastEMA - slowEMA

  const macdHistory = []
  for (let i = slow - 1; i < prices.length; i++) {
    const fEMA = ema(prices.slice(0, i + 1), fast)
    const sEMA = ema(prices.slice(0, i + 1), slow)
    macdHistory.push(fEMA - sEMA)
  }

  const signalLine = ema(macdHistory, signal)
  const histogram = macdLine - signalLine

  return {
    value: macdLine,
    signal: signalLine,
    histogram: histogram,
  }
}

/**
 * Bollinger Bands
 */
export function calculateBollingerBands(prices: number[], period = 20, stdDevs = 2) {
  const recentPrices = prices.slice(-period)
  const middle = mean(recentPrices)
  const stdDev = std(recentPrices)

  return {
    upper: middle + stdDevs * stdDev,
    middle: middle,
    lower: middle - stdDevs * stdDev,
  }
}

/**
 * Average True Range (ATR)
 * Measures volatility
 */
export function calculateATR(prices: number[], period = 14): number {
  if (prices.length < 2) return 0

  const trueRanges = []
  for (let i = 1; i < prices.length; i++) {
    const tr = Math.max(
      prices[i] - prices[i - 1],
      Math.abs(prices[i] - prices[i - 1])
    )
    trueRanges.push(tr)
  }

  const recentTR = trueRanges.slice(-period)
  return mean(recentTR)
}

export function calculateTechnicalIndicators(prices: number[]): TechnicalIndicators {
  return {
    rsi: calculateRSI(prices),
    macd: calculateMACD(prices),
    bollingerBands: calculateBollingerBands(prices),
    atr: calculateATR(prices),
  }
}

// ===== RISK METRICS =====

/**
 * Value at Risk (VaR)
 * Worst-case loss at confidence level (default 95%)
 */
export function calculateVaR(returns: number[], confidenceLevel = 0.95): number {
  if (returns.length === 0) return 0
  const sorted = [...returns].sort((a, b) => a - b)
  const index = Math.floor(returns.length * (1 - confidenceLevel))
  return Math.abs(sorted[index])
}

/**
 * Conditional Value at Risk (CVaR)
 * Expected loss beyond VaR
 */
export function calculateCVaR(returns: number[], confidenceLevel = 0.95): number {
  if (returns.length === 0) return 0
  const sorted = [...returns].sort((a, b) => a - b)
  const index = Math.floor(returns.length * (1 - confidenceLevel))
  const tailReturns = sorted.slice(0, index + 1)
  return Math.abs(mean(tailReturns))
}

/**
 * Sharpe Ratio
 * Risk-adjusted return metric
 */
export function calculateSharpeRatio(returns: number[], riskFreeRate = 0.02): number {
  if (returns.length === 0) return 0
  const avgReturn = mean(returns) * 252 // Annualized
  const stdDev = std(returns) * Math.sqrt(252)
  if (stdDev === 0) return 0
  return (avgReturn - riskFreeRate) / stdDev
}

// ===== CORRELATION & COVARIANCE =====

export function covarianceMatrix(assetReturns: number[][]): number[][] {
  const n = assetReturns.length
  const matrix: number[][] = Array(n)
    .fill(null)
    .map(() => Array(n).fill(0))

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const corr = correlation(assetReturns[i], assetReturns[j])
      const std_i = std(assetReturns[i])
      const std_j = std(assetReturns[j])
      matrix[i][j] = corr * std_i * std_j
    }
  }

  return matrix
}

// ===== HIERARCHICAL RISK PARITY (HRP) =====

export function calculateHRP(correlationMatrix: number[][]): number[] {
  const n = correlationMatrix.length

  // Simple HRP implementation: inverse volatility weighting
  const weights = Array(n).fill(0)

  // Calculate returns volatility
  const volatilities = correlationMatrix.map((row) => Math.sqrt(row[0]))
  const totalVol = volatilities.reduce((a, b) => a + b, 0)

  // Allocate inversely to volatility
  for (let i = 0; i < n; i++) {
    weights[i] = (1 / volatilities[i]) / (n / totalVol)
  }

  // Normalize
  const sum = weights.reduce((a, b) => a + b, 0)
  return weights.map((w) => w / sum)
}
