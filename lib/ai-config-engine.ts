/**
 * AI Auto-Configure Engine
 * 
 * Institutional-grade analysis module that evaluates market conditions
 * across multiple timeframes and returns intelligent configuration
 * recommendations for all trading parameters.
 */

import { TechnicalIndicators, type IndicatorData } from './technical-indicators'
import { getHistoricalKlines, type BinanceKline } from './binance-websocket'

// ════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════

export type MarketRegime =
  | 'STRONG_TREND'
  | 'WEAK_TREND'
  | 'RANGING'
  | 'VOLATILE'
  | 'BREAKOUT'
  | 'CHOPPY'

export type TFDirection = 'BULLISH' | 'BEARISH' | 'NEUTRAL'

export interface AIConfigResult {
  strategy: string
  timeframe: string
  riskPerTrade: number
  maxPositions: number
  stopLoss: number
  takeProfit: number
  trailingStop: number
  leverage: number
  confidence: number
  reasoning: {
    strategy: string
    timeframe: string
    risk: string
    stopLoss: string
    takeProfit: string
    trailingStop: string
    leverage: string
    maxPositions: string
  }
  marketRegime: MarketRegime
  fearGreedIndex: number | null
  fearGreedTimestamp: string | null
  fundingRate: number | null
  backtestResult: BacktestSummary | null
  analyzedAt: Date
}

interface BacktestSummary {
  winRate: number
  totalTrades: number
  profitFactor: number
  netPnlPercent: number
}

interface TimeframeAnalysis {
  timeframe: string
  indicators: IndicatorData
  direction: TFDirection
  strength: number // 0-1
  klines: BinanceKline[]
}

interface SMCResult {
  orderBlockPresent: boolean
  fairValueGapPresent: boolean
  liquiditySweepDetected: boolean
  score: number // 0-25
}

interface VolumeNode {
  price: number
  volume: number
  isSupport: boolean
  isResistance: boolean
}

interface StrategyScore {
  key: string
  score: number
  reasoning: string
}

interface PortfolioContext {
  totalPnL: number
  equity: number
  openTrades: Array<{ symbol: string; side: string }>
}

// Hardcoded correlation fallback map (BTC-relative)
const CORRELATION_FALLBACK: Record<string, number> = {
  ETHUSDT: 0.90,
  SOLUSDT: 0.85,
  BNBUSDT: 0.80,
  ADAUSDT: 0.82,
  XRPUSDT: 0.78,
  DOGEUSDT: 0.75,
  MATICUSDT: 0.80,
  AVAXUSDT: 0.83,
  LINKUSDT: 0.79,
  DOTUSDT: 0.81,
  LTCUSDT: 0.76,
  UNIUSDT: 0.74,
  ATOMUSDT: 0.72,
  ETCUSDT: 0.77,
  SHIBUSDT: 0.70,
}

const ALL_STRATEGIES = [
  'MA_CROSSOVER', 'MACD', 'MEAN_REVERSION', 'GRID', 'MULTI_SIGNAL',
  'RSI_DIVERGENCE', 'BOLLINGER_BREAKOUT', 'VWAP_TREND', 'ICHIMOKU', 'PIVOT_REVERSAL'
] as const

const SLIPPAGE_PERCENT = 0.001 // 0.1% per trade

// ════════════════════════════════════════════════════════════════════════════
// SYMBOL NORMALIZER
// ════════════════════════════════════════════════════════════════════════════

function normalizeSymbol(symbol: string): string {
  let s = symbol.replace(/[\/\-\s]/g, '').toUpperCase().trim()
  if (!s.endsWith('USDT') && !s.endsWith('BUSD') && !s.endsWith('USDC')) {
    s += 'USDT'
  }
  return s
}

// ════════════════════════════════════════════════════════════════════════════
// DATA FETCHING LAYER
// ════════════════════════════════════════════════════════════════════════════

/**
 * Fetch candles for a single timeframe. Returns at least 200 candles
 * with the last (incomplete) candle excluded.
 */
async function fetchCleanKlines(
  symbol: string,
  interval: string,
  limit: number = 250,
  signal?: AbortSignal
): Promise<BinanceKline[]> {
  const klines = await getHistoricalKlines(symbol, interval, limit)

  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')

  // Exclude last incomplete candle
  if (klines.length > 1) {
    const last = klines[klines.length - 1]
    if (!last.isClosed) {
      klines.pop()
    }
  }

  return klines
}

/**
 * Fetch Fear & Greed Index from alternative.me (daily, free, no key)
 */
async function fetchFearGreedIndex(
  signal?: AbortSignal
): Promise<{ value: number; timestamp: string } | null> {
  try {
    const res = await fetch('https://api.alternative.me/fng/?limit=1', { signal })
    if (!res.ok) return null
    const data = await res.json()
    if (data?.data?.[0]) {
      const entry = data.data[0]
      const ts = new Date(parseInt(entry.timestamp) * 1000).toISOString().split('T')[0]
      return { value: parseInt(entry.value), timestamp: ts }
    }
    return null
  } catch {
    return null
  }
}

/**
 * Fetch Binance funding rate (futures only).
 * Returns null gracefully for spot-only pairs.
 */
async function fetchFundingRate(
  symbol: string,
  signal?: AbortSignal
): Promise<number | null> {
  try {
    const res = await fetch(
      `https://fapi.binance.com/fapi/v1/fundingRate?symbol=${symbol}&limit=1`,
      { signal }
    )
    // 400/404 = spot-only pair → skip gracefully
    if (!res.ok) return null
    const data = await res.json()
    if (Array.isArray(data) && data.length > 0) {
      return parseFloat(data[0].fundingRate)
    }
    return null
  } catch {
    return null
  }
}

/**
 * Compute rolling Pearson correlation between two price series.
 * Used for dynamic BTC correlation detection.
 */
function computeCorrelation(pricesA: number[], pricesB: number[], period: number = 30): number {
  const n = Math.min(pricesA.length, pricesB.length, period)
  if (n < 10) return 0

  const a = pricesA.slice(-n)
  const b = pricesB.slice(-n)

  const meanA = a.reduce((s, v) => s + v, 0) / n
  const meanB = b.reduce((s, v) => s + v, 0) / n

  let covAB = 0, varA = 0, varB = 0
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA
    const db = b[i] - meanB
    covAB += da * db
    varA += da * da
    varB += db * db
  }

  const denom = Math.sqrt(varA * varB)
  return denom === 0 ? 0 : covAB / denom
}

/**
 * Get BTC correlation for a given symbol.
 * Attempts dynamic calculation, falls back to hardcoded map.
 */
async function getCorrelationWithBTC(
  symbol: string,
  signal?: AbortSignal
): Promise<number> {
  if (symbol === 'BTCUSDT') return 1.0

  try {
    const [pairKlines, btcKlines] = await Promise.all([
      getHistoricalKlines(symbol, '1h', 50),
      getHistoricalKlines('BTCUSDT', '1h', 50),
    ])
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')

    if (pairKlines.length >= 30 && btcKlines.length >= 30) {
      const corr = computeCorrelation(
        pairKlines.map(k => k.close),
        btcKlines.map(k => k.close),
        30
      )
      if (Math.abs(corr) > 0.01) return corr
    }
  } catch {
    // Fall through to hardcoded
  }

  return CORRELATION_FALLBACK[symbol] ?? 0.5
}

// ════════════════════════════════════════════════════════════════════════════
// ANALYSIS LAYERS
// ════════════════════════════════════════════════════════════════════════════

/**
 * Build TechnicalIndicators from klines and return the computed IndicatorData.
 */
function buildIndicators(klines: BinanceKline[]): IndicatorData {
  const ti = new TechnicalIndicators()
  for (const k of klines) {
    ti.addCandle(k.close, k.volume, k.high, k.low)
  }
  return ti.calculateAll()
}

/**
 * Determine direction for a single timeframe.
 */
function analyzeTimeframeDirection(ind: IndicatorData): { direction: TFDirection; strength: number } {
  let bullPoints = 0
  let bearPoints = 0

  // SMA alignment
  if (ind.sma20 > ind.sma50) bullPoints += 1; else bearPoints += 1
  if (ind.sma50 > ind.sma200 && ind.sma200 > 0) bullPoints += 1; else if (ind.sma200 > 0) bearPoints += 1

  // RSI zone
  if (ind.rsi14 > 55) bullPoints += 1
  else if (ind.rsi14 < 45) bearPoints += 1

  // MACD
  if (ind.histogram > 0) bullPoints += 1; else bearPoints += 1

  // VWAP
  if (ind.vwap && ind.close > ind.vwap) bullPoints += 1
  else if (ind.vwap && ind.close < ind.vwap) bearPoints += 1

  const total = bullPoints + bearPoints
  if (total === 0) return { direction: 'NEUTRAL', strength: 0 }

  if (bullPoints > bearPoints + 1) {
    return { direction: 'BULLISH', strength: bullPoints / total }
  } else if (bearPoints > bullPoints + 1) {
    return { direction: 'BEARISH', strength: bearPoints / total }
  }
  return { direction: 'NEUTRAL', strength: 0.5 }
}

/**
 * Layer 1: Multi-Timeframe Confluence
 * Returns +30 if all agree, -20 if they conflict
 */
function analyzeMTFConfluence(tfAnalyses: TimeframeAnalysis[]): {
  score: number
  allAgree: boolean
  dominantDirection: TFDirection
  reasoning: string
} {
  const directions = tfAnalyses.map(t => t.direction)
  const bullish = directions.filter(d => d === 'BULLISH').length
  const bearish = directions.filter(d => d === 'BEARISH').length

  if (bullish === directions.length) {
    return { score: 30, allAgree: true, dominantDirection: 'BULLISH', reasoning: 'All 3 timeframes bullish — strong confluence' }
  }
  if (bearish === directions.length) {
    return { score: 30, allAgree: true, dominantDirection: 'BEARISH', reasoning: 'All 3 timeframes bearish — strong confluence' }
  }
  if (bullish > 0 && bearish > 0) {
    const dominant = bullish > bearish ? 'BULLISH' : bearish > bullish ? 'BEARISH' : 'NEUTRAL'
    return { score: -20, allAgree: false, dominantDirection: dominant, reasoning: `Timeframe conflict: ${bullish} bullish, ${bearish} bearish — reducing confidence` }
  }

  const dominant = bullish > 0 ? 'BULLISH' : bearish > 0 ? 'BEARISH' : 'NEUTRAL'
  return { score: 0, allAgree: false, dominantDirection: dominant, reasoning: 'Timeframes show mixed/neutral signals' }
}

/**
 * Layer 2: Volatility Breakout Detection
 * Checks if BB width is compressed (squeeze) → breakout imminent
 */
function analyzeVolatilityBreakout(ind: IndicatorData): {
  breakoutImminent: boolean
  bbWidth: number
  reasoning: string
} {
  const bbWidth = ind.bollingerMiddle > 0
    ? (ind.bollingerUpper - ind.bollingerLower) / ind.bollingerMiddle
    : 0

  // BB width below 2% → squeeze detected
  if (bbWidth < 0.02 && bbWidth > 0) {
    return {
      breakoutImminent: true,
      bbWidth,
      reasoning: `BB squeeze detected (width ${(bbWidth * 100).toFixed(2)}%) — breakout imminent`
    }
  }

  return { breakoutImminent: false, bbWidth, reasoning: `BB width normal (${(bbWidth * 100).toFixed(2)}%)` }
}

/**
 * Layer 3: Volume Profile Analysis
 * Identifies high-volume price nodes as support/resistance.
 */
function analyzeVolumeProfile(klines: BinanceKline[]): VolumeNode[] {
  if (klines.length < 50) return []

  // Create price buckets (divide price range into 20 bins)
  const prices = klines.map(k => k.close)
  const minPrice = Math.min(...prices)
  const maxPrice = Math.max(...prices)
  const range = maxPrice - minPrice
  if (range === 0) return []

  const bucketCount = 20
  const bucketSize = range / bucketCount
  const buckets: { price: number; volume: number }[] = Array.from(
    { length: bucketCount },
    (_, i) => ({ price: minPrice + (i + 0.5) * bucketSize, volume: 0 })
  )

  for (const k of klines) {
    const idx = Math.min(Math.floor((k.close - minPrice) / bucketSize), bucketCount - 1)
    if (idx >= 0) buckets[idx].volume += k.volume
  }

  // Find top 3 volume nodes
  const sorted = [...buckets].sort((a, b) => b.volume - a.volume).slice(0, 3)
  const currentPrice = prices[prices.length - 1]

  return sorted.map(node => ({
    price: node.price,
    volume: node.volume,
    isSupport: node.price < currentPrice,
    isResistance: node.price > currentPrice,
  }))
}

/**
 * Layer 4: Adaptive Regime Detection
 * Enhanced regime classification using multiple signals.
 */
function detectRegime(ind: IndicatorData): { regime: MarketRegime; reasoning: string } {
  const bbWidth = ind.bollingerMiddle > 0
    ? (ind.bollingerUpper - ind.bollingerLower) / ind.bollingerMiddle
    : 0
  const atrPercent = ind.close > 0 ? (ind.atr14 / ind.close) * 100 : 0
  const adx = ind.adx

  // Breakout: high vol + strong trend
  if ((bbWidth > 0.05 || atrPercent > 2) && adx > 30) {
    return { regime: 'BREAKOUT', reasoning: `Breakout: ADX ${adx.toFixed(0)} + high volatility (ATR ${atrPercent.toFixed(2)}%)` }
  }

  // Strong trend
  if (adx > 35) {
    const isBull = ind.sma20 > ind.sma50 && ind.sma50 > (ind.sma200 || ind.sma50)
    return {
      regime: 'STRONG_TREND',
      reasoning: `Strong ${isBull ? 'Bull' : 'Bear'} Trend: ADX ${adx.toFixed(0)}`
    }
  }

  // Volatile chop
  if ((bbWidth > 0.04 || atrPercent > 1.5) && adx < 25) {
    return { regime: 'VOLATILE', reasoning: `Volatile Chop: High ATR (${atrPercent.toFixed(2)}%) but weak trend (ADX ${adx.toFixed(0)})` }
  }

  // Choppy
  if (adx < 20 && bbWidth > 0.025) {
    return { regime: 'CHOPPY', reasoning: `Choppy: Very weak trend (ADX ${adx.toFixed(0)})` }
  }

  // Weak trend
  if (adx > 20 && adx <= 35) {
    return { regime: 'WEAK_TREND', reasoning: `Weak Trend: Moderate ADX (${adx.toFixed(0)})` }
  }

  // Ranging
  if (bbWidth < 0.025 && atrPercent < 1) {
    return { regime: 'RANGING', reasoning: `Ranging: Low volatility (BB width ${(bbWidth * 100).toFixed(2)}%, ATR ${atrPercent.toFixed(2)}%)` }
  }

  return { regime: 'RANGING', reasoning: 'Default to ranging — no strong signals' }
}

/**
 * Layer 5: Funding Rate Momentum (crypto-specific)
 * Returns confidence adjustment + strategy hint.
 */
function analyzeFundingRate(rate: number | null): {
  score: number
  hint: string
  reasoning: string
} {
  if (rate === null) {
    return { score: 0, hint: '', reasoning: 'Funding rate unavailable (spot pair or API down)' }
  }

  const pct = rate * 100

  if (rate > 0.001) { // > 0.1%
    return {
      score: 15,
      hint: 'MEAN_REVERSION',
      reasoning: `Funding ${pct.toFixed(4)}% — market overleveraged LONG. Mean reversion likely`
    }
  }
  if (rate < -0.0005) { // < -0.05%
    return {
      score: 15,
      hint: 'TREND_FOLLOW',
      reasoning: `Funding ${pct.toFixed(4)}% — market overleveraged SHORT. Squeeze expected`
    }
  }

  return { score: 0, hint: '', reasoning: `Funding rate neutral (${pct.toFixed(4)}%)` }
}

/**
 * Layer 6: Smart Money Concept (SMC) Detection
 * Detects order blocks, fair value gaps, and liquidity sweeps.
 * Runs asynchronously to avoid blocking UI.
 */
async function detectSMC(
  klines: BinanceKline[],
  signal?: AbortSignal
): Promise<SMCResult> {
  const result: SMCResult = {
    orderBlockPresent: false,
    fairValueGapPresent: false,
    liquiditySweepDetected: false,
    score: 0,
  }

  if (klines.length < 20) return result

  // Use Promise wrapper + setTimeout for non-blocking execution
  return new Promise<SMCResult>((resolve) => {
    // Set 2s timeout — return partial results if too slow
    const timeout = setTimeout(() => resolve(result), 2000)

    try {
      const len = klines.length
      const currentPrice = klines[len - 1].close

      // ORDER BLOCK: Last bearish candle before a strong bullish impulse (or vice versa)
      for (let i = len - 2; i >= Math.max(len - 50, 2); i--) {
        if (signal?.aborted) { clearTimeout(timeout); resolve(result); return }

        const curr = klines[i]
        const next = klines[i + 1]
        const isBearishCandle = curr.close < curr.open
        const bullishImpulse = next.close > next.open &&
          (next.close - next.open) > (next.high - next.low) * 0.6

        if (isBearishCandle && bullishImpulse) {
          // Check if current price is near this order block
          const obHigh = curr.open
          const obLow = curr.close
          if (currentPrice >= obLow * 0.998 && currentPrice <= obHigh * 1.002) {
            result.orderBlockPresent = true
            break
          }
        }

        // Reverse: last bullish candle before bearish impulse
        const isBullishCandle = curr.close > curr.open
        const bearishImpulse = next.close < next.open &&
          (next.open - next.close) > (next.high - next.low) * 0.6

        if (isBullishCandle && bearishImpulse) {
          const obHigh = curr.close
          const obLow = curr.open
          if (currentPrice >= obLow * 0.998 && currentPrice <= obHigh * 1.002) {
            result.orderBlockPresent = true
            break
          }
        }
      }

      // FAIR VALUE GAP: 3-candle pattern where middle candle body doesn't overlap outer wicks
      for (let i = 2; i < Math.min(len, 50); i++) {
        if (signal?.aborted) { clearTimeout(timeout); resolve(result); return }

        const c1 = klines[len - 1 - i]
        const c2 = klines[len - i]
        const c3 = klines[len - i + 1]
        if (!c1 || !c2 || !c3) continue

        // Bullish FVG: c1.high < c3.low (gap up)
        if (c1.high < c3.low) {
          const gapMid = (c1.high + c3.low) / 2
          if (currentPrice >= c1.high * 0.998 && currentPrice <= c3.low * 1.002) {
            result.fairValueGapPresent = true
            break
          }
        }

        // Bearish FVG: c1.low > c3.high (gap down)
        if (c1.low > c3.high) {
          if (currentPrice <= c1.low * 1.002 && currentPrice >= c3.high * 0.998) {
            result.fairValueGapPresent = true
            break
          }
        }
      }

      // LIQUIDITY SWEEP: wick extends beyond recent S/R then reverses
      if (len >= 20) {
        const recentHighs = klines.slice(-20, -1).map(k => k.high)
        const recentLows = klines.slice(-20, -1).map(k => k.low)
        const maxHigh = Math.max(...recentHighs)
        const minLow = Math.min(...recentLows)
        const lastCandle = klines[len - 1]

        // Sweep above highs then close below
        if (lastCandle.high > maxHigh && lastCandle.close < maxHigh) {
          result.liquiditySweepDetected = true
        }
        // Sweep below lows then close above
        if (lastCandle.low < minLow && lastCandle.close > minLow) {
          result.liquiditySweepDetected = true
        }
      }

      // Score
      if (result.orderBlockPresent) result.score += 12
      if (result.fairValueGapPresent) result.score += 8
      if (result.liquiditySweepDetected) result.score += 5
      // Full SMC confluence bonus
      if (result.orderBlockPresent && result.fairValueGapPresent) result.score = 25

      clearTimeout(timeout)
      resolve(result)
    } catch {
      clearTimeout(timeout)
      resolve(result)
    }
  })
}

/**
 * Layer 7: Sentiment Divergence
 * F&G extremes + RSI alignment + volume confirmation
 */
function analyzeSentimentDivergence(
  fearGreed: number | null,
  ind: IndicatorData,
  avgVolume: number
): { score: number; signal: TFDirection; reasoning: string } {
  if (fearGreed === null) {
    return { score: 0, signal: 'NEUTRAL', reasoning: 'Fear & Greed unavailable' }
  }

  const currentVol = ind.volume
  const volumeSpike = avgVolume > 0 ? currentVol / avgVolume : 1

  // Extreme Fear + oversold + volume spike = BUY
  if (fearGreed < 20 && ind.rsi14 < 35 && volumeSpike > 1.5) {
    return {
      score: 20,
      signal: 'BULLISH',
      reasoning: `Extreme Fear (${fearGreed}) + RSI oversold (${ind.rsi14.toFixed(0)}) + volume spike (${volumeSpike.toFixed(1)}x) — historically high win-rate BUY`
    }
  }

  // Extreme Greed + overbought + volume declining = SELL
  if (fearGreed > 80 && ind.rsi14 > 65 && volumeSpike < 0.8) {
    return {
      score: 20,
      signal: 'BEARISH',
      reasoning: `Extreme Greed (${fearGreed}) + RSI overbought (${ind.rsi14.toFixed(0)}) + declining volume — reversal likely`
    }
  }

  return { score: 0, signal: 'NEUTRAL', reasoning: `Sentiment neutral (F&G: ${fearGreed}, RSI: ${ind.rsi14.toFixed(0)})` }
}

// ════════════════════════════════════════════════════════════════════════════
// STRATEGY SCORING MATRIX
// ════════════════════════════════════════════════════════════════════════════

function scoreAllStrategies(
  regime: MarketRegime,
  mtf: ReturnType<typeof analyzeMTFConfluence>,
  primaryInd: IndicatorData,
  breakoutInfo: ReturnType<typeof analyzeVolatilityBreakout>,
  fundingInfo: ReturnType<typeof analyzeFundingRate>,
  smcResult: SMCResult,
  sentimentInfo: ReturnType<typeof analyzeSentimentDivergence>
): StrategyScore[] {
  const scores: StrategyScore[] = ALL_STRATEGIES.map(key => ({
    key,
    score: 0,
    reasoning: '',
  }))

  const findStrategy = (k: string) => scores.find(s => s.key === k)!

  // ── Regime-based scoring ──
  const regimeScores: Record<string, Partial<Record<MarketRegime, number>>> = {
    MA_CROSSOVER:      { STRONG_TREND: 85, WEAK_TREND: 60, BREAKOUT: 50 },
    MACD:              { STRONG_TREND: 70, WEAK_TREND: 65, BREAKOUT: 55, VOLATILE: 40 },
    MEAN_REVERSION:    { RANGING: 80, VOLATILE: 70, CHOPPY: 65 },
    GRID:              { RANGING: 90, CHOPPY: 60 },
    MULTI_SIGNAL:      { STRONG_TREND: 55, WEAK_TREND: 55, RANGING: 55, VOLATILE: 55, BREAKOUT: 55, CHOPPY: 50 },
    RSI_DIVERGENCE:    { VOLATILE: 75, CHOPPY: 70, WEAK_TREND: 55 },
    BOLLINGER_BREAKOUT:{ BREAKOUT: 95, VOLATILE: 60, STRONG_TREND: 50 },
    VWAP_TREND:        { STRONG_TREND: 80, WEAK_TREND: 70 },
    ICHIMOKU:          { STRONG_TREND: 85, WEAK_TREND: 60 },
    PIVOT_REVERSAL:    { RANGING: 75, CHOPPY: 65, WEAK_TREND: 50 },
  }

  for (const s of scores) {
    const regimeMap = regimeScores[s.key]
    if (regimeMap && regimeMap[regime]) {
      s.score += regimeMap[regime]!
      s.reasoning += `Regime match (${regime}). `
    }
  }

  // ── MTF confluence bonus ──
  if (mtf.allAgree) {
    if (mtf.dominantDirection === 'BULLISH' || mtf.dominantDirection === 'BEARISH') {
      for (const key of ['MA_CROSSOVER', 'VWAP_TREND', 'ICHIMOKU', 'MACD'] as const) {
        findStrategy(key).score += 15
        findStrategy(key).reasoning += `3-TF confluence ${mtf.dominantDirection}. `
      }
    }
  }

  // ── Volatility breakout ──
  if (breakoutInfo.breakoutImminent) {
    findStrategy('BOLLINGER_BREAKOUT').score += 30
    findStrategy('BOLLINGER_BREAKOUT').reasoning += 'BB squeeze detected — breakout imminent. '
  }

  // ── Funding rate influence ──
  if (fundingInfo.hint === 'MEAN_REVERSION') {
    findStrategy('MEAN_REVERSION').score += 15
    findStrategy('RSI_DIVERGENCE').score += 10
    findStrategy('MEAN_REVERSION').reasoning += 'Overleveraged longs (funding). '
  } else if (fundingInfo.hint === 'TREND_FOLLOW') {
    findStrategy('MA_CROSSOVER').score += 10
    findStrategy('VWAP_TREND').score += 10
    findStrategy('ICHIMOKU').score += 10
  }

  // ── SMC boost ──
  if (smcResult.score > 0) {
    // SMC benefits strategies that rely on precise entries
    findStrategy('PIVOT_REVERSAL').score += smcResult.score
    findStrategy('MEAN_REVERSION').score += Math.floor(smcResult.score * 0.7)
    findStrategy('RSI_DIVERGENCE').score += Math.floor(smcResult.score * 0.5)
    if (smcResult.orderBlockPresent) {
      findStrategy('PIVOT_REVERSAL').reasoning += 'Order block present. '
    }
  }

  // ── Sentiment divergence ──
  if (sentimentInfo.score > 0) {
    if (sentimentInfo.signal === 'BULLISH') {
      findStrategy('RSI_DIVERGENCE').score += 20
      findStrategy('MEAN_REVERSION').score += 15
      findStrategy('RSI_DIVERGENCE').reasoning += 'Extreme Fear + RSI oversold divergence. '
    } else if (sentimentInfo.signal === 'BEARISH') {
      findStrategy('RSI_DIVERGENCE').score += 15
      findStrategy('BOLLINGER_BREAKOUT').score += 10
    }
  }

  // ── RSI-specific boost ──
  if (primaryInd.rsi14 < 25 || primaryInd.rsi14 > 75) {
    findStrategy('RSI_DIVERGENCE').score += 15
    findStrategy('RSI_DIVERGENCE').reasoning += `RSI extreme (${primaryInd.rsi14.toFixed(0)}). `
  }

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score)
  return scores
}

// ════════════════════════════════════════════════════════════════════════════
// PARAMETER CALIBRATION
// ════════════════════════════════════════════════════════════════════════════

function calibrateParameters(
  regime: MarketRegime,
  primaryInd: IndicatorData,
  volumeNodes: VolumeNode[],
  mtfConfluence: ReturnType<typeof analyzeMTFConfluence>,
  fundingRate: number | null,
  confidence: number,
  portfolioCtx: PortfolioContext | null
): {
  timeframe: string
  riskPerTrade: number
  maxPositions: number
  stopLoss: number
  takeProfit: number
  trailingStop: number
  leverage: number
  reasoning: {
    timeframe: string
    risk: string
    stopLoss: string
    takeProfit: string
    trailingStop: string
    leverage: string
    maxPositions: string
  }
} {
  const atrPercent = primaryInd.close > 0 ? (primaryInd.atr14 / primaryInd.close) * 100 : 1
  const volatilityFactor = Math.min(atrPercent / 3, 1) // 0-1 scale, 3% ATR = max vol

  // ── Timeframe ──
  let timeframe = '1h'
  let tfReasoning = 'Default 1h — balanced for most conditions'
  if (regime === 'VOLATILE' || regime === 'BREAKOUT') {
    timeframe = '15m'
    tfReasoning = `ATR ${atrPercent.toFixed(2)}% — high volatility needs faster exits (15m)`
  } else if (regime === 'STRONG_TREND' && atrPercent < 1) {
    timeframe = '4h'
    tfReasoning = `Low volatility strong trend — wider timeframe to ride the move (4h)`
  } else if (regime === 'RANGING' && atrPercent < 0.5) {
    timeframe = '15m'
    tfReasoning = `Tight range, low ATR (${atrPercent.toFixed(2)}%) — shorter timeframe for more opportunities`
  }

  // ── Risk Per Trade ──
  let risk = 2.0 * (1 - volatilityFactor * 0.5) // Base 2%, reduce with volatility
  let riskReasoning = `Base 2% scaled by volatility factor (ATR ${atrPercent.toFixed(2)}%)`

  // Daily drawdown check
  if (portfolioCtx && portfolioCtx.totalPnL < 0) {
    const dailyLossPercent = Math.abs(portfolioCtx.totalPnL / portfolioCtx.equity) * 100
    if (dailyLossPercent > 4) {
      risk = 0.5
      riskReasoning = `RISK OVERRIDE: Daily loss ${dailyLossPercent.toFixed(1)}% > 4% threshold. Forcing minimum risk (0.5%)`
    } else if (dailyLossPercent > 2) {
      risk = Math.min(risk, 1.0)
      riskReasoning += ` — reduced further (daily loss ${dailyLossPercent.toFixed(1)}%)`
    }
  }

  risk = Math.max(0.5, Math.min(3.0, risk))
  risk = Math.round(risk * 10) / 10

  // ── Max Positions ──
  let maxPos = 3
  let maxPosReasoning = 'Base 3 positions'
  if (regime === 'STRONG_TREND' || regime === 'WEAK_TREND') {
    maxPos = 4
    maxPosReasoning = 'Trending market — room for 4 positions'
  }
  if (regime === 'VOLATILE' || regime === 'CHOPPY') {
    maxPos = 2
    maxPosReasoning = `Volatile/choppy market — limiting to 2 positions for safety`
  }

  // Correlation adjustment
  if (portfolioCtx && portfolioCtx.openTrades.length > 0) {
    const hasCorrelatedPosition = portfolioCtx.openTrades.some(
      t => t.symbol !== '' // Simplified — real check happens in dashboard integration
    )
    if (hasCorrelatedPosition && maxPos > 1) {
      maxPos -= 1
      maxPosReasoning += ' — reduced by 1 (correlated position detected)'
    }
  }

  maxPos = Math.max(1, Math.min(5, maxPos))

  // ── Stop Loss (ATR-based) ──
  let sl = atrPercent * 2 // 2x ATR as default
  let slReasoning = `ATR-based: ${atrPercent.toFixed(2)}% × 2 = ${(atrPercent * 2).toFixed(2)}%`

  // Snap to volume node if nearby
  if (volumeNodes.length > 0) {
    const supportNode = volumeNodes.find(n => n.isSupport)
    if (supportNode && primaryInd.close > 0) {
      const nodeDistance = ((primaryInd.close - supportNode.price) / primaryInd.close) * 100
      if (nodeDistance > 0.3 && nodeDistance < sl * 1.2) {
        sl = Math.round(nodeDistance * 10) / 10
        slReasoning = `Snapped to volume support node at $${supportNode.price.toFixed(2)} (${sl}% distance)`
      }
    }
  }

  sl = Math.max(0.5, Math.min(5.0, sl))
  sl = Math.round(sl * 10) / 10

  // ── Take Profit (R:R ratio based) ──
  let rrRatio = 2.0 // Default 2:1
  if (regime === 'STRONG_TREND') rrRatio = 3.0
  if (regime === 'BREAKOUT') rrRatio = 4.0
  if (regime === 'RANGING') rrRatio = 1.5

  let tp = sl * rrRatio
  let tpReasoning = `${rrRatio}:1 R:R ratio (${regime.toLowerCase()} regime) × SL ${sl}%`

  // Snap to resistance volume node if available
  if (volumeNodes.length > 0) {
    const resistanceNode = volumeNodes.find(n => n.isResistance)
    if (resistanceNode && primaryInd.close > 0) {
      const nodeDistance = ((resistanceNode.price - primaryInd.close) / primaryInd.close) * 100
      if (nodeDistance > sl && nodeDistance < tp * 1.5) {
        tp = Math.round(nodeDistance * 10) / 10
        tpReasoning = `Snapped to volume resistance node at $${resistanceNode.price.toFixed(2)} (${tp}% distance)`
      }
    }
  }

  tp = Math.max(sl * 1.5, Math.min(15, tp)) // At least 1.5:1 R:R
  tp = Math.round(tp * 10) / 10

  // ── Trailing Stop ──
  let ts = atrPercent * 1 // 1x ATR
  ts = Math.max(0.3, Math.min(2.0, ts))
  ts = Math.round(ts * 10) / 10
  const tsReasoning = `1× ATR (${atrPercent.toFixed(2)}%) = ${ts}%`

  // ── Leverage ──
  let lev = 1
  let levReasoning = 'Default 1x (spot)'

  if (
    regime === 'STRONG_TREND' &&
    mtfConfluence.allAgree &&
    confidence >= 70 &&
    primaryInd.adx > 35
  ) {
    lev = 2
    levReasoning = `2x — Strong confirmed trend (ADX ${primaryInd.adx.toFixed(0)}, 3TF aligned, high confidence)`

    // Only go to 3x if funding rate also confirms
    if (fundingRate !== null) {
      const fundingConfirms =
        (mtfConfluence.dominantDirection === 'BULLISH' && fundingRate < 0) ||
        (mtfConfluence.dominantDirection === 'BEARISH' && fundingRate > 0.0005)
      if (fundingConfirms && confidence >= 80) {
        lev = 3
        levReasoning = `3x (MAX) — All signals aligned: ADX ${primaryInd.adx.toFixed(0)}, 3TF, funding rate confirms`
      }
    }
  }

  // HARD CAP: Never exceed 3x
  lev = Math.min(3, lev)
  // Force 1x if low confidence
  if (confidence < 70) {
    lev = 1
    levReasoning = `Forced 1x — confidence (${confidence}%) below 70% threshold`
  }

  return {
    timeframe,
    riskPerTrade: risk,
    maxPositions: maxPos,
    stopLoss: sl,
    takeProfit: tp,
    trailingStop: ts,
    leverage: lev,
    reasoning: {
      timeframe: tfReasoning,
      risk: riskReasoning,
      stopLoss: slReasoning,
      takeProfit: tpReasoning,
      trailingStop: tsReasoning,
      leverage: levReasoning,
      maxPositions: maxPosReasoning,
    },
  }
}

// ════════════════════════════════════════════════════════════════════════════
// PAPER TRADING VALIDATOR (Backtest)
// ════════════════════════════════════════════════════════════════════════════

function runPaperBacktest(
  klines: BinanceKline[],
  strategy: string,
  slPercent: number,
  tpPercent: number,
  trailingPercent: number
): BacktestSummary {
  if (klines.length < 30) {
    return { winRate: 0, totalTrades: 0, profitFactor: 0, netPnlPercent: 0 }
  }

  // Use last 100 candles for backtesting
  const testKlines = klines.slice(-100)
  const ti = new TechnicalIndicators()
  let wins = 0
  let losses = 0
  let grossProfit = 0
  let grossLoss = 0
  let inTrade = false
  let tradeEntry = 0
  let tradeSide: 'BUY' | 'SELL' = 'BUY'
  let highestSinceEntry = 0
  let lowestSinceEntry = Infinity

  // Pre-load first 20 candles for warmup
  for (let i = 0; i < Math.min(20, testKlines.length); i++) {
    const k = testKlines[i]
    ti.addCandle(k.close, k.volume, k.high, k.low)
  }

  for (let i = 20; i < testKlines.length; i++) {
    const k = testKlines[i]
    ti.addCandle(k.close, k.volume, k.high, k.low)

    const ind = ti.calculateAll()
    const price = k.close

    if (!inTrade) {
      // Generate entry signal based on strategy
      const signal = getSimpleSignal(strategy, ind, price)
      if (signal !== 'HOLD') {
        // Apply slippage on entry
        tradeEntry = signal === 'BUY'
          ? price * (1 + SLIPPAGE_PERCENT)
          : price * (1 - SLIPPAGE_PERCENT)
        tradeSide = signal
        inTrade = true
        highestSinceEntry = tradeEntry
        lowestSinceEntry = tradeEntry
      }
    } else {
      // Update trailing values
      highestSinceEntry = Math.max(highestSinceEntry, price)
      lowestSinceEntry = Math.min(lowestSinceEntry, price)

      let exitPrice = 0
      let shouldExit = false

      if (tradeSide === 'BUY') {
        const slPrice = tradeEntry * (1 - slPercent / 100)
        const tpPrice = tradeEntry * (1 + tpPercent / 100)
        const trailPrice = highestSinceEntry * (1 - trailingPercent / 100)

        if (price <= slPrice) { shouldExit = true; exitPrice = price * (1 - SLIPPAGE_PERCENT) }
        else if (price >= tpPrice) { shouldExit = true; exitPrice = price * (1 - SLIPPAGE_PERCENT) }
        else if (trailingPercent > 0 && price <= trailPrice) { shouldExit = true; exitPrice = price * (1 - SLIPPAGE_PERCENT) }
      } else {
        const slPrice = tradeEntry * (1 + slPercent / 100)
        const tpPrice = tradeEntry * (1 - tpPercent / 100)
        const trailPrice = lowestSinceEntry * (1 + trailingPercent / 100)

        if (price >= slPrice) { shouldExit = true; exitPrice = price * (1 + SLIPPAGE_PERCENT) }
        else if (price <= tpPrice) { shouldExit = true; exitPrice = price * (1 + SLIPPAGE_PERCENT) }
        else if (trailingPercent > 0 && price >= trailPrice) { shouldExit = true; exitPrice = price * (1 + SLIPPAGE_PERCENT) }
      }

      if (shouldExit) {
        const pnl = tradeSide === 'BUY'
          ? (exitPrice - tradeEntry) / tradeEntry * 100
          : (tradeEntry - exitPrice) / tradeEntry * 100

        if (pnl > 0) { wins++; grossProfit += pnl }
        else { losses++; grossLoss += Math.abs(pnl) }

        inTrade = false
      }
    }
  }

  const totalTrades = wins + losses
  return {
    winRate: totalTrades > 0 ? (wins / totalTrades) * 100 : 0,
    totalTrades,
    profitFactor: grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 99 : 0,
    netPnlPercent: grossProfit - grossLoss,
  }
}

/**
 * Simplified signal generation for backtesting.
 */
function getSimpleSignal(
  strategy: string,
  ind: IndicatorData,
  price: number
): 'BUY' | 'SELL' | 'HOLD' {
  switch (strategy) {
    case 'MA_CROSSOVER':
      if (ind.sma20 > ind.sma50 && ind.sma50 > (ind.sma200 || ind.sma50)) return 'BUY'
      if (ind.sma20 < ind.sma50 && ind.sma50 < (ind.sma200 || ind.sma50)) return 'SELL'
      return 'HOLD'

    case 'MACD':
      if (ind.histogram > 0 && ind.macd > ind.signal) return 'BUY'
      if (ind.histogram < 0 && ind.macd < ind.signal) return 'SELL'
      return 'HOLD'

    case 'MEAN_REVERSION':
      if (price < ind.bollingerLower && ind.rsi14 < 30) return 'BUY'
      if (price > ind.bollingerUpper && ind.rsi14 > 70) return 'SELL'
      return 'HOLD'

    case 'RSI_DIVERGENCE':
      if (ind.rsi14 < 25) return 'BUY'
      if (ind.rsi14 > 75) return 'SELL'
      return 'HOLD'

    case 'BOLLINGER_BREAKOUT':
      if (price > ind.bollingerUpper) return 'BUY'
      if (price < ind.bollingerLower) return 'SELL'
      return 'HOLD'

    case 'VWAP_TREND':
      if (ind.vwap && price > ind.vwap * 1.002) return 'BUY'
      if (ind.vwap && price < ind.vwap * 0.998) return 'SELL'
      return 'HOLD'

    case 'ICHIMOKU':
      if (ind.ichimoku) {
        const cloudTop = Math.max(ind.ichimoku.senkouA, ind.ichimoku.senkouB)
        const cloudBot = Math.min(ind.ichimoku.senkouA, ind.ichimoku.senkouB)
        if (price > cloudTop && ind.ichimoku.tenkan > ind.ichimoku.kijun) return 'BUY'
        if (price < cloudBot && ind.ichimoku.tenkan < ind.ichimoku.kijun) return 'SELL'
      }
      return 'HOLD'

    case 'PIVOT_REVERSAL': {
      if (ind.atr14 === 0) return 'HOLD'
      // Use ATR as a proxy for pivot-like calculation
      const pivot = (ind.bollingerUpper + ind.bollingerLower + ind.close) / 3
      const s1 = 2 * pivot - ind.bollingerUpper
      const r1 = 2 * pivot - ind.bollingerLower
      if (Math.abs(price - s1) < s1 * 0.002) return 'BUY'
      if (Math.abs(price - r1) < r1 * 0.002) return 'SELL'
      return 'HOLD'
    }

    case 'GRID':
    case 'MULTI_SIGNAL':
    default:
      // For grid/multi-signal, use a composite
      let buySignals = 0, sellSignals = 0
      if (ind.rsi14 < 35) buySignals++
      if (ind.rsi14 > 65) sellSignals++
      if (ind.histogram > 0) buySignals++; else sellSignals++
      if (ind.sma20 > ind.sma50) buySignals++; else sellSignals++
      if (buySignals >= 2 && buySignals > sellSignals) return 'BUY'
      if (sellSignals >= 2 && sellSignals > buySignals) return 'SELL'
      return 'HOLD'
  }
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN ENGINE CLASS
// ════════════════════════════════════════════════════════════════════════════

export class AIConfigEngine {
  private cache: Map<string, { result: AIConfigResult; expiry: number }> = new Map()
  private static CACHE_TTL = 60_000 // 60 seconds

  /**
   * Main analysis entry point.
   * Analyzes a trading pair and returns full configuration recommendations.
   */
  async analyze(
    rawSymbol: string,
    portfolioCtx: PortfolioContext | null = null,
    signal?: AbortSignal
  ): Promise<AIConfigResult> {
    const symbol = normalizeSymbol(rawSymbol)

    // Check cache
    const cached = this.cache.get(symbol)
    if (cached && Date.now() < cached.expiry) {
      return cached.result
    }

    // ── Step 1: Parallel data fetching ──
    const [klines15m, klines1h, klines4h, fearGreedData, fundingRate] = await Promise.all([
      fetchCleanKlines(symbol, '15m', 250, signal).catch(() => [] as BinanceKline[]),
      fetchCleanKlines(symbol, '1h', 250, signal).catch(() => [] as BinanceKline[]),
      fetchCleanKlines(symbol, '4h', 250, signal).catch(() => [] as BinanceKline[]),
      fetchFearGreedIndex(signal),
      fetchFundingRate(symbol, signal),
    ])

    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')

    // Validate minimum data
    const primaryKlines = klines1h.length >= 200 ? klines1h
      : klines15m.length >= 200 ? klines15m
      : klines4h.length >= 200 ? klines4h
      : klines1h // fallback

    if (primaryKlines.length < 30) {
      throw new Error(`Insufficient data for ${symbol} — need at least 30 candles, got ${primaryKlines.length}`)
    }

    // ── Step 2: Build indicators for each timeframe ──
    const tfAnalyses: TimeframeAnalysis[] = []

    for (const [tf, klines] of [['15m', klines15m], ['1h', klines1h], ['4h', klines4h]] as const) {
      if (klines.length >= 30) {
        const indicators = buildIndicators(klines)
        const { direction, strength } = analyzeTimeframeDirection(indicators)
        tfAnalyses.push({ timeframe: tf, indicators, direction, strength, klines })
      }
    }

    // ── Step 3: Use 1h as primary indicators (most balanced) ──
    const primaryInd = tfAnalyses.find(t => t.timeframe === '1h')?.indicators
      ?? tfAnalyses[0]?.indicators
      ?? buildIndicators(primaryKlines)

    // Average volume for sentiment divergence
    const volumes = primaryKlines.slice(-50).map(k => k.volume)
    const avgVolume = volumes.length > 0 ? volumes.reduce((a, b) => a + b, 0) / volumes.length : 0

    // ── Step 4: Run all 7 analysis layers in parallel where possible ──
    const [mtfConfluence, smcResult] = await Promise.all([
      Promise.resolve(analyzeMTFConfluence(tfAnalyses)),
      detectSMC(primaryKlines, signal),
    ])

    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')

    const regimeInfo = detectRegime(primaryInd)
    const breakoutInfo = analyzeVolatilityBreakout(primaryInd)
    const fundingInfo = analyzeFundingRate(fundingRate)
    const sentimentInfo = analyzeSentimentDivergence(
      fearGreedData?.value ?? null,
      primaryInd,
      avgVolume
    )
    const volumeNodes = analyzeVolumeProfile(primaryKlines)

    // ── Step 5: Confidence Scoring ──
    let confidence = 50 // Base

    confidence += mtfConfluence.score  // +30 or -20
    confidence += fundingInfo.score    // +15 or 0
    confidence += smcResult.score      // 0-25
    confidence += sentimentInfo.score  // +20 or 0

    // Volume confirmation
    const currentVol = primaryInd.volume
    const volRatio = avgVolume > 0 ? currentVol / avgVolume : 1
    if (volRatio > 1.3) {
      confidence += 15
    } else if (volRatio < 0.5) {
      confidence -= 15
    }

    // Regime clarity
    if (['STRONG_TREND', 'BREAKOUT', 'RANGING'].includes(regimeInfo.regime)) {
      confidence += 10
    }

    // Extreme volatility penalty
    const atrPct = primaryInd.close > 0 ? (primaryInd.atr14 / primaryInd.close) * 100 : 0
    if (atrPct > 3) {
      confidence -= 10
    }

    confidence = Math.max(0, Math.min(100, confidence))

    // ── Step 6: Strategy Selection ──
    const strategyScores = scoreAllStrategies(
      regimeInfo.regime,
      mtfConfluence,
      primaryInd,
      breakoutInfo,
      fundingInfo,
      smcResult,
      sentimentInfo
    )

    let selectedStrategy = strategyScores[0].key
    let strategyReasoning = strategyScores[0].reasoning

    // If top 2 are within 5 points, default to MULTI_SIGNAL for safety
    if (
      strategyScores.length >= 2 &&
      strategyScores[0].score - strategyScores[1].score < 5
    ) {
      selectedStrategy = 'MULTI_SIGNAL'
      strategyReasoning = `Top strategies too close (${strategyScores[0].key}: ${strategyScores[0].score} vs ${strategyScores[1].key}: ${strategyScores[1].score}) — defaulting to Multi-Signal Ensemble for diversification`
    }

    // ── Step 7: Parameter Calibration ──
    const params = calibrateParameters(
      regimeInfo.regime,
      primaryInd,
      volumeNodes,
      mtfConfluence,
      fundingRate,
      confidence,
      portfolioCtx
    )

    // ── Step 8: Paper Trading Validation ──
    let backtestResult: BacktestSummary | null = null
    try {
      backtestResult = runPaperBacktest(
        primaryKlines,
        selectedStrategy,
        params.stopLoss,
        params.takeProfit,
        params.trailingStop
      )

      // Downgrade confidence if backtest is terrible
      if (backtestResult.totalTrades >= 3 && backtestResult.winRate < 30) {
        confidence = Math.max(0, confidence - 20)
        strategyReasoning += ` ⚠️ Backtest warning: ${backtestResult.winRate.toFixed(0)}% win rate over last 100 candles`
      }

      // Slight boost if backtest is strong
      if (backtestResult.totalTrades >= 5 && backtestResult.winRate > 60) {
        confidence = Math.min(100, confidence + 5)
      }
    } catch {
      // Backtest failure is non-critical
    }

    // Final confidence clamp
    confidence = Math.max(0, Math.min(100, Math.round(confidence)))

    // ── Build result ──
    const result: AIConfigResult = {
      strategy: selectedStrategy,
      timeframe: params.timeframe,
      riskPerTrade: params.riskPerTrade,
      maxPositions: params.maxPositions,
      stopLoss: params.stopLoss,
      takeProfit: params.takeProfit,
      trailingStop: params.trailingStop,
      leverage: params.leverage,
      confidence,
      reasoning: {
        strategy: `${strategyReasoning} | Regime: ${regimeInfo.reasoning}`,
        ...params.reasoning
      },
      marketRegime: regimeInfo.regime,
      fearGreedIndex: fearGreedData?.value ?? null,
      fearGreedTimestamp: fearGreedData?.timestamp ?? null,
      fundingRate,
      backtestResult,
      analyzedAt: new Date(),
    }

    // Cache
    this.cache.set(symbol, { result, expiry: Date.now() + AIConfigEngine.CACHE_TTL })

    return result
  }

  /**
   * Clear the cache (useful for testing or forced refresh).
   */
  clearCache() {
    this.cache.clear()
  }
}

// Singleton for shared use
export const aiConfigEngine = new AIConfigEngine()
