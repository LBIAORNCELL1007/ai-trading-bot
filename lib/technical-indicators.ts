export interface IndicatorData {
  sma20: number;
  sma50: number;
  sma200: number;
  ema12: number;
  ema26: number;
  macd: number;
  signal: number;
  histogram: number;
  rsi14: number;
  bollingerUpper: number;
  bollingerMiddle: number;
  bollingerLower: number;
  bollingerWidthPct: number;       // (upper-lower)/middle*100
  bollingerWidthPctile?: number;   // percentile of bbWidthPct over recent N bars (squeeze detector)
  atr14: number;
  atrPercent: number;              // atr14 / close * 100
  volume: number;
  avgVolume20: number;             // SMA(volume, 20) for surge detection
  volumeRatio: number;             // volume / avgVolume20
  vwap?: number;
  ichimoku?: {
    tenkan: number;
    kijun: number;
    senkouA: number;        // current cloud (was projected 26 bars ago)
    senkouB: number;        // current cloud (was projected 26 bars ago)
    senkouAFuture: number;  // newly projected (for next 26 bars)
    senkouBFuture: number;
  };
  adx: number;
  plusDI: number;
  minusDI: number;
  close: number;
  high: number;
  low: number;
  // event-based deltas (vs previous bar's calculated values)
  histogramPrev: number;
  sma20Prev: number;
  sma50Prev: number;
  sma200Prev: number;
  rsi14Prev: number;
  closePrev: number;
}

/**
 * Streaming, event-aware technical indicators.
 *
 * Key correctness improvements over a naive impl:
 *  - RSI uses Wilder's smoothing (not simple-mean) for parity with industry charts.
 *  - MACD signal line is computed once incrementally on a streaming MACD history
 *    (not O(N²) re-computation). Histogram is therefore correct from bar `slow+signal-1`.
 *  - ADX uses Wilder smoothing on TR/+DM/-DM AND on DX (true Wilder ADX, not simple mean).
 *  - Ichimoku exposes BOTH the current cloud (senkouA/B that were projected 26 bars ago)
 *    AND the newly projected cloud — so strategies can compare price to a real, time-shifted
 *    cloud rather than the textbook bug of comparing to the unshifted current Tenkan/Kijun avg.
 *  - Tracks previous-bar values so strategies can detect *cross events* instead of states.
 *  - Provides Bollinger-band width and rolling percentile for squeeze/breakout filtering.
 *  - Provides volume-surge ratio for entry confirmation.
 */
export class TechnicalIndicators {
  private prices: number[] = [];
  private volumes: number[] = [];
  private highs: number[] = [];
  private lows: number[] = [];
  private maxSize: number = 600;

  // Streaming EMA state (rebuilt on demand from prices, but cached per call).
  private macdHistory: number[] = []; // macd line per bar from index slow-1 onwards
  private signalHistory: number[] = []; // signal line per bar from index slow+signal-2 onwards
  private rsiHistory: number[] = []; // Wilder RSI per bar from index period onwards
  private prevAvgGain: number = 0;
  private prevAvgLoss: number = 0;
  private rsiInitialized: boolean = false;

  // Wilder ADX state
  private smoothTR: number = 0;
  private smoothPlusDM: number = 0;
  private smoothMinusDM: number = 0;
  private adxValue: number = 0;
  private dxHistory: number[] = [];

  // Cached indicator snapshot from previous bar (for event/cross detection)
  private prevSnapshot: IndicatorData | null = null;
  private currentSnapshot: IndicatorData | null = null;

  addCandle(close: number, volume: number = 0, high: number = close, low: number = close) {
    // Snapshot current as previous BEFORE adding the new bar
    if (this.currentSnapshot) {
      this.prevSnapshot = this.currentSnapshot;
    }
    this.currentSnapshot = null; // invalidate; next calculateAll will rebuild

    this.prices.push(close);
    this.volumes.push(volume);
    this.highs.push(high);
    this.lows.push(low);

    if (this.prices.length > this.maxSize) {
      this.prices.shift();
      this.volumes.shift();
      this.highs.shift();
      this.lows.shift();
      // The streaming Wilder/EMA states use only the most recent value, but trimming
      // arrays does NOT change their internal "memory". macdHistory/rsiHistory we
      // keep aligned with `prices` length — also shift.
      if (this.macdHistory.length > this.maxSize) this.macdHistory.shift();
      if (this.signalHistory.length > this.maxSize) this.signalHistory.shift();
      if (this.rsiHistory.length > this.maxSize) this.rsiHistory.shift();
    }
  }

  private sma(period: number, prices: number[] = this.prices): number {
    if (prices.length < period) return 0;
    const slice = prices.slice(-period);
    return slice.reduce((a, b) => a + b, 0) / period;
  }

  private ema(period: number, prices: number[] = this.prices): number {
    if (prices.length < period) return 0;
    let emaValue = this.sma(period, prices.slice(0, period));
    const multiplier = 2 / (period + 1);
    for (let i = period; i < prices.length; i++) {
      emaValue = prices[i] * multiplier + emaValue * (1 - multiplier);
    }
    return emaValue;
  }

  /**
   * Build the full MACD line history from scratch in O(N), then EMA9 of that history
   * for the signal line in O(N).  Replaces the previous O(N²) implementation that
   * silently broke the signal line on early bars.
   */
  private rebuildMACDHistory(fast = 12, slow = 26, signal = 9): void {
    const n = this.prices.length;
    this.macdHistory = [];
    this.signalHistory = [];
    if (n < slow) return;

    // Streaming EMAs across full series
    let emaFast = 0;
    let emaSlow = 0;
    const kFast = 2 / (fast + 1);
    const kSlow = 2 / (slow + 1);

    // Seed both EMAs with their respective SMAs once enough data exists
    for (let i = 0; i < n; i++) {
      const p = this.prices[i];
      if (i === fast - 1) {
        let s = 0;
        for (let j = 0; j < fast; j++) s += this.prices[j];
        emaFast = s / fast;
      } else if (i >= fast) {
        emaFast = p * kFast + emaFast * (1 - kFast);
      }

      if (i === slow - 1) {
        let s = 0;
        for (let j = 0; j < slow; j++) s += this.prices[j];
        emaSlow = s / slow;
        this.macdHistory.push(emaFast - emaSlow);
      } else if (i >= slow) {
        emaSlow = p * kSlow + emaSlow * (1 - kSlow);
        this.macdHistory.push(emaFast - emaSlow);
      }
    }

    // Signal = EMA(signal) of macdHistory
    if (this.macdHistory.length < signal) return;
    const kSig = 2 / (signal + 1);
    let sig = 0;
    for (let j = 0; j < signal; j++) sig += this.macdHistory[j];
    sig /= signal;
    this.signalHistory.push(sig);
    for (let i = signal; i < this.macdHistory.length; i++) {
      sig = this.macdHistory[i] * kSig + sig * (1 - kSig);
      this.signalHistory.push(sig);
    }
  }

  /**
   * Streaming Wilder RSI rebuilt over the full series in O(N).
   * Stored as `rsiHistory` aligned right (last value = current bar RSI).
   */
  private rebuildRSIHistory(period = 14): void {
    const n = this.prices.length;
    this.rsiHistory = [];
    if (n <= period) return;

    let avgGain = 0;
    let avgLoss = 0;

    // Seed with simple averages of the first `period` changes
    for (let i = 1; i <= period; i++) {
      const ch = this.prices[i] - this.prices[i - 1];
      if (ch > 0) avgGain += ch;
      else avgLoss += -ch;
    }
    avgGain /= period;
    avgLoss /= period;

    const rsiAt = (g: number, l: number) => {
      if (l === 0) return g === 0 ? 50 : 100;
      const rs = g / l;
      return 100 - 100 / (1 + rs);
    };

    this.rsiHistory.push(rsiAt(avgGain, avgLoss));

    // Wilder smoothing for the rest
    for (let i = period + 1; i < n; i++) {
      const ch = this.prices[i] - this.prices[i - 1];
      const gain = ch > 0 ? ch : 0;
      const loss = ch < 0 ? -ch : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
      this.rsiHistory.push(rsiAt(avgGain, avgLoss));
    }

    this.prevAvgGain = avgGain;
    this.prevAvgLoss = avgLoss;
    this.rsiInitialized = true;
  }

  calculateAll(): IndicatorData {
    if (this.currentSnapshot) return this.currentSnapshot;

    const len = this.prices.length;

    // Moving Averages
    const sma20 = this.sma(20);
    const sma50 = this.sma(50);
    const sma200 = this.sma(200);

    // EMA for MACD (current bar values)
    const ema12 = this.ema(12);
    const ema26 = this.ema(26);

    // Streaming MACD + signal (correct & fast)
    this.rebuildMACDHistory(12, 26, 9);
    const macd = this.macdHistory.length > 0 ? this.macdHistory[this.macdHistory.length - 1] : 0;
    const signalLine = this.signalHistory.length > 0 ? this.signalHistory[this.signalHistory.length - 1] : 0;
    const histogram = macd - signalLine;

    // Wilder RSI
    this.rebuildRSIHistory(14);
    const rsi14 = this.rsiHistory.length > 0 ? this.rsiHistory[this.rsiHistory.length - 1] : 50;

    // Bollinger Bands + width %
    const { upper, middle, lower } = this.calculateBollingerBands(20);
    const bollingerWidthPct = middle > 0 ? ((upper - lower) / middle) * 100 : 0;
    const bollingerWidthPctile = this.calculateBBWidthPercentile(20, 100);

    // ATR
    const atr14 = this.calculateATR(14);
    const close = this.prices[len - 1] || 0;
    const atrPercent = close > 0 ? (atr14 / close) * 100 : 0;

    // Volume surge
    const avgVolume20 = this.volumes.length >= 20
      ? this.volumes.slice(-20).reduce((a, b) => a + b, 0) / 20
      : (this.volumes.length > 0 ? this.volumes.reduce((a, b) => a + b, 0) / this.volumes.length : 0);
    const currentVolume = this.volumes[this.volumes.length - 1] || 0;
    const volumeRatio = avgVolume20 > 0 ? currentVolume / avgVolume20 : 1;

    // VWAP
    const vwap = this.calculateVWAP(50);

    // Ichimoku (with proper time shift)
    const ichimoku = this.calculateIchimoku();

    // ADX (Wilder) — compute fresh per-call from full history (still O(N), but correct)
    const { adx, plusDI, minusDI } = this.calculateADXFull(14);

    const snap: IndicatorData = {
      sma20,
      sma50,
      sma200,
      ema12,
      ema26,
      macd,
      signal: signalLine,
      histogram,
      rsi14,
      bollingerUpper: upper,
      bollingerMiddle: middle,
      bollingerLower: lower,
      bollingerWidthPct,
      bollingerWidthPctile,
      atr14,
      atrPercent,
      volume: currentVolume,
      avgVolume20,
      volumeRatio,
      vwap,
      ichimoku,
      adx,
      plusDI,
      minusDI,
      close,
      high: this.highs[len - 1] || 0,
      low: this.lows[len - 1] || 0,
      // Previous bar tracked deltas
      histogramPrev: this.prevSnapshot?.histogram ?? histogram,
      sma20Prev: this.prevSnapshot?.sma20 ?? sma20,
      sma50Prev: this.prevSnapshot?.sma50 ?? sma50,
      sma200Prev: this.prevSnapshot?.sma200 ?? sma200,
      rsi14Prev: this.prevSnapshot?.rsi14 ?? rsi14,
      closePrev: this.prevSnapshot?.close ?? close,
    };

    this.currentSnapshot = snap;
    return snap;
  }

  private calculateBollingerBands(period: number) {
    const sma = this.sma(period);
    if (this.prices.length < period) {
      return { upper: sma, middle: sma, lower: sma };
    }
    const slice = this.prices.slice(-period);
    const variance = slice.reduce((a, p) => a + (p - sma) * (p - sma), 0) / period;
    const stdDev = Math.sqrt(variance);
    return { upper: sma + 2 * stdDev, middle: sma, lower: sma - 2 * stdDev };
  }

  /** Rolling percentile of BB-width (for squeeze detection). */
  private calculateBBWidthPercentile(period: number, lookback: number): number {
    if (this.prices.length < period + lookback) return 50;
    const widths: number[] = [];
    for (let i = this.prices.length - lookback; i < this.prices.length; i++) {
      const slice = this.prices.slice(Math.max(0, i - period + 1), i + 1);
      if (slice.length < period) continue;
      const m = slice.reduce((a, b) => a + b, 0) / period;
      const v = slice.reduce((a, p) => a + (p - m) * (p - m), 0) / period;
      const sd = Math.sqrt(v);
      widths.push(m > 0 ? (4 * sd / m) * 100 : 0);
    }
    if (widths.length === 0) return 50;
    const current = widths[widths.length - 1];
    const sorted = [...widths].sort((a, b) => a - b);
    const rank = sorted.findIndex(w => w >= current);
    return (rank / widths.length) * 100;
  }

  private calculateATR(period: number): number {
    if (this.prices.length < period + 1) return 0;
    const trueRanges: number[] = [];
    for (let i = 1; i < this.prices.length; i++) {
      const high = this.highs[i];
      const low = this.lows[i];
      const prevClose = this.prices[i - 1];
      const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
      trueRanges.push(tr);
    }
    // Wilder smoothing
    let atr = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
    for (let i = period; i < trueRanges.length; i++) {
      atr = (atr * (period - 1) + trueRanges[i]) / period;
    }
    if (atr === 0 && this.prices.length > period) {
      const recentPrices = this.prices.slice(-period - 1);
      let sumChanges = 0;
      for (let i = 1; i < recentPrices.length; i++) {
        sumChanges += Math.abs(recentPrices[i] - recentPrices[i - 1]);
      }
      return sumChanges / period;
    }
    return atr;
  }

  private calculateVWAP(period: number): number {
    if (this.prices.length < period) return this.prices[this.prices.length - 1];
    const prices = this.prices.slice(-period);
    const volumes = this.volumes.slice(-period);
    const highs = this.highs.slice(-period);
    const lows = this.lows.slice(-period);
    let cumTPV = 0;
    let cumVol = 0;
    for (let i = 0; i < prices.length; i++) {
      const tp = (highs[i] + lows[i] + prices[i]) / 3;
      cumTPV += tp * volumes[i];
      cumVol += volumes[i];
    }
    return cumVol === 0 ? prices[prices.length - 1] : cumTPV / cumVol;
  }

  /**
   * Ichimoku Cloud — properly time-shifted.
   *
   *  - Tenkan = (high9 + low9) / 2 at *current* bar
   *  - Kijun  = (high26 + low26) / 2 at *current* bar
   *  - Senkou A *current* (cloud at price=now) = the (Tenkan+Kijun)/2 from 26 bars ago
   *  - Senkou B *current* = the (high52+low52)/2 from 26 bars ago
   *  - Senkou A/B *future* = newly computed values projected 26 bars forward
   *
   * Strategies should compare current price to senkouA / senkouB (the *current* cloud),
   * not the future projections.
   */
  private calculateIchimoku() {
    if (this.prices.length < 52 + 26) return undefined;

    const sliceHighAt = (p: number, endIdx: number) => {
      let m = -Infinity;
      const start = Math.max(0, endIdx - p + 1);
      for (let i = start; i <= endIdx; i++) if (this.highs[i] > m) m = this.highs[i];
      return m;
    };
    const sliceLowAt = (p: number, endIdx: number) => {
      let m = Infinity;
      const start = Math.max(0, endIdx - p + 1);
      for (let i = start; i <= endIdx; i++) if (this.lows[i] < m) m = this.lows[i];
      return m;
    };

    const lastIdx = this.prices.length - 1;
    const tenkan = (sliceHighAt(9, lastIdx) + sliceLowAt(9, lastIdx)) / 2;
    const kijun = (sliceHighAt(26, lastIdx) + sliceLowAt(26, lastIdx)) / 2;

    // FUTURE cloud (newly projected at lastIdx, plotted at lastIdx+26)
    const senkouAFuture = (tenkan + kijun) / 2;
    const senkouBFuture = (sliceHighAt(52, lastIdx) + sliceLowAt(52, lastIdx)) / 2;

    // CURRENT cloud (= future cloud computed 26 bars ago)
    const pastIdx = lastIdx - 26;
    const tenkanPast = (sliceHighAt(9, pastIdx) + sliceLowAt(9, pastIdx)) / 2;
    const kijunPast = (sliceHighAt(26, pastIdx) + sliceLowAt(26, pastIdx)) / 2;
    const senkouA = (tenkanPast + kijunPast) / 2;
    const senkouB = (sliceHighAt(52, pastIdx) + sliceLowAt(52, pastIdx)) / 2;

    return { tenkan, kijun, senkouA, senkouB, senkouAFuture, senkouBFuture };
  }

  /** Wilder ADX over the full series (correct, simple-mean DX replaced with Wilder smoothing). */
  private calculateADXFull(period: number): { adx: number; plusDI: number; minusDI: number } {
    const n = this.prices.length;
    if (n < period * 2 + 1) return { adx: 0, plusDI: 0, minusDI: 0 };

    const tr: number[] = [];
    const plusDM: number[] = [];
    const minusDM: number[] = [];
    for (let i = 1; i < n; i++) {
      const high = this.highs[i];
      const low = this.lows[i];
      const prevClose = this.prices[i - 1];
      const prevHigh = this.highs[i - 1];
      const prevLow = this.lows[i - 1];
      const m = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
      tr.push(m);
      const upMove = high - prevHigh;
      const downMove = prevLow - low;
      plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
      minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
    }

    // Seed Wilder sums with first `period` values
    let smTR = tr.slice(0, period).reduce((a, b) => a + b, 0);
    let smP = plusDM.slice(0, period).reduce((a, b) => a + b, 0);
    let smM = minusDM.slice(0, period).reduce((a, b) => a + b, 0);

    const dxList: number[] = [];
    let lastPlusDI = 0;
    let lastMinusDI = 0;
    for (let i = period; i < tr.length; i++) {
      smTR = smTR - smTR / period + tr[i];
      smP = smP - smP / period + plusDM[i];
      smM = smM - smM / period + minusDM[i];
      const diP = smTR > 0 ? (smP / smTR) * 100 : 0;
      const diM = smTR > 0 ? (smM / smTR) * 100 : 0;
      const sumDI = diP + diM;
      const dx = sumDI > 0 ? (Math.abs(diP - diM) / sumDI) * 100 : 0;
      dxList.push(dx);
      lastPlusDI = diP;
      lastMinusDI = diM;
    }

    if (dxList.length < period) return { adx: 0, plusDI: lastPlusDI, minusDI: lastMinusDI };

    // Wilder ADX = Wilder smoothing of DX
    let adx = dxList.slice(0, period).reduce((a, b) => a + b, 0) / period;
    for (let i = period; i < dxList.length; i++) {
      adx = (adx * (period - 1) + dxList[i]) / period;
    }
    return { adx, plusDI: lastPlusDI, minusDI: lastMinusDI };
  }

  getCurrentPrice(): number {
    return this.prices[this.prices.length - 1] || 0;
  }

  getHighestPrice(period: number): number {
    if (this.prices.length < period) return Math.max(...this.highs);
    return Math.max(...this.highs.slice(-period));
  }

  getLowestPrice(period: number): number {
    if (this.prices.length < period) return Math.min(...this.lows);
    return Math.min(...this.lows.slice(-period));
  }

  getPriceArray(): number[] { return this.prices; }
  getVolumeArray(): number[] { return this.volumes; }
  getHighArray(): number[] { return this.highs; }
  getLowArray(): number[] { return this.lows; }

  /** Simple regime classifier — used by strategies to gate entries by environment. */
  classifyRegime(): 'STRONG_TREND' | 'WEAK_TREND' | 'RANGING' | 'VOLATILE' | 'CHOPPY' {
    const ind = this.calculateAll();
    if (ind.adx >= 30 && ind.atrPercent < 4) return 'STRONG_TREND';
    if (ind.adx >= 20 && ind.adx < 30) return 'WEAK_TREND';
    if (ind.atrPercent >= 4) return 'VOLATILE';
    if (ind.adx < 18 && ind.bollingerWidthPct < 2.5) return 'RANGING';
    return 'CHOPPY';
  }
}
