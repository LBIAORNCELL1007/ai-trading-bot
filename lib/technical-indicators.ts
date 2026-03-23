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
  atr14: number;
  volume: number;
  vwap?: number;
  ichimoku?: {
    tenkan: number;
    kijun: number;
    senkouA: number;
    senkouB: number;
  };
  adx: number;
  close: number;
}

export class TechnicalIndicators {
  private prices: number[] = [];
  private volumes: number[] = [];
  private highs: number[] = [];
  private lows: number[] = [];
  private maxSize: number = 500;

  addCandle(close: number, volume: number = 0, high: number = close, low: number = close) {
    this.prices.push(close);
    this.volumes.push(volume);
    this.highs.push(high);
    this.lows.push(low);

    if (this.prices.length > this.maxSize) {
      this.prices.shift();
      this.volumes.shift();
      this.highs.shift();
      this.lows.shift();
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

  calculateAll(): IndicatorData {
    const len = this.prices.length;

    // Moving Averages
    const sma20 = this.sma(20);
    const sma50 = this.sma(50);
    const sma200 = this.sma(200);

    // EMA for MACD
    const ema12 = this.ema(12);
    const ema26 = this.ema(26);

    // MACD
    const macd = ema12 - ema26;
    const signalLine = this.emaOfArray(9, this.getMACDArray());
    const histogram = macd - signalLine;

    // RSI
    const rsi14 = this.calculateRSI(14);

    // Bollinger Bands
    const { upper, middle, lower } = this.calculateBollingerBands(20);

    // ATR
    const atr14 = this.calculateATR(14);

    // VWAP
    const vwap = this.calculateVWAP(50); // Rolling 50

    // Ichimoku
    const ichimoku = this.calculateIchimoku();

    return {
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
      atr14,
      volume: this.volumes[this.volumes.length - 1] || 0,
      vwap,
      ichimoku,
      adx: this.calculateADX(14),
      close: this.prices[len - 1] || 0,
    };
  }

  private getMACDArray(): number[] {
    const macdArray: number[] = [];
    for (let i = 25; i < this.prices.length; i++) {
      const ema12 = this.ema(12, this.prices.slice(0, i + 1));
      const ema26 = this.ema(26, this.prices.slice(0, i + 1));
      macdArray.push(ema12 - ema26);
    }
    return macdArray;
  }

  private emaOfArray(period: number, arr: number[]): number {
    if (arr.length < period) return 0;

    let emaValue = arr.slice(0, period).reduce((a, b) => a + b, 0) / period;
    const multiplier = 2 / (period + 1);

    for (let i = period; i < arr.length; i++) {
      emaValue = arr[i] * multiplier + emaValue * (1 - multiplier);
    }

    return emaValue;
  }

  private calculateRSI(period: number): number {
    if (this.prices.length < period + 1) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = this.prices.length - period; i < this.prices.length; i++) {
      const change = this.prices[i] - this.prices[i - 1];
      if (change > 0) gains += change;
      else losses += Math.abs(change);
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - 100 / (1 + rs);
  }

  private calculateBollingerBands(period: number) {
    const sma = this.sma(period);
    if (this.prices.length < period) {
      return { upper: sma, middle: sma, lower: sma };
    }

    const slice = this.prices.slice(-period);
    const squaredDifferences = slice.map((price) => Math.pow(price - sma, 2));
    const variance = squaredDifferences.reduce((a, b) => a + b, 0) / period;
    const stdDev = Math.sqrt(variance);

    return {
      upper: sma + 2 * stdDev,
      middle: sma,
      lower: sma - 2 * stdDev,
    };
  }

  private calculateATR(period: number): number {
    if (this.prices.length < period + 1) return 0;

    const trueRanges: number[] = [];
    for (let i = 1; i < this.prices.length; i++) {
      const high = this.highs[i];
      const low = this.lows[i];
      const prevClose = this.prices[i - 1];

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      trueRanges.push(tr);
    }

    const atr = trueRanges.slice(-period).reduce((a, b) => a + b, 0) / period;

    // Fallback: if ATR is 0 (high==low==close), use close-to-close changes
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

    // Rolling VWAP for last N candles
    const prices = this.prices.slice(-period);
    const volumes = this.volumes.slice(-period);
    const highs = this.highs.slice(-period);
    const lows = this.lows.slice(-period);

    let cumTPV = 0; // Cumulative Typical Price * Vol
    let cumVol = 0;

    for (let i = 0; i < prices.length; i++) {
      const tp = (highs[i] + lows[i] + prices[i]) / 3;
      cumTPV += tp * volumes[i];
      cumVol += volumes[i];
    }

    return cumVol === 0 ? prices[prices.length - 1] : cumTPV / cumVol;
  }

  private calculateIchimoku() {
    if (this.prices.length < 52) return undefined;

    const sliceHigh = (p: number) => Math.max(...this.highs.slice(-p));
    const sliceLow = (p: number) => Math.min(...this.lows.slice(-p));

    const tenkan = (sliceHigh(9) + sliceLow(9)) / 2;
    const kijun = (sliceHigh(26) + sliceLow(26)) / 2;

    // Senkou A (shifted forward 26 - theoretical)
    const senkouA = (tenkan + kijun) / 2;

    // Senkou B
    const senkouB = (sliceHigh(52) + sliceLow(52)) / 2;

    return { tenkan, kijun, senkouA, senkouB };
  }

  getCurrentPrice(): number {
    return this.prices[this.prices.length - 1] || 0;
  }

  getHighestPrice(period: number): number {
    if (this.prices.length < period) return Math.max(...this.prices);
    return Math.max(...this.prices.slice(-period));
  }

  getLowestPrice(period: number): number {
    if (this.prices.length < period) return Math.min(...this.prices);
    return Math.min(...this.prices.slice(-period));
  }

  getPriceArray(): number[] {
    return this.prices;
  }

  getVolumeArray(): number[] {
    return this.volumes;
  }

  private calculateADX(period: number): number {
    if (this.prices.length < period * 2) return 0;

    const tr: number[] = [];
    const plusDM: number[] = [];
    const minusDM: number[] = [];

    // 1. Calculate TR, +DM, -DM
    for (let i = 1; i < this.prices.length; i++) {
      const high = this.highs[i];
      const low = this.lows[i];
      const prevClose = this.prices[i - 1];
      const prevHigh = this.highs[i - 1];
      const prevLow = this.lows[i - 1];

      const mTR = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      tr.push(mTR);

      const upMove = high - prevHigh;
      const downMove = prevLow - low;

      if (upMove > downMove && upMove > 0) plusDM.push(upMove);
      else plusDM.push(0);

      if (downMove > upMove && downMove > 0) minusDM.push(downMove);
      else minusDM.push(0);
    }

    // 2. Smooth TR, +DM, -DM (Wilder's Smoothing)
    // First value is simple sum
    let smoothTR = tr.slice(0, period).reduce((a, b) => a + b, 0);
    let smoothPlusDM = plusDM.slice(0, period).reduce((a, b) => a + b, 0);
    let smoothMinusDM = minusDM.slice(0, period).reduce((a, b) => a + b, 0);

    const dxList: number[] = [];

    // Calculate subsequent values
    for (let i = period; i < tr.length; i++) {
      smoothTR = smoothTR - (smoothTR / period) + tr[i];
      smoothPlusDM = smoothPlusDM - (smoothPlusDM / period) + plusDM[i];
      smoothMinusDM = smoothMinusDM - (smoothMinusDM / period) + minusDM[i];

      const diPlus = (smoothPlusDM / smoothTR) * 100;
      const diMinus = (smoothMinusDM / smoothTR) * 100;

      const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;

      // Handle division by zero
      if (isNaN(dx)) dxList.push(0);
      else dxList.push(dx);
    }

    // 3. ADX = SMA of DX
    if (dxList.length < period) return 0;

    // Average the last 'period' DX values
    // Wilder actually smoothes ADX too, but simple average is common approx
    return dxList.slice(-period).reduce((a, b) => a + b, 0) / period;
  }
}
