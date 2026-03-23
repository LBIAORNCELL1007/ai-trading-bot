import { TechnicalIndicators, IndicatorData } from './technical-indicators';

export interface StrategySignal {
  strategy: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  indicators: Partial<IndicatorData>;
  reasoning: string;
}

export interface GridTrade {
  gridLevel: number;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  filledBuys: boolean;
  filledSells: boolean;
}

export class TradingStrategy {
  protected indicators: TechnicalIndicators;

  constructor() {
    this.indicators = new TechnicalIndicators();
  }

  addCandle(close: number, volume: number = 0, high: number = close, low: number = close) {
    this.indicators.addCandle(close, volume, high, low);
  }

  // New method to pre-load history
  initialize(klines: any[]) {
    klines.forEach(k => {
      this.addCandle(k.close, k.volume, k.high, k.low);
    });
  }

  public getIndicators(): IndicatorData {
    return this.indicators.calculateAll();
  }
}

/**
 * Moving Average Crossover Strategy
 * Signals based on SMA 20/50/200 crossovers
 */
export class MovingAverageCrossoverStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Golden Cross: SMA 20 crosses above SMA 50
    if (indicators.sma20 > indicators.sma50 && indicators.sma50 > indicators.sma200) {
      action = 'BUY';
      confidence = 0.85;
      reasoning = 'Golden cross detected: SMA20 > SMA50 > SMA200 (uptrend)';
    }
    // Death Cross: SMA 20 crosses below SMA 50
    else if (indicators.sma20 < indicators.sma50 && indicators.sma50 < indicators.sma200) {
      action = 'SELL';
      confidence = 0.85;
      reasoning = 'Death cross detected: SMA20 < SMA50 < SMA200 (downtrend)';
    }
    // Price above 200 SMA but consolidating
    else if (indicators.sma20 > indicators.sma200 && Math.abs(indicators.sma20 - indicators.sma50) < indicators.sma50 * 0.01) {
      action = 'BUY';
      confidence = 0.60;
      reasoning = 'Price above 200 SMA with consolidation, potential breakout';
    }

    return {
      strategy: 'Moving Average Crossover',
      action,
      confidence,
      indicators: { sma20: indicators.sma20, sma50: indicators.sma50, sma200: indicators.sma200 },
      reasoning,
    };
  }
}

/**
 * MACD Strategy
 * Signals based on MACD line and signal line crossovers
 */
export class MACDStrategy extends TradingStrategy {
  private previousHistogram = 0;

  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Bullish: MACD crosses above signal line
    if (this.previousHistogram < 0 && indicators.histogram > 0) {
      action = 'BUY';
      confidence = 0.75;
      reasoning = 'MACD bullish crossover: MACD crosses above signal line';
    }
    // Bearish: MACD crosses below signal line
    else if (this.previousHistogram > 0 && indicators.histogram < 0) {
      action = 'SELL';
      confidence = 0.75;
      reasoning = 'MACD bearish crossover: MACD crosses below signal line';
    }
    // Divergence: MACD above 0 and increasing
    else if (indicators.histogram > 0 && indicators.histogram > this.previousHistogram) {
      action = 'BUY';
      confidence = 0.65;
      reasoning = 'MACD positive divergence: momentum increasing';
    }
    // Divergence: MACD below 0 and decreasing
    else if (indicators.histogram < 0 && indicators.histogram < this.previousHistogram) {
      action = 'SELL';
      confidence = 0.65;
      reasoning = 'MACD negative divergence: momentum decreasing';
    }

    this.previousHistogram = indicators.histogram;

    return {
      strategy: 'MACD',
      action,
      confidence,
      indicators: { macd: indicators.macd, signal: indicators.signal, histogram: indicators.histogram },
      reasoning,
    };
  }
}

/**
 * Mean Reversion Strategy
 * Uses Bollinger Bands and RSI for mean reversion signals
 */
export class MeanReversionStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = this.indicators.getCurrentPrice();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Price below lower Bollinger Band + RSI oversold
    if (currentPrice < indicators.bollingerLower && indicators.rsi14 < 30) {
      action = 'BUY';
      confidence = 0.85;
      reasoning = 'Oversold: Price below BB lower band, RSI < 30 (mean reversion buy)';
    }
    // Price above upper Bollinger Band + RSI overbought
    else if (currentPrice > indicators.bollingerUpper && indicators.rsi14 > 70) {
      action = 'SELL';
      confidence = 0.85;
      reasoning = 'Overbought: Price above BB upper band, RSI > 70 (mean reversion sell)';
    }
    // Mild RSI signals
    else if (indicators.rsi14 < 20) {
      action = 'BUY';
      confidence = 0.70;
      reasoning = 'Extreme oversold: RSI < 20';
    } else if (indicators.rsi14 > 80) {
      action = 'SELL';
      confidence = 0.70;
      reasoning = 'Extreme overbought: RSI > 80';
    }

    return {
      strategy: 'Mean Reversion',
      action,
      confidence,
      indicators: {
        rsi14: indicators.rsi14,
        bollingerUpper: indicators.bollingerUpper,
        bollingerLower: indicators.bollingerLower,
        bollingerMiddle: indicators.bollingerMiddle,
      },
      reasoning,
    };
  }
}

/**
 * Grid Trading Strategy
 * Places buy/sell orders at fixed intervals to capture oscillations
 */
export class GridTradingStrategy extends TradingStrategy {
  private grids: GridTrade[] = [];
  private initialPrice = 0;

  initializeGrid(basePrice: number, gridCount: number = 10, gridPercentage: number = 0.02) {
    this.initialPrice = basePrice;
    this.grids = [];

    for (let i = 0; i < gridCount; i++) {
      const offset = (i - gridCount / 2) * gridPercentage;
      const gridPrice = basePrice * (1 + offset);

      this.grids.push({
        gridLevel: i,
        buyPrice: gridPrice * (1 - gridPercentage / 2),
        sellPrice: gridPrice * (1 + gridPercentage / 2),
        quantity: 1 / gridCount,
        filledBuys: false,
        filledSells: false,
      });
    }
  }

  generateSignal(): StrategySignal {
    const currentPrice = this.indicators.getCurrentPrice();
    const indicators = this.getIndicators();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    if (this.grids.length === 0) {
      return {
        strategy: 'Grid Trading',
        action: 'HOLD',
        confidence: 0,
        indicators,
        reasoning: 'Grid not initialized',
      };
    }

    // Check for buy opportunities
    for (const grid of this.grids) {
      if (currentPrice <= grid.buyPrice && !grid.filledBuys) {
        action = 'BUY';
        confidence = 0.80;
        reasoning = `Buy signal at grid level ${grid.gridLevel}: Price ${currentPrice.toFixed(2)} <= ${grid.buyPrice.toFixed(2)}`;
        grid.filledBuys = true;
        break;
      }
    }

    // Check for sell opportunities
    for (const grid of this.grids) {
      if (currentPrice >= grid.sellPrice && !grid.filledSells && grid.filledBuys) {
        action = 'SELL';
        confidence = 0.80;
        reasoning = `Sell signal at grid level ${grid.gridLevel}: Price ${currentPrice.toFixed(2)} >= ${grid.sellPrice.toFixed(2)}`;
        grid.filledSells = true;
        break;
      }
    }

    return {
      strategy: 'Grid Trading',
      action,
      confidence,
      indicators,
      reasoning,
    };
  }

  getGridStatus() {
    return this.grids;
  }

  resetGrid() {
    this.grids = [];
  }
}

/**
 * RSI Divergence Strategy
 * Detects reversals when price makes new high/low but RSI does not
 */
export class RSIDivergenceStrategy extends TradingStrategy {
  private lastHighPrice = 0;
  private lastLowPrice = 0;
  private lastHighRSI = 0;
  private lastLowRSI = 100;

  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = indicators.sma20 ? this.indicators.getCurrentPrice() : 0; // Ensure data exists

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Track local highs/lows (simplified for stream)
    if (this.lastHighPrice === 0) {
      this.lastHighPrice = currentPrice;
      this.lastLowPrice = currentPrice;
      this.lastHighRSI = indicators.rsi14;
      this.lastLowRSI = indicators.rsi14;
    }

    // Bearish Divergence: Price Higher High, RSI Lower High
    if (currentPrice > this.lastHighPrice) {
      if (indicators.rsi14 < this.lastHighRSI && indicators.rsi14 > 70) {
        action = 'SELL';
        confidence = 0.85;
        reasoning = 'Bearish Divergence: Price Higher High, RSI Lower High (Reversal)';
      }
      this.lastHighPrice = currentPrice;
      this.lastHighRSI = indicators.rsi14;
    }

    // Bullish Divergence: Price Lower Low, RSI Higher Low
    if (currentPrice < this.lastLowPrice) {
      if (indicators.rsi14 > this.lastLowRSI && indicators.rsi14 < 30) {
        action = 'BUY';
        confidence = 0.85;
        reasoning = 'Bullish Divergence: Price Lower Low, RSI Higher Low (Reversal)';
      }
      this.lastLowPrice = currentPrice;
      this.lastLowRSI = indicators.rsi14;
    }

    // Regular RSI Overbought/Oversold as backup
    if (action === 'HOLD') {
      if (indicators.rsi14 > 80) {
        action = 'SELL';
        confidence = 0.6;
        reasoning = 'RSI Overbought (>80)';
      } else if (indicators.rsi14 < 20) {
        action = 'BUY';
        confidence = 0.6;
        reasoning = 'RSI Oversold (<20)';
      }
    }

    return {
      strategy: 'RSI Divergence',
      action,
      confidence,
      indicators: { rsi14: indicators.rsi14 },
      reasoning,
    };
  }
}

/**
 * Bollinger Band Breakout Strategy
 * Trends following breakouts of volatility bands
 */
export class BollingerBreakoutStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = this.indicators.getCurrentPrice();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Breakout ABOVE Upper Band
    if (currentPrice > indicators.bollingerUpper) {
      action = 'BUY';
      confidence = 0.8;
      reasoning = 'Price broke ABOVE Upper Bollinger Band (Volatility Breakout)';
    }
    // Breakout BELOW Lower Band
    else if (currentPrice < indicators.bollingerLower) {
      action = 'SELL';
      confidence = 0.8;
      reasoning = 'Price broke BELOW Lower Bollinger Band (Volatility Breakdown)';
    }

    return {
      strategy: 'Bollinger Breakout',
      action,
      confidence,
      indicators: {
        bollingerUpper: indicators.bollingerUpper,
        bollingerLower: indicators.bollingerLower
      },
      reasoning,
    };
  }
}

/**
 * VWAP Trend Strategy
 * Helper for intraday institutional trend following
 */
export class VWAPTrendStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = this.indicators.getCurrentPrice();
    const vwap = indicators.vwap || currentPrice; // Fallback

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Price crossing ABOVE VWAP + Volume Confirmation
    if (currentPrice > vwap * 1.002) { // 0.2% buffer
      action = 'BUY';
      confidence = 0.75;
      reasoning = 'Price trending ABOVE VWAP (Institutional Buy Zone)';
    }
    // Price crossing BELOW VWAP
    else if (currentPrice < vwap * 0.998) {
      action = 'SELL';
      confidence = 0.75;
      reasoning = 'Price trending BELOW VWAP (Institutional Sell Zone)';
    }

    return {
      strategy: 'VWAP Trend',
      action,
      confidence,
      indicators: { vwap },
      reasoning,
    };
  }
}

/**
 * Ichimoku Cloud Strategy
 * Full implementation of Kumo Breakout
 */
export class IchimokuStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = this.indicators.getCurrentPrice();
    const ichi = indicators.ichimoku;

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    if (ichi) {
      const { tenkan, kijun, senkouA, senkouB } = ichi;
      const cloudTop = Math.max(senkouA, senkouB);
      const cloudBottom = Math.min(senkouA, senkouB);

      // Strong Buy: Price > Cloud AND Tenkan (9) > Kijun (26)
      if (currentPrice > cloudTop && tenkan > kijun) {
        action = 'BUY';
        confidence = 0.9;
        reasoning = 'Strong Buy: Price ABOVE Cloud + Tenkan/Kijun Cross';
      }
      // Strong Sell: Price < Cloud AND Tenkan < Kijun
      else if (currentPrice < cloudBottom && tenkan < kijun) {
        action = 'SELL';
        confidence = 0.9;
        reasoning = 'Strong Sell: Price BELOW Cloud + Tenkan/Kijun Cross';
      }
    }

    return {
      strategy: 'Ichimoku Cloud',
      action,
      confidence,
      indicators: { ichimoku: ichi },
      reasoning,
    };
  }
}

/**
 * Pivot Point Reversal Strategy
 * Trades bounces off Support/Resistance levels
 */
export class PivotReversalStrategy extends TradingStrategy {
  generateSignal(): StrategySignal {
    const indicators = this.getIndicators();
    const currentPrice = this.indicators.getCurrentPrice();

    // Calculate Rolling Pivot (Last 50 candles as "Session")
    const high = this.indicators.getHighestPrice(50);
    const low = this.indicators.getLowestPrice(50);
    const close = this.indicators.getCurrentPrice(); // Using current close as proxy for prev close in rolling window

    const pp = (high + low + close) / 3;
    const r1 = 2 * pp - low;
    const s1 = 2 * pp - high;

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let reasoning = '';

    // Reversal off Support (S1)
    if (Math.abs(currentPrice - s1) < s1 * 0.002) { // Within 0.2% of S1
      action = 'BUY';
      confidence = 0.8;
      reasoning = 'Price bouncing off Support S1 (Pivot Reversal)';
    }
    // Reversal off Resistance (R1)
    else if (Math.abs(currentPrice - r1) < r1 * 0.002) {
      action = 'SELL';
      confidence = 0.8;
      reasoning = 'Price rejecting Resistance R1 (Pivot Reversal)';
    }

    return {
      strategy: 'Pivot Points',
      action,
      confidence,
      indicators: {
        // Custom indicator data for UI
        // Using atr14 field as generic placeholder or add custom field in future
        atr14: pp // Hack to show Pivot Level in debug if needed
      },
      reasoning,
    };
  }
}

/**
 * Combined Multi-Signal Strategy
 * Uses voting system from multiple strategies
 */
export class MultiSignalStrategy {
  private maStrategy: MovingAverageCrossoverStrategy;
  private macdStrategy: MACDStrategy;
  private reversionStrategy: MeanReversionStrategy;
  private gridStrategy: GridTradingStrategy;

  constructor(gridBasePrice?: number) {
    this.maStrategy = new MovingAverageCrossoverStrategy();
    this.macdStrategy = new MACDStrategy();
    this.reversionStrategy = new MeanReversionStrategy();
    this.gridStrategy = new GridTradingStrategy();

    if (gridBasePrice) {
      this.gridStrategy.initializeGrid(gridBasePrice);
    }
  }

  addCandle(close: number, volume: number = 0, high: number = close, low: number = close) {
    this.maStrategy.addCandle(close, volume, high, low);
    this.macdStrategy.addCandle(close, volume, high, low);
    this.reversionStrategy.addCandle(close, volume, high, low);
    this.gridStrategy.addCandle(close, volume, high, low);
  }

  generateSignals(): {
    combined: StrategySignal;
    individual: StrategySignal[];
  } {
    const signals = [
      this.maStrategy.generateSignal(),
      this.macdStrategy.generateSignal(),
      this.reversionStrategy.generateSignal(),
      this.gridStrategy.generateSignal(),
    ];

    // Voting system
    const buyVotes = signals.filter((s) => s.action === 'BUY').length;
    const sellVotes = signals.filter((s) => s.action === 'SELL').length;

    let combinedAction: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;

    if (buyVotes >= 2 && buyVotes > sellVotes) {
      combinedAction = 'BUY';
      confidence = Math.min(0.95, 0.5 + buyVotes * 0.2);
    } else if (sellVotes >= 2 && sellVotes > buyVotes) {
      combinedAction = 'SELL';
      confidence = Math.min(0.95, 0.5 + sellVotes * 0.2);
    } else {
      confidence = 0.3;
    }

    const avgConfidence =
      signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length;

    return {
      combined: {
        strategy: 'Multi-Signal Ensemble',
        action: combinedAction,
        confidence: (confidence + avgConfidence) / 2,
        indicators: signals[0].indicators,
        reasoning: `${combinedAction}: ${buyVotes} buy signals, ${sellVotes} sell signals from 4 strategies`,
      },
      individual: signals,
    };
  }

  getIndicators(): IndicatorData {
    // Expose indicators from one of the sub-strategies (they all share the same data feed)
    return this.maStrategy.getIndicators();
  }
}
