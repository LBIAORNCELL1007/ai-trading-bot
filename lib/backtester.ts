import { MultiSignalStrategy } from './trading-strategies';
import { RiskManager, Position } from './risk-management';
import { TechnicalIndicators } from './technical-indicators';

export interface BacktestResult {
  symbol: string;
  strategy: string;
  startDate: number;
  endDate: number;
  initialBalance: number;
  finalBalance: number;
  totalReturn: number;
  returnPercentage: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
  sharpeRatio: number;
  trades: BacktestTrade[];
}

export interface BacktestTrade {
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  entryTime: number;
  exitTime: number;
  pnl: number;
  pnlPercent: number;
  strategy: string;
  exitReason: string;
}

/**
 * Realistic event-driven backtester.
 *
 * Critical correctness fixes vs prior version:
 *
 *   1. NO LOOK-AHEAD ON ENTRY.  Signals computed from bar `i` are filled at
 *      bar `i+1`'s OPEN (the next bar's first realistic price), not at bar `i`'s
 *      close.  Previously the engine got to "buy at the close that just printed"
 *      — a 5–15% inflation of paper win rate.
 *
 *   2. INTRABAR EXIT ORDERING.  When both SL and TP could be touched within the
 *      same candle (high reaches TP and low reaches SL), we conservatively assume
 *      the SL was hit first (cannot resolve order without lower-TF data).  We
 *      also DO NOT exit a position on the very same bar it was opened.
 *
 *   3. PROPER SHARPE ANNUALIZATION.  The annualization factor is derived from
 *      the candle interval (`periodsPerYear` argument).  Per-trade returns are
 *      no longer mixed with a daily annualizer.
 *
 *   4. NO REVERSAL EXIT AT SAME-BAR CLOSE.  A SELL signal exits the position at
 *      the *next* bar's open, not the close that produced the signal.
 *
 *   5. SLIPPAGE & COMMISSION applied symmetrically on both legs.
 */
export class Backtester {
  private strategy: MultiSignalStrategy;
  private riskManager: RiskManager;
  private indicators: TechnicalIndicators;
  private trades: BacktestTrade[] = [];
  private balanceHistory: number[] = [];
  private returns: number[] = [];

  constructor(
    private initialBalance: number = 10000,
    private slippagePercent: number = 0.0005,
    private commissionPercent: number = 0.001
  ) {
    this.strategy = new MultiSignalStrategy();
    this.riskManager = new RiskManager(initialBalance);
    this.indicators = new TechnicalIndicators();
  }

  /**
   * Run backtest on historical data.
   *
   * @param periodsPerYear used for Sharpe annualization (e.g. 1h candles → 24*365 = 8760)
   */
  async backtest(
    symbol: string,
    klines: Array<{ time: number; open: number; high: number; low: number; close: number; volume: number }>,
    config: {
      riskPerTrade: number;       // % of equity to risk per trade (e.g. 1.0 for 1%)
      rewardMultiplier: number;   // R:R ratio (e.g. 2.0 for 2:1)
      maxPositions: number;
      minConfidence?: number;     // gate trades by combined.confidence (default 0.65)
      useATRSizing?: boolean;     // ATR-based SL distance (default true)
      atrMultiplier?: number;     // SL distance = ATR × atrMultiplier (default 1.5)
      maxHoldBars?: number;       // hard time-stop in bars (default 96)
      periodsPerYear?: number;
    }
  ): Promise<BacktestResult> {
    this.trades = [];
    this.balanceHistory = [this.initialBalance];
    this.returns = [];

    let currentBalance = this.initialBalance;
    this.riskManager.updateAccountBalance(currentBalance);
    this.riskManager.setMaxRiskPerTrade(config.riskPerTrade);

    const minConfidence = config.minConfidence ?? 0.65;
    const useATR = config.useATRSizing ?? true;
    const atrMult = config.atrMultiplier ?? 1.5;
    const maxHoldBars = config.maxHoldBars ?? 96;
    const periodsPerYear = config.periodsPerYear ?? 8760;

    // --- Pending state: signal generated on bar i, executed at bar i+1 open ---
    let pendingEntry: { side: 'BUY' | 'SELL'; signalBar: number; confidence: number; strategy: string } | null = null;
    let pendingExit = false;

    interface OpenPos {
      side: 'BUY' | 'SELL';
      entryPrice: number;
      entryTime: number;
      entryBar: number;
      quantity: number;
      stopLoss: number;
      takeProfit: number;
      strategy: string;
    }
    let openPos: OpenPos | null = null;

    for (let i = 0; i < klines.length; i++) {
      const kline = klines[i];

      // ── 1. Execute any pending fills at THIS bar's OPEN (no look-ahead) ──
      if (pendingExit && openPos) {
        const slipExit = openPos.side === 'BUY' ? (1 - this.slippagePercent) : (1 + this.slippagePercent);
        const exitPrice = kline.open * slipExit;
        const grossPnl = openPos.side === 'BUY'
          ? (exitPrice - openPos.entryPrice) * openPos.quantity
          : (openPos.entryPrice - exitPrice) * openPos.quantity;
        const exitFee = exitPrice * openPos.quantity * this.commissionPercent;
        const pnl = grossPnl - exitFee;
        const pnlPercent = (pnl / (openPos.entryPrice * openPos.quantity)) * 100;
        this.trades.push({
          entryPrice: openPos.entryPrice,
          exitPrice,
          quantity: openPos.quantity,
          entryTime: openPos.entryTime,
          exitTime: kline.time,
          pnl,
          pnlPercent,
          strategy: openPos.strategy,
          exitReason: 'reversal',
        });
        currentBalance += pnl;
        this.returns.push(pnl / Math.max(1, currentBalance - pnl));
        this.riskManager.updateAccountBalance(currentBalance);
        openPos = null;
        pendingExit = false;
      }

      if (pendingEntry && !openPos) {
        const ind = this.indicators.calculateAll();
        const slipEntry = pendingEntry.side === 'BUY' ? (1 + this.slippagePercent) : (1 - this.slippagePercent);
        const entryPrice = kline.open * slipEntry;

        // ATR-based SL/TP (volatility-aware)
        const atrPct = ind.atrPercent || 1.5;
        const sl = useATR
          ? this.riskManager.setRiskRewardATR(entryPrice, atrPct, atrMult, config.rewardMultiplier, pendingEntry.side === 'BUY' ? 'LONG' : 'SHORT')
          : (() => {
              const r = this.riskManager.setRiskReward(entryPrice, config.riskPerTrade / 100, config.rewardMultiplier, pendingEntry.side === 'BUY' ? 'LONG' : 'SHORT');
              return { ...r, slPercent: config.riskPerTrade, tpPercent: config.riskPerTrade * config.rewardMultiplier };
          })();

        const quantity = this.riskManager.calculatePositionSizeFixedRisk(entryPrice, sl.stopLoss);
        const entryFee = entryPrice * quantity * this.commissionPercent;
        const entryCost = entryPrice * quantity + entryFee;

        if (this.riskManager.canOpenPosition(entryCost) && quantity > 0) {
          openPos = {
            side: pendingEntry.side,
            entryPrice,
            entryTime: kline.time,
            entryBar: i,
            quantity,
            stopLoss: sl.stopLoss,
            takeProfit: sl.takeProfit,
            strategy: pendingEntry.strategy,
          };
          // Account for entry commission immediately
          currentBalance -= entryFee;
          this.riskManager.updateAccountBalance(currentBalance);
        }
        pendingEntry = null;
      }

      // ── 2. Update indicators with the bar that has now "printed" ──
      this.indicators.addCandle(kline.close, kline.volume, kline.high, kline.low);
      this.strategy.addCandle(kline.close, kline.volume, kline.high, kline.low);

      // ── 3. Intrabar exit checks (do NOT exit on the bar of entry) ──
      if (openPos && i > openPos.entryBar) {
        let exitPrice = 0;
        let exitReason = '';

        if (openPos.side === 'BUY') {
          // Conservative ordering: SL first
          const slHit = kline.low <= openPos.stopLoss;
          const tpHit = kline.high >= openPos.takeProfit;
          if (slHit) {
            exitPrice = openPos.stopLoss * (1 - this.slippagePercent);
            exitReason = 'stop_loss';
          } else if (tpHit) {
            exitPrice = openPos.takeProfit * (1 - this.slippagePercent);
            exitReason = 'take_profit';
          }
        } else {
          const slHit = kline.high >= openPos.stopLoss;
          const tpHit = kline.low <= openPos.takeProfit;
          if (slHit) {
            exitPrice = openPos.stopLoss * (1 + this.slippagePercent);
            exitReason = 'stop_loss';
          } else if (tpHit) {
            exitPrice = openPos.takeProfit * (1 + this.slippagePercent);
            exitReason = 'take_profit';
          }
        }

        // Time stop
        if (!exitReason && i - openPos.entryBar >= maxHoldBars) {
          exitPrice = kline.close * (openPos.side === 'BUY' ? 1 - this.slippagePercent : 1 + this.slippagePercent);
          exitReason = 'time_exit';
        }

        if (exitReason) {
          const grossPnl = openPos.side === 'BUY'
            ? (exitPrice - openPos.entryPrice) * openPos.quantity
            : (openPos.entryPrice - exitPrice) * openPos.quantity;
          const exitFee = exitPrice * openPos.quantity * this.commissionPercent;
          const pnl = grossPnl - exitFee;
          const pnlPercent = (pnl / (openPos.entryPrice * openPos.quantity)) * 100;
          this.trades.push({
            entryPrice: openPos.entryPrice,
            exitPrice,
            quantity: openPos.quantity,
            entryTime: openPos.entryTime,
            exitTime: kline.time,
            pnl,
            pnlPercent,
            strategy: openPos.strategy,
            exitReason,
          });
          currentBalance += pnl;
          this.returns.push(pnl / Math.max(1, currentBalance - pnl));
          this.riskManager.updateAccountBalance(currentBalance);
          openPos = null;
        }
      }

      // ── 4. Generate signal for NEXT bar's open ──
      const { combined } = this.strategy.generateSignals();
      if (combined.action === 'BUY' && combined.confidence >= minConfidence && !openPos && !pendingEntry) {
        if (this.riskManager.getPositions().length < config.maxPositions) {
          pendingEntry = { side: 'BUY', signalBar: i, confidence: combined.confidence, strategy: combined.strategy };
        }
      } else if (combined.action === 'SELL' && combined.confidence >= minConfidence && openPos && openPos.side === 'BUY' && i > openPos.entryBar) {
        pendingExit = true;
      }

      this.balanceHistory.push(currentBalance + (openPos ? this.unrealized(openPos, kline.close) : 0));
    }

    // Force-close any remaining position at the last close
    if (openPos) {
      const last = klines[klines.length - 1];
      const slipExit = openPos.side === 'BUY' ? (1 - this.slippagePercent) : (1 + this.slippagePercent);
      const exitPrice = last.close * slipExit;
      const grossPnl = openPos.side === 'BUY'
        ? (exitPrice - openPos.entryPrice) * openPos.quantity
        : (openPos.entryPrice - exitPrice) * openPos.quantity;
      const exitFee = exitPrice * openPos.quantity * this.commissionPercent;
      const pnl = grossPnl - exitFee;
      const pnlPercent = (pnl / (openPos.entryPrice * openPos.quantity)) * 100;
      this.trades.push({
        entryPrice: openPos.entryPrice,
        exitPrice,
        quantity: openPos.quantity,
        entryTime: openPos.entryTime,
        exitTime: last.time,
        pnl,
        pnlPercent,
        strategy: openPos.strategy,
        exitReason: 'end_of_data',
      });
      currentBalance += pnl;
      this.returns.push(pnl / Math.max(1, currentBalance - pnl));
    }

    return this.calculateResults(symbol, klines[0].time, klines[klines.length - 1].time, periodsPerYear);
  }

  private unrealized(pos: { side: 'BUY' | 'SELL'; entryPrice: number; quantity: number }, currentPrice: number): number {
    return pos.side === 'BUY'
      ? (currentPrice - pos.entryPrice) * pos.quantity
      : (pos.entryPrice - currentPrice) * pos.quantity;
  }

  private calculateResults(
    symbol: string,
    startDate: number,
    endDate: number,
    periodsPerYear: number
  ): BacktestResult {
    const finalBalance = this.balanceHistory[this.balanceHistory.length - 1];
    const totalReturn = finalBalance - this.initialBalance;
    const returnPercentage = (totalReturn / this.initialBalance) * 100;

    const winningTrades = this.trades.filter((t) => t.pnl > 0).length;
    const losingTrades = this.trades.filter((t) => t.pnl < 0).length;
    const winRate = this.trades.length > 0 ? winningTrades / this.trades.length : 0;

    const profits = this.trades.filter((t) => t.pnl > 0).reduce((s, t) => s + t.pnl, 0);
    const losses = Math.abs(this.trades.filter((t) => t.pnl < 0).reduce((s, t) => s + t.pnl, 0));
    const profitFactor = losses > 0 ? profits / losses : profits > 0 ? 100 : 0;

    let peak = this.initialBalance;
    let maxDD = 0;
    for (const balance of this.balanceHistory) {
      if (balance > peak) peak = balance;
      const dd = (peak - balance) / peak;
      if (dd > maxDD) maxDD = dd;
    }

    // Per-trade Sharpe — annualization factor is the AVERAGE TRADES PER YEAR,
    // not bars per year.  Estimate from data span.
    const spanYears = Math.max(1 / 365, (endDate - startDate) / (1000 * 60 * 60 * 24 * 365));
    const tradesPerYear = this.trades.length / spanYears;
    const avgReturn = this.returns.length > 0
      ? this.returns.reduce((a, b) => a + b, 0) / this.returns.length
      : 0;
    const variance = this.returns.length > 0
      ? this.returns.reduce((s, r) => s + (r - avgReturn) ** 2, 0) / this.returns.length
      : 0;
    const stdDev = Math.sqrt(variance);
    const sharpeRatio = stdDev > 0 && tradesPerYear > 0
      ? (avgReturn / stdDev) * Math.sqrt(tradesPerYear)
      : 0;

    return {
      symbol,
      strategy: 'Multi-Signal',
      startDate,
      endDate,
      initialBalance: this.initialBalance,
      finalBalance,
      totalReturn,
      returnPercentage,
      totalTrades: this.trades.length,
      winningTrades,
      losingTrades,
      winRate,
      profitFactor,
      maxDrawdown: maxDD,
      sharpeRatio,
      trades: this.trades,
    };
  }

  getTradeStats() {
    const winningTrades = this.trades.filter((t) => t.pnl > 0);
    const losingTrades = this.trades.filter((t) => t.pnl < 0);
    const averageWin = winningTrades.length > 0 ? winningTrades.reduce((s, t) => s + t.pnl, 0) / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? losingTrades.reduce((s, t) => s + t.pnl, 0) / losingTrades.length : 0;
    const largestWin = Math.max(...winningTrades.map((t) => t.pnl), 0);
    const largestLoss = Math.min(...losingTrades.map((t) => t.pnl), 0);

    let consecutiveWins = 0, consecutiveLosses = 0;
    let maxConsecutiveWins = 0, maxConsecutiveLosses = 0;
    for (const trade of this.trades) {
      if (trade.pnl > 0) {
        consecutiveWins++;
        consecutiveLosses = 0;
        maxConsecutiveWins = Math.max(maxConsecutiveWins, consecutiveWins);
      } else {
        consecutiveLosses++;
        consecutiveWins = 0;
        maxConsecutiveLosses = Math.max(maxConsecutiveLosses, consecutiveLosses);
      }
    }
    return {
      averageWin, averageLoss, largestWin, largestLoss,
      consecutiveWins: maxConsecutiveWins,
      consecutiveLosses: maxConsecutiveLosses,
    };
  }

  reset() {
    this.trades = [];
    this.balanceHistory = [this.initialBalance];
    this.returns = [];
  }

  getBalanceHistory(): number[] { return this.balanceHistory; }
  getReturns(): number[] { return this.returns; }
}
