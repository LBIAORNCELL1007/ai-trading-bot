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
}

export class Backtester {
  private strategy: MultiSignalStrategy;
  private riskManager: RiskManager;
  private indicators: TechnicalIndicators;
  private trades: BacktestTrade[] = [];
  private balanceHistory: number[] = [];
  private returns: number[] = [];

  constructor(private initialBalance: number = 10000) {
    this.strategy = new MultiSignalStrategy();
    this.riskManager = new RiskManager(initialBalance);
    this.indicators = new TechnicalIndicators();
  }

  /**
   * Run backtest on historical data
   */
  async backtest(
    symbol: string,
    klines: Array<{ time: number; open: number; high: number; low: number; close: number; volume: number }>,
    config: {
      riskPerTrade: number;
      rewardMultiplier: number;
      maxPositions: number;
    }
  ): Promise<BacktestResult> {
    this.trades = [];
    this.balanceHistory = [this.initialBalance];
    this.returns = [];

    let currentBalance = this.initialBalance;
    this.riskManager.updateAccountBalance(currentBalance);
    this.riskManager.setMaxRiskPerTrade(config.riskPerTrade);

    for (const kline of klines) {
      this.indicators.addCandle(kline.close, kline.volume, kline.high, kline.low);
      this.strategy.addCandle(kline.close, kline.volume, kline.high, kline.low);

      const { combined } = this.strategy.generateSignals();

      // Process buy signals
      if (combined.action === 'BUY' && this.riskManager.getPositions().length < config.maxPositions) {
        const { stopLoss, takeProfit } = this.riskManager.setRiskReward(
          kline.close,
          config.riskPerTrade / 100,
          config.rewardMultiplier
        );

        const quantity = this.riskManager.calculatePositionSizeFixedRisk(
          kline.close,
          stopLoss
        );

          const slippageFactor = 1.0005; // 0.05% slippage on entry
          const commissionRate = 0.001; // 0.1% commission
          const adjustedEntryPrice = kline.close * slippageFactor;
          const entryCost = adjustedEntryPrice * quantity * (1 + commissionRate);

        if (this.riskManager.canOpenPosition(entryCost)) {
          const position: Position = {
            symbol,
            side: 'LONG',
            entryPrice: adjustedEntryPrice,
            quantity,
            entryTime: kline.time,
            takeProfitPrice: takeProfit,
            stopLossPrice: stopLoss,
          };
          this.riskManager.addPosition(position);
        }
      }

      // Check for exits
      const positions = this.riskManager.getPositions();
      for (const position of positions) {
        if (position.symbol !== symbol) continue;

        let exitPrice = kline.close;
        let shouldExit = false;

        // Check stop loss
        if (position.stopLossPrice && kline.low <= position.stopLossPrice) {
          exitPrice = position.stopLossPrice;
          shouldExit = true;
        }
        // Check take profit
        else if (position.takeProfitPrice && kline.high >= position.takeProfitPrice) {
          exitPrice = position.takeProfitPrice;
          shouldExit = true;
        }

        // Sell on short term if conditions reverse
        if (!shouldExit && combined.action === 'SELL') {
          exitPrice = kline.close;
          shouldExit = true;
        }

        if (shouldExit) {
          const slippageFactor = 0.9995; // 0.05% slippage on exit (against you)
          const commissionRate = 0.001; // 0.1% commission
          const adjustedExitPrice = exitPrice * slippageFactor;
          const exitCommission = adjustedExitPrice * position.quantity * commissionRate;
          const pnl = (adjustedExitPrice - position.entryPrice) * position.quantity - exitCommission;
          const pnlPercent = ((adjustedExitPrice - position.entryPrice) / position.entryPrice) * 100;

          this.trades.push({
            entryPrice: position.entryPrice,
            exitPrice: adjustedExitPrice,
            quantity: position.quantity,
            entryTime: position.entryTime,
            exitTime: kline.time,
            pnl,
            pnlPercent,
            strategy: combined.strategy,
          });

          currentBalance += pnl;
          this.returns.push(pnl / (currentBalance - pnl));
          this.riskManager.updateAccountBalance(currentBalance);
          this.riskManager.removePosition(symbol);
          this.riskManager.recordTrade(symbol, 'SELL', position.entryPrice, adjustedExitPrice, position.quantity);
        }
      }

      this.balanceHistory.push(currentBalance);
    }

    return this.calculateResults(symbol, klines[0].time, klines[klines.length - 1].time);
  }

  /**
   * Calculate backtest statistics
   */
  private calculateResults(
    symbol: string,
    startDate: number,
    endDate: number
  ): BacktestResult {
    const totalReturn = this.balanceHistory[this.balanceHistory.length - 1] - this.initialBalance;
    const returnPercentage = (totalReturn / this.initialBalance) * 100;

    const winningTrades = this.trades.filter((t) => t.pnl > 0).length;
    const losingTrades = this.trades.filter((t) => t.pnl < 0).length;
    const winRate = this.trades.length > 0 ? winningTrades / this.trades.length : 0;

    const profits = this.trades
      .filter((t) => t.pnl > 0)
      .reduce((sum, t) => sum + t.pnl, 0);
    const losses = Math.abs(
      this.trades
        .filter((t) => t.pnl < 0)
        .reduce((sum, t) => sum + t.pnl, 0)
    );
    const profitFactor = losses > 0 ? profits / losses : profits > 0 ? 100 : 0;

    // Max drawdown
    let peak = this.initialBalance;
    let maxDD = 0;
    for (const balance of this.balanceHistory) {
      if (balance > peak) {
        peak = balance;
      }
      const drawdown = (peak - balance) / peak;
      if (drawdown > maxDD) {
        maxDD = drawdown;
      }
    }

    // Sharpe ratio
    const avgReturn = this.returns.length > 0 ? this.returns.reduce((a, b) => a + b, 0) / this.returns.length : 0;
    const variance = this.returns.length > 0
      ? this.returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / this.returns.length
      : 0;
    const stdDev = Math.sqrt(variance);
    const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;

    return {
      symbol,
      strategy: 'Multi-Signal',
      startDate,
      endDate,
      initialBalance: this.initialBalance,
      finalBalance: this.balanceHistory[this.balanceHistory.length - 1],
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

  /**
   * Get detailed trade by trade statistics
   */
  getTradeStats(): {
    averageWin: number;
    averageLoss: number;
    largestWin: number;
    largestLoss: number;
    consecutiveWins: number;
    consecutiveLosses: number;
  } {
    const winningTrades = this.trades.filter((t) => t.pnl > 0);
    const losingTrades = this.trades.filter((t) => t.pnl < 0);

    const averageWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length : 0;

    const largestWin = Math.max(...winningTrades.map((t) => t.pnl), 0);
    const largestLoss = Math.min(...losingTrades.map((t) => t.pnl), 0);

    let consecutiveWins = 0;
    let consecutiveLosses = 0;
    let maxConsecutiveWins = 0;
    let maxConsecutiveLosses = 0;

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
      averageWin,
      averageLoss,
      largestWin,
      largestLoss,
      consecutiveWins: maxConsecutiveWins,
      consecutiveLosses: maxConsecutiveLosses,
    };
  }

  /**
   * Reset backtester
   */
  reset() {
    this.trades = [];
    this.balanceHistory = [this.initialBalance];
    this.returns = [];
  }

  /**
   * Get balance history
   */
  getBalanceHistory(): number[] {
    return this.balanceHistory;
  }

  /**
   * Get returns
   */
  getReturns(): number[] {
    return this.returns;
  }
}
