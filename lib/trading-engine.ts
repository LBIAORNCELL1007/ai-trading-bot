import { BinanceClient } from './binance-client';
import { MultiSignalStrategy, StrategySignal } from './trading-strategies';
import { RiskManager, Position, PortfolioMetrics } from './risk-management';
import { TechnicalIndicators } from './technical-indicators';
import { MarketAnalyzer } from './market-analyzer';
import { AgenticOrchestrator, StrategicDecision } from './agentic-orchestrator';

export interface TradeExecutionConfig {
  symbol: string;
  strategy: 'MA_CROSSOVER' | 'MACD' | 'MEAN_REVERSION' | 'GRID' | 'MULTI_SIGNAL';
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  riskPerTrade: number; // percentage of equity to risk per trade (e.g. 1 = 1%)
  rewardMultiplier: number;
  maxPositions: number;
  paperTrading: boolean;
  gridCount?: number;
  gridPercentage?: number;
  /** Minimum strategy-confidence to enter a trade (default 0.65). */
  minConfidence?: number;
  /** Use ATR-scaled SL/TP instead of static % (default true). */
  useATRSizing?: boolean;
  /** SL = atr% × atrMultiplier (default 1.5). */
  atrMultiplier?: number;
  /** Hard max-bars-in-trade (default 96). */
  maxHoldBars?: number;
  /** Minimum ADX to trade trend strategies (default 20). */
  minADX?: number;
  /** Use AgenticOrchestrator regime detection to scale risk/TP per bar. */
  useAgenticOrchestrator?: boolean;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  entryTime: number;
  exitTime?: number;
  status: 'OPEN' | 'CLOSED' | 'PENDING';
  pnl?: number;
  pnlPercent?: number;
  strategy: string;
  signal?: StrategySignal;
}

export class TradingEngine {
  private binanceClient: BinanceClient;
  private riskManager: RiskManager;
  private strategy: MultiSignalStrategy;
  private openTrades: Trade[] = [];
  private closedTrades: Trade[] = [];
  private config: TradeExecutionConfig;
  private isRunning: boolean = false;
  private indicators: TechnicalIndicators;
  private priceHistory: number[] = [];
  private currentPrices: Record<string, number> = {};
  private websocketConnections: any[] = [];
  private lastKlineTime: number = 0; // deduplicate kline candle updates
  private barIndex: number = 0; // monotonic bar counter (one per closed candle)
  private lastEvaluatedBar: number = -1; // ensures one strategy eval per closed bar
  private entryBarByTradeId: Map<string, number> = new Map();
  private orchestrator: AgenticOrchestrator;
  private currentDecision: StrategicDecision | null = null;
  private strategyByTradeId: Map<string, string> = new Map();

  constructor(
    binanceClient: BinanceClient,
    accountBalance: number,
    config: TradeExecutionConfig
  ) {
    this.binanceClient = binanceClient;
    this.config = config;
    this.riskManager = new RiskManager(accountBalance);
    this.riskManager.setMaxRiskPerTrade(config.riskPerTrade);
    this.strategy = new MultiSignalStrategy(100); // Default grid base price
    this.indicators = new TechnicalIndicators();
    this.orchestrator = new AgenticOrchestrator();
  }

  /**
   * Start the trading engine
   */
  async start() {
    if (this.isRunning) return;

    this.isRunning = true;
    console.log(`Starting trading engine for ${this.config.symbol} on ${this.config.timeframe}`);

    try {
      // Get initial price data
      const klines = await this.binanceClient.getKlines(
        this.config.symbol,
        this.config.timeframe,
        100
      );

      klines.forEach((k) => {
        this.indicators.addCandle(k.close, k.volume, k.high, k.low);
        this.strategy.addCandle(k.close, k.volume, k.high, k.low);
        this.priceHistory.push(k.close);
        this.barIndex++;
      });

      // Subscribe to real-time data
      this.subscribeToRealtimeData();

      // Start execution loop
      this.startExecutionLoop();
    } catch (error) {
      console.error('Error starting trading engine:', error);
      this.isRunning = false;
    }
  }

  /**
   * Stop the trading engine
   */
  async stop() {
    this.isRunning = false;
    this.websocketConnections.forEach((ws) => ws?.close?.());
    console.log('Trading engine stopped');
  }

  /**
   * Subscribe to real-time price updates
   */
  private subscribeToRealtimeData() {
    const ws = this.binanceClient.subscribeToKline(
      this.config.symbol,
      this.config.timeframe,
      (klineData) => {
        // Deduplicate: only process each candle once (Binance sends multiple
        // updates per candle interval; we only want the first or final close)
        if (klineData.time === this.lastKlineTime) {
          // Same candle — just update the live price for exit checks
          this.currentPrices[this.config.symbol] = klineData.close;
          this.checkPositionExits(klineData.close);
          return;
        }

        this.lastKlineTime = klineData.time;

        this.indicators.addCandle(klineData.close, klineData.volume, klineData.high, klineData.low);
        this.strategy.addCandle(klineData.close, klineData.volume, klineData.high, klineData.low);
        this.priceHistory.push(klineData.close);
        this.currentPrices[this.config.symbol] = klineData.close;
        this.barIndex++;

        // Check for position exits on new bar close
        this.checkPositionExits(klineData.close);

        // Trigger one strategy evaluation per *new closed bar* (not every 5s).
        // This prevents the engine from re-firing identical signals 12× per minute
        // and ensures event-based strategies (MA cross, MACD cross) work correctly.
        this.evaluateNewBar().catch(err => console.error('Bar eval error:', err));
      }
    );

    this.websocketConnections.push(ws);
  }

  /**
   * Periodic safety / housekeeping loop. Strategy evaluation itself is
   * triggered by `evaluateNewBar()` on each closed candle (see
   * subscribeToRealtimeData) — NOT here. This loop only:
   *   • re-checks intrabar exits against the latest tick price
   *   • polls REST as a fallback if websocket has gone silent
   */
  private startExecutionLoop() {
    let lastBarSeenAt = Date.now();
    const interval = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(interval);
        return;
      }

      try {
        // Fallback: if no kline update for >2× the timeframe, poll REST.
        const tfMs = this.timeframeToMs(this.config.timeframe);
        if (Date.now() - lastBarSeenAt > tfMs * 2 && this.lastKlineTime > 0) {
          const last = await this.binanceClient.getKlines(this.config.symbol, this.config.timeframe, 2);
          if (last.length > 0) {
            const k = last[last.length - 1];
            if (k.time !== this.lastKlineTime) {
              this.lastKlineTime = k.time;
              this.indicators.addCandle(k.close, k.volume, k.high, k.low);
              this.strategy.addCandle(k.close, k.volume, k.high, k.low);
              this.priceHistory.push(k.close);
              this.currentPrices[this.config.symbol] = k.close;
              this.barIndex++;
              await this.evaluateNewBar();
            }
          }
        }
        if (this.lastKlineTime > 0) lastBarSeenAt = Math.max(lastBarSeenAt, this.lastKlineTime);
      } catch (error) {
        console.error('Error in execution loop:', error);
      }
    }, 5000);
  }

  private timeframeToMs(tf: string): number {
    const map: Record<string, number> = {
      '1m': 60_000, '5m': 300_000, '15m': 900_000,
      '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
    };
    return map[tf] ?? 60_000;
  }

  /**
   * Evaluate strategy ONCE per newly-closed bar. Idempotent: even if called
   * multiple times for the same bar (e.g. fallback path) it only runs once.
   */
  private async evaluateNewBar() {
    if (this.barIndex === this.lastEvaluatedBar) return;
    this.lastEvaluatedBar = this.barIndex;
    try {
      await this.executeStrategy();
    } catch (e) {
      console.error('executeStrategy error:', e);
    }
  }

  /**
   * Execute trading strategy. Triggered exactly once per closed bar by
   * `evaluateNewBar`. Gates entries on:
   *   • confidence ≥ minConfidence (default 0.65)
   *   • ADX ≥ minADX for trend strategies
   *   • risk-manager safety checks (drawdown / cooldown)
   */
  private async executeStrategy() {
    if (this.openTrades.length >= this.config.maxPositions) {
      return; // Max positions reached
    }

    // Need enough warmup data for indicators (BB squeeze percentile uses 100).
    if (this.barIndex < 50) return;

    const { combined } = this.strategy.generateSignals();

    if (combined.action === 'HOLD') return;

    // --- Confidence gate ---
    const minConf = this.config.minConfidence ?? 0.65;
    if (combined.confidence < minConf) {
      return;
    }

    // --- Trend-strength filter (skip noise environments) ---
    const ind = this.strategy.getIndicators();
    const minADX = this.config.minADX ?? 20;
    const isTrendStrategy =
      this.config.strategy === 'MA_CROSSOVER' ||
      this.config.strategy === 'MACD' ||
      this.config.strategy === 'MULTI_SIGNAL';
    if (isTrendStrategy && ind.adx < minADX) {
      return;
    }

    // --- Agentic regime gating ---
    // Compute the strategic decision once per bar.  The orchestrator's
    // regime detection can:
    //   • veto entries when its confidence is too low
    //   • require the active strategy to match the regime's recommendation
    //   • scale risk and TP via riskMultiplier / tpMultiplier (used below)
    if (this.config.useAgenticOrchestrator !== false) {
      const analysis = MarketAnalyzer.analyze(ind);
      const recentPnl = this.closedTrades.slice(-10).reduce((s, t) => s + (t.pnl ?? 0), 0);
      this.currentDecision = this.orchestrator.decide(
        analysis,
        ind,
        this.config.strategy,
        recentPnl
      );

      // Gate: orchestrator has low conviction in current conditions
      if (this.currentDecision.confidence < 0.5) {
        return;
      }
      // Gate: regime conflicts with the configured strategy AND orchestrator
      // strongly prefers a different one — skip rather than fight the regime.
      if (
        this.currentDecision.shouldSwitch &&
        this.currentDecision.recommendedStrategy !== this.config.strategy &&
        this.currentDecision.confidence >= 0.75
      ) {
        console.log(
          `[Orchestrator] Regime ${this.currentDecision.regime} prefers ` +
          `${this.currentDecision.recommendedStrategy} over ${this.config.strategy} — skipping entry`
        );
        return;
      }
    } else {
      this.currentDecision = null;
    }

    // --- Safety layer ---
    const equity = this.riskManager.calculatePortfolioMetrics(this.currentPrices).equity;
    const allowed = this.riskManager.isTradingAllowed(equity);
    if (!allowed.allowed) {
      console.log(`Trading blocked: ${allowed.reason}`);
      return;
    }

    const currentPrice = this.indicators.getCurrentPrice();

    if (combined.action === 'BUY') {
      await this.executeBuy(combined, currentPrice);
    } else if (combined.action === 'SELL') {
      await this.executeSell(combined, currentPrice);
    }
  }

  /**
   * Execute a buy order with ATR-scaled SL/TP and fixed-fractional sizing.
   */
  private async executeBuy(signal: StrategySignal, currentPrice: number) {
    const ind = this.strategy.getIndicators();
    const useATR = this.config.useATRSizing !== false; // default true
    const atrMult = this.config.atrMultiplier ?? 1.5;
    const baseRR = this.config.rewardMultiplier || 2.0;

    // Apply orchestrator's volatility scaling (if enabled).
    const riskMult = this.currentDecision?.riskMultiplier ?? 1.0;
    const tpMult = this.currentDecision?.tpMultiplier ?? 1.0;
    const rrRatio = baseRR * tpMult;

    // Temporarily scale per-trade risk (riskManager.maxRiskPerTrade) by riskMult.
    // setMaxRiskPerTrade takes a percentage (e.g. 1.0 = 1% of equity).
    const baseRiskPct = this.config.riskPerTrade;
    const scaledRiskPct = Math.max(0.25, Math.min(5.0, baseRiskPct * riskMult));
    this.riskManager.setMaxRiskPerTrade(scaledRiskPct);

    let stopLoss: number;
    let takeProfit: number;
    if (useATR && ind.atrPercent > 0) {
      const r = this.riskManager.setRiskRewardATR(
        currentPrice,
        ind.atrPercent,
        atrMult,
        rrRatio,
        'LONG'
      );
      stopLoss = r.stopLoss;
      takeProfit = r.takeProfit;
    } else {
      // Fallback to static % (legacy). Treat riskPerTrade as SL distance %.
      const slDistFrac = (this.config.riskPerTrade || 1) / 100;
      const r = this.riskManager.setRiskReward(currentPrice, slDistFrac, rrRatio, 'LONG');
      stopLoss = r.stopLoss;
      takeProfit = r.takeProfit;
    }

    const quantity = this.riskManager.calculatePositionSizeFixedRisk(currentPrice, stopLoss);
    if (quantity <= 0) return;

    if (!this.riskManager.canOpenPosition(currentPrice * quantity)) {
      console.log('Cannot open position: insufficient balance');
      return;
    }

    const tradeId = `${this.config.symbol}-${Date.now()}`;
    const trade: Trade = {
      id: tradeId,
      symbol: this.config.symbol,
      side: 'BUY',
      entryPrice: currentPrice,
      quantity,
      entryTime: Date.now(),
      status: 'PENDING',
      strategy: signal.strategy,
      signal,
    };

    if (!this.config.paperTrading) {
      try {
        const order = await this.binanceClient.buyLimit(
          this.config.symbol,
          quantity,
          currentPrice
        );
        console.log('Buy order placed:', order);
      } catch (error) {
        console.error('Error placing buy order:', error);
        return;
      }
    }

    const position: Position = {
      symbol: this.config.symbol,
      side: 'LONG',
      entryPrice: currentPrice,
      quantity,
      entryTime: Date.now(),
      takeProfitPrice: takeProfit,
      stopLossPrice: stopLoss,
    };

    this.riskManager.addPosition(position);
    this.openTrades.push({ ...trade, status: 'OPEN' });
    this.entryBarByTradeId.set(tradeId, this.barIndex);
    this.strategyByTradeId.set(tradeId, signal.strategy);

    // Restore base risk for the next iteration (orchestrator scaling is per-bar).
    this.riskManager.setMaxRiskPerTrade(this.config.riskPerTrade);

    console.log(
      `BUY ${this.config.symbol} @ ${currentPrice.toFixed(4)} ` +
      `qty=${quantity.toFixed(6)} SL=${stopLoss.toFixed(4)} TP=${takeProfit.toFixed(4)} ` +
      `conf=${signal.confidence.toFixed(2)} ADX=${ind.adx.toFixed(1)}`
    );
  }

  /**
   * Execute a sell order
   */
  private async executeSell(signal: StrategySignal, currentPrice: number) {
    const openBuyTrades = this.openTrades.filter(
      (t) => t.side === 'BUY' && t.status === 'OPEN' && t.symbol === this.config.symbol
    );

    if (openBuyTrades.length === 0) {
      console.log('No open buy positions to sell');
      return;
    }

    const trade = openBuyTrades[0];
    const quantity = trade.quantity;

    if (!this.config.paperTrading) {
      try {
        const order = await this.binanceClient.sellLimit(
          this.config.symbol,
          quantity,
          currentPrice
        );
        console.log('Sell order placed:', order);
      } catch (error) {
        console.error('Error placing sell order:', error);
        return;
      }
    }

    // Record the trade
    const pnl = (currentPrice - trade.entryPrice) * quantity;
    const pnlPercent = ((currentPrice - trade.entryPrice) / trade.entryPrice) * 100;

    this.riskManager.recordTrade(
      this.config.symbol,
      'SELL',
      trade.entryPrice,
      currentPrice,
      quantity
    );

    // Update trade status
    trade.exitPrice = currentPrice;
    trade.exitTime = Date.now();
    trade.status = 'CLOSED';
    trade.pnl = pnl;
    trade.pnlPercent = pnlPercent;

    this.closedTrades.push(trade);
    this.openTrades = this.openTrades.filter((t) => t.id !== trade.id);
    this.entryBarByTradeId.delete(trade.id);
    const stratExit = this.strategyByTradeId.get(trade.id) ?? trade.strategy;
    this.strategyByTradeId.delete(trade.id);
    this.orchestrator.recordTradeResult(stratExit, pnl);
    this.riskManager.removePosition(this.config.symbol);

    console.log(`Sell signal: ${this.config.symbol} @ ${currentPrice} (PnL: $${pnl.toFixed(2)})`);
  }

  /**
   * Check for position exits (stop loss, take profit, max-hold-bars).
   * Called on every kline update so it sees intrabar prices.
   */
  private async checkPositionExits(currentPrice: number) {
    const positions = this.riskManager.getPositions();
    const maxHold = this.config.maxHoldBars ?? 96;

    for (const position of positions) {
      const exitCheck = this.riskManager.shouldClosePosition(position, currentPrice);

      const openTrade = this.openTrades.find(
        (t) => t.symbol === position.symbol && t.status === 'OPEN'
      );

      let shouldClose = exitCheck.shouldClose;
      let reason = exitCheck.reason;

      // Time-based exit (bar-based, not ms-based)
      if (!shouldClose && openTrade) {
        const entryBar = this.entryBarByTradeId.get(openTrade.id);
        if (entryBar !== undefined && this.barIndex - entryBar >= maxHold) {
          shouldClose = true;
          reason = `Max hold bars reached (${maxHold})`;
        }
      }

      if (shouldClose && openTrade) {
        const pnl = (currentPrice - openTrade.entryPrice) * openTrade.quantity;
        const pnlPercent = ((currentPrice - openTrade.entryPrice) / openTrade.entryPrice) * 100;

        if (!this.config.paperTrading) {
          try {
            await this.binanceClient.sellMarket(this.config.symbol, openTrade.quantity);
          } catch (error) {
            console.error('Error closing position:', error);
            continue;
          }
        }

        this.riskManager.recordTrade(
          openTrade.symbol,
          'SELL',
          openTrade.entryPrice,
          currentPrice,
          openTrade.quantity
        );

        openTrade.exitPrice = currentPrice;
        openTrade.exitTime = Date.now();
        openTrade.status = 'CLOSED';
        openTrade.pnl = pnl;
        openTrade.pnlPercent = pnlPercent;

        this.closedTrades.push(openTrade);
        this.openTrades = this.openTrades.filter((t) => t.id !== openTrade.id);
        this.entryBarByTradeId.delete(openTrade.id);
        const strat = this.strategyByTradeId.get(openTrade.id) ?? openTrade.strategy;
        this.strategyByTradeId.delete(openTrade.id);
        this.orchestrator.recordTradeResult(strat, pnl);
        this.riskManager.removePosition(position.symbol);

        console.log(`Position closed (${reason}): PnL: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`);
      }
    }
  }

  /**
   * Get all open trades
   */
  getOpenTrades(): Trade[] {
    return this.openTrades;
  }

  /**
   * Get closed trades
   */
  getClosedTrades(): Trade[] {
    return this.closedTrades;
  }

  /**
   * Get portfolio metrics
   */
  getPortfolioMetrics(): PortfolioMetrics {
    return this.riskManager.calculatePortfolioMetrics(this.currentPrices);
  }

  /**
   * Get current price
   */
  getCurrentPrice(): number {
    return this.indicators.getCurrentPrice();
  }

  /**
   * Get strategy signals
   */
  getLatestSignals(): {
    combined: StrategySignal;
    individual: StrategySignal[];
  } {
    return this.strategy.generateSignals();
  }

  /**
   * Check if engine is running
   */
  isEngineRunning(): boolean {
    return this.isRunning;
  }
}
