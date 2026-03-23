import { BinanceClient } from './binance-client';
import { MultiSignalStrategy, StrategySignal } from './trading-strategies';
import { RiskManager, Position, PortfolioMetrics } from './risk-management';
import { TechnicalIndicators } from './technical-indicators';

export interface TradeExecutionConfig {
  symbol: string;
  strategy: 'MA_CROSSOVER' | 'MACD' | 'MEAN_REVERSION' | 'GRID' | 'MULTI_SIGNAL';
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  riskPerTrade: number; // percentage
  rewardMultiplier: number;
  maxPositions: number;
  paperTrading: boolean;
  gridCount?: number;
  gridPercentage?: number;
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

        // Check for position exits
        this.checkPositionExits(klineData.close);
      }
    );

    this.websocketConnections.push(ws);
  }

  /**
   * Start the main execution loop
   */
  private startExecutionLoop() {
    const interval = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(interval);
        return;
      }

      try {
        await this.executeStrategy();
      } catch (error) {
        console.error('Error in execution loop:', error);
      }
    }, 5000); // Check every 5 seconds
  }

  /**
   * Execute trading strategy
   */
  private async executeStrategy() {
    if (this.openTrades.length >= this.config.maxPositions) {
      return; // Max positions reached
    }

    const { combined, individual } = this.strategy.generateSignals();

    if (combined.action === 'HOLD') {
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
   * Execute a buy order
   */
  private async executeBuy(signal: StrategySignal, currentPrice: number) {
    const { stopLoss, takeProfit } = this.riskManager.setRiskReward(
      currentPrice,
      this.config.riskPerTrade / 100,
      this.config.rewardMultiplier
    );

    const quantity = this.riskManager.calculatePositionSizeFixedRisk(
      currentPrice,
      stopLoss
    );

    if (!this.riskManager.canOpenPosition(currentPrice * quantity)) {
      console.log('Cannot open position: insufficient balance');
      return;
    }

    const trade: Trade = {
      id: `${this.config.symbol}-${Date.now()}`,
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

    console.log(`Buy signal: ${this.config.symbol} @ ${currentPrice} (Qty: ${quantity})`);
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
    this.riskManager.removePosition(this.config.symbol);

    console.log(`Sell signal: ${this.config.symbol} @ ${currentPrice} (PnL: $${pnl.toFixed(2)})`);
  }

  /**
   * Check for position exits (stop loss, take profit)
   */
  private async checkPositionExits(currentPrice: number) {
    const positions = this.riskManager.getPositions();

    for (const position of positions) {
      const exitCheck = this.riskManager.shouldClosePosition(position, currentPrice);

      if (exitCheck.shouldClose) {
        const openTrade = this.openTrades.find((t) => t.symbol === position.symbol && t.status === 'OPEN');

        if (openTrade) {
          const pnl = (currentPrice - openTrade.entryPrice) * openTrade.quantity;
          const pnlPercent = ((currentPrice - openTrade.entryPrice) / openTrade.entryPrice) * 100;

          if (!this.config.paperTrading) {
            try {
              await this.binanceClient.sellMarket(
                this.config.symbol,
                openTrade.quantity
              );
            } catch (error) {
              console.error('Error closing position:', error);
              continue;
            }
          }

          // Record the trade
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
          this.riskManager.removePosition(position.symbol);

          console.log(`Position closed (${exitCheck.reason}): PnL: $${pnl.toFixed(2)}`);
        }
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
