export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  quantity: number;
  entryTime: number;
  takeProfitPrice?: number;
  stopLossPrice?: number;
  trailingStop?: number;
}

export interface PortfolioMetrics {
  totalBalance: number;
  usedBalance: number;
  availableBalance: number;
  equity: number;
  margin: number;
  marginLevel: number;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPnL: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
}

export class RiskManager {
  private positions: Position[] = [];
  private tradeHistory: any[] = [];
  private maxRiskPerTrade: number = 0.02; // 2% risk per trade
  private maxDrawdown: number = 0.1; // 10% max drawdown
  private currentDrawdown: number = 0;

  // --- Safety Layer State ---
  private initialEquity: number = 0;
  private isCircuitBreakerTripped: boolean = false;
  private consecutiveLosses: number = 0;
  private cooldownEndTime: number = 0;
  private maxConsecutiveLosses: number = 3;
  private cooldownDuration: number = 3600000; // 1 hour in ms

  constructor(private accountBalance: number) {
    this.initialEquity = accountBalance;
  }

  // --- Safety Checks ---

  /**
   * Check if trading is allowed based on Safety Rules
   */
  isTradingAllowed(currentEquity: number): { allowed: boolean; reason: string } {
    // 1. Check Circuit Breaker (Max Drawdown)
    const drawdown = (this.initialEquity - currentEquity) / this.initialEquity;
    if (drawdown >= this.maxDrawdown) {
      this.isCircuitBreakerTripped = true;
      return { allowed: false, reason: `Circuit Breaker Tripped! Drawdown ${(drawdown * 100).toFixed(2)}% > ${(this.maxDrawdown * 100).toFixed(2)}%` };
    }

    // 2. Check Cooldown
    if (Date.now() < this.cooldownEndTime) {
      const remainingMinutes = Math.ceil((this.cooldownEndTime - Date.now()) / 60000);
      return { allowed: false, reason: `Cooldown Active. ${remainingMinutes}m remaining (3 Consecutive Losses).` };
    }

    // 3. Check Manual Kill Switch
    if (this.isCircuitBreakerTripped) {
      return { allowed: false, reason: 'System locked via Kill Switch or Circuit Breaker.' };
    }

    return { allowed: true, reason: '' };
  }

  /**
   * Record a trade result and update safety counters
   */
  recordTrade(
    symbol: string,
    side: 'BUY' | 'SELL',
    entryPrice: number,
    exitPrice: number,
    quantity: number
  ) {
    const pnl =
      side === 'BUY'
        ? (exitPrice - entryPrice) * quantity
        : (entryPrice - exitPrice) * quantity;

    this.tradeHistory.push({
      symbol,
      side,
      entryPrice,
      exitPrice,
      quantity,
      pnl,
      timestamp: Date.now(),
    });

    // Update Safety Counters
    if (pnl < 0) {
      this.consecutiveLosses++;
      if (this.consecutiveLosses >= this.maxConsecutiveLosses) {
        this.cooldownEndTime = Date.now() + this.cooldownDuration;
        this.consecutiveLosses = 0; // Reset after triggering cooldown
      }
    } else {
      this.consecutiveLosses = 0; // Reset on win
    }
  }

  /**
   * Manual Emergency Stop
   */
  triggerKillSwitch() {
    this.isCircuitBreakerTripped = true;
  }

  /**
   * Reset Safety Locks (Manual Override)
   */
  resetSafety() {
    this.isCircuitBreakerTripped = false;
    this.cooldownEndTime = 0;
    this.consecutiveLosses = 0;
    // Reset initial equity to current if needed, or keep original? 
    // Usually keep original to prevent "revenge trading" immediately.
  }

  /**
   * Get trade history
   */
  getTradeHistory() {
    return this.tradeHistory;
  }

  /**
   * Set max risk per trade (percentage)
   */
  setMaxRiskPerTrade(percentage: number) {
    this.maxRiskPerTrade = percentage / 100;
  }

  /**
   * Set max drawdown tolerance
   */
  setMaxDrawdown(percentage: number) {
    this.maxDrawdown = percentage / 100;
  }

  /**
   * Update account balance
   */
  updateAccountBalance(newBalance: number) {
    this.accountBalance = newBalance;
  }

  /**
   * Get current account balance
   */
  getAccountBalance(): number {
    return this.accountBalance;
  }

  /**
   * Calculate Dynamic Position Size (The Gas Pedal)
   * Scales risk based on confidence.
   */
  calculateDynamicPositionSize(
    currentPrice: number,
    stopLossPercent: number,
    confidenceMultiplier: number
  ): number {
    // 1. Base Risk Amount (e.g., $10,000 * 2% = $200)
    const baseRiskAmount = this.accountBalance * this.maxRiskPerTrade;

    // 2. Adjust Risk by Confidence (e.g., $200 * 1.5 = $300)
    // Limit multiplier between 0.5x (Low Conf) and 2.0x (High Conf)
    const safeMultiplier = Math.max(0.5, Math.min(confidenceMultiplier, 2.0));
    const adjustedRiskAmount = baseRiskAmount * safeMultiplier;

    // 3. Calculate Position Size based on SL Distance
    // Size = Risk / (Price * SL%)
    const stopLossDistance = currentPrice * (stopLossPercent / 100);

    if (stopLossDistance === 0) return 0;

    const positionSize = adjustedRiskAmount / stopLossDistance;

    return positionSize;
  }

  // ========================================================================
  // POSITION & PORTFOLIO MANAGEMENT
  // ========================================================================

  /**
   * Calculate stop-loss and take-profit prices from entry price
   */
  setRiskReward(
    entryPrice: number,
    riskPercent: number,
    rewardMultiplier: number
  ): { stopLoss: number; takeProfit: number } {
    const stopLoss = entryPrice * (1 - riskPercent);
    const takeProfit = entryPrice * (1 + riskPercent * rewardMultiplier);
    return { stopLoss, takeProfit };
  }

  /**
   * Calculate position size so max loss = maxRiskPerTrade * accountBalance
   */
  calculatePositionSizeFixedRisk(
    entryPrice: number,
    stopLossPrice: number
  ): number {
    const riskAmount = this.accountBalance * this.maxRiskPerTrade;
    const riskPerUnit = Math.abs(entryPrice - stopLossPrice);

    if (riskPerUnit === 0) return 0;

    const positionSize = riskAmount / riskPerUnit;
    // Cap at 95% of balance to leave room for fees
    const maxAffordable = (this.accountBalance * 0.95) / entryPrice;

    return Math.min(positionSize, maxAffordable);
  }

  /**
   * Check if a new position can be opened (balance + safety layer)
   */
  canOpenPosition(cost: number): boolean {
    // 1. Check safety layer (circuit breaker, cooldown, kill switch)
    const safetyCheck = this.isTradingAllowed(this.accountBalance);
    if (!safetyCheck.allowed) {
      console.warn(`[RiskManager] Trading blocked: ${safetyCheck.reason}`);
      return false;
    }

    // 2. Check available balance
    const usedBalance = this.positions.reduce(
      (sum, p) => sum + p.entryPrice * p.quantity,
      0
    );
    const availableBalance = this.accountBalance - usedBalance;

    return cost <= availableBalance;
  }

  /**
   * Add a position to the tracked list
   */
  addPosition(position: Position): void {
    this.positions.push(position);
  }

  /**
   * Remove the first position matching a symbol
   */
  removePosition(symbol: string): void {
    const index = this.positions.findIndex((p) => p.symbol === symbol);
    if (index !== -1) {
      this.positions.splice(index, 1);
    }
  }

  /**
   * Get all open positions
   */
  getPositions(): Position[] {
    return [...this.positions];
  }

  /**
   * Check if a position should be closed (stop-loss or take-profit hit)
   */
  shouldClosePosition(
    position: Position,
    currentPrice: number
  ): { shouldClose: boolean; reason: string } {
    // Check stop-loss
    if (
      position.stopLossPrice &&
      position.side === 'LONG' &&
      currentPrice <= position.stopLossPrice
    ) {
      return { shouldClose: true, reason: 'Stop-Loss Hit' };
    }
    if (
      position.stopLossPrice &&
      position.side === 'SHORT' &&
      currentPrice >= position.stopLossPrice
    ) {
      return { shouldClose: true, reason: 'Stop-Loss Hit' };
    }

    // Check take-profit
    if (
      position.takeProfitPrice &&
      position.side === 'LONG' &&
      currentPrice >= position.takeProfitPrice
    ) {
      return { shouldClose: true, reason: 'Take-Profit Hit' };
    }
    if (
      position.takeProfitPrice &&
      position.side === 'SHORT' &&
      currentPrice <= position.takeProfitPrice
    ) {
      return { shouldClose: true, reason: 'Take-Profit Hit' };
    }

    // Check trailing stop
    if (position.trailingStop && position.side === 'LONG') {
      const trailPrice = currentPrice * (1 - position.trailingStop / 100);
      if (currentPrice <= trailPrice) {
        return { shouldClose: true, reason: 'Trailing Stop Hit' };
      }
    }

    return { shouldClose: false, reason: '' };
  }

  /**
   * Calculate full portfolio metrics
   */
  calculatePortfolioMetrics(
    currentPrices: Record<string, number>
  ): PortfolioMetrics {
    const usedBalance = this.positions.reduce(
      (sum, p) => sum + p.entryPrice * p.quantity,
      0
    );
    const availableBalance = this.accountBalance - usedBalance;

    // Unrealized P&L
    let unrealizedPnL = 0;
    for (const pos of this.positions) {
      const price = currentPrices[pos.symbol] ?? pos.entryPrice;
      if (pos.side === 'LONG') {
        unrealizedPnL += (price - pos.entryPrice) * pos.quantity;
      } else {
        unrealizedPnL += (pos.entryPrice - price) * pos.quantity;
      }
    }

    const equity = this.accountBalance + unrealizedPnL;

    // Realized P&L from trade history
    const realizedPnL = this.tradeHistory.reduce(
      (sum: number, t: any) => sum + (t.pnl ?? 0),
      0
    );

    // Win rate
    const wins = this.tradeHistory.filter((t: any) => t.pnl > 0).length;
    const winRate =
      this.tradeHistory.length > 0 ? wins / this.tradeHistory.length : 0;

    // Profit factor
    const grossProfit = this.tradeHistory
      .filter((t: any) => t.pnl > 0)
      .reduce((s: number, t: any) => s + t.pnl, 0);
    const grossLoss = Math.abs(
      this.tradeHistory
        .filter((t: any) => t.pnl < 0)
        .reduce((s: number, t: any) => s + t.pnl, 0)
    );
    const profitFactor =
      grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 100 : 0;

    // Max drawdown from balance history
    const maxDrawdownCalc =
      (this.initialEquity - equity) / this.initialEquity;

    // Margin (sum of position costs)
    const margin = usedBalance;
    const marginLevel = margin > 0 ? (equity / margin) * 100 : 0;

    return {
      totalBalance: this.accountBalance,
      usedBalance,
      availableBalance,
      equity,
      margin,
      marginLevel,
      unrealizedPnL,
      realizedPnL,
      totalPnL: realizedPnL + unrealizedPnL,
      winRate,
      profitFactor,
      maxDrawdown: Math.max(0, maxDrawdownCalc),
    };
  }
}
