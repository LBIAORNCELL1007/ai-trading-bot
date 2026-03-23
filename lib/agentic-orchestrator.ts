import { MarketAnalysis, MarketTrend, VolatilityState } from './market-analyzer';
import { IndicatorData } from './technical-indicators';

export type MarketRegime =
    | 'BULL_TREND'      // Strong Uptown, Low/Normal Volatility
    | 'BEAR_TREND'      // Strong Downtrend, Low/Normal Volatility
    | 'RANGING'         // Sideways, Low Volatility
    | 'VOLATILE'        // High Volatility (Any Trend)
    | 'BREAKOUT'        // Sudden volatility expansion + Trend
    | 'UNCERTAIN';      // Conflicting signals

export interface StrategicDecision {
    regime: MarketRegime;
    recommendedStrategy: string;
    confidence: number;
    riskMultiplier: number; // 0.5x to 2.0x
    tpMultiplier: number;   // 1.0x to 3.0x for Volatility Scaling
    reasoning: string;
    shouldSwitch: boolean;
}

/**
 * @deprecated This orchestrator is not yet integrated into TradingEngine.
 * It runs standalone and its strategy recommendations are not applied.
 * TODO: Wire decide() into the trading loop.
 */
export class AgenticOrchestrator {
    private lastStrategy: string = 'MULTI_SIGNAL';
    private strategyPerformance: Map<string, { consecutiveLosses: number; lastLossTime: number }> = new Map();

    // Institutional Stickiness Variables
    private lastSwitchTime: number = 0;
    private readonly MIN_LINGER_TIME_MS: number = 5 * 60 * 1000; // 5 minutes minimum lock

    /**
     * The Brain: Decides on the best course of action based on all available data.
     */
    public decide(
        analysis: MarketAnalysis,
        indicators: IndicatorData,
        currentStrategy: string,
        pnl: number // Recent PnL to detect "Stagnation"
    ): StrategicDecision {
        const regime = this.detectRegime(analysis, indicators);
        const decision = this.selectStrategy(regime, analysis);

        // Feedback Loop: Downgrade confidence if losing money
        if (pnl < 0) {
            decision.confidence *= 0.8;
            decision.reasoning += " (Reduced confidence due to recent global PnL drop).";
        }

        // STRATEGY SPECIFIC FEEDBACK
        // Check if the recommended strategy is currently "in the penalty box"
        const stats = this.strategyPerformance.get(decision.recommendedStrategy);
        if (stats && stats.consecutiveLosses >= 3) {
            // Major Penalty: Is it time to give up on this strategy?
            decision.confidence *= 0.5; // Halve confidence
            decision.reasoning += ` ⚠️ PENALIZED: ${stats.consecutiveLosses} consecutive losses on this strategy.`;

            // If this drops confidence below 0.4, force switch to Multi-Signal (Safety)
            if (decision.confidence < 0.4) {
                decision.recommendedStrategy = 'MULTI_SIGNAL';
                decision.reasoning += " FORCED SAFETY: Switching to Multi-Signal Ensemble.";
                decision.confidence = 0.6; // Reset to moderate confidence for safety
                decision.shouldSwitch = true;
            }
        }

        // Stickiness: Prevent "Twitching" and Whipsaws
        if (decision.recommendedStrategy !== currentStrategy) {
            const timeSinceSwitch = Date.now() - this.lastSwitchTime;
            const isLingerActive = timeSinceSwitch < this.MIN_LINGER_TIME_MS;
            let allowedToSwitch = false;
            let rejectReason = '';

            // Rule 1: Linger Rule (Time Lock)
            if (isLingerActive && this.lastSwitchTime !== 0) {
                // Only allow breakout from Linger if confidence is absolute MAXIMUM
                if (decision.confidence >= 0.90) {
                    allowedToSwitch = true;
                } else {
                    allowedToSwitch = false;
                    const minLeft = Math.ceil((this.MIN_LINGER_TIME_MS - timeSinceSwitch) / 60000);
                    rejectReason = `Strategy Time-Locked (${minLeft}m remaining).`;
                }
            }
            // Rule 2: Switching Cost (Hysteresis)
            else {
                // Linger is over, but we still need high conviction to pay the "cost" of switching
                if (decision.confidence >= 0.75) {
                    allowedToSwitch = true;
                } else {
                    allowedToSwitch = false;
                    rejectReason = "Conviction too low (<75%) to justify switch cost.";
                }
            }

            if (!allowedToSwitch) {
                decision.recommendedStrategy = currentStrategy;
                decision.shouldSwitch = false;
                decision.reasoning += ` 🛡️ STICKINESS ALIVE: Overriding switch attempt. ${rejectReason}`;
            } else {
                decision.shouldSwitch = true;
                this.lastSwitchTime = Date.now(); // Record the new switch time!
            }
        } else {
            decision.shouldSwitch = false;
        }

        return decision;
    }

    private detectRegime(analysis: MarketAnalysis, data: IndicatorData): MarketRegime {
        // 1. Check for High Volatility First
        if (analysis.volatility === 'HIGH') {
            // Is it a breakout or just chop?
            if (analysis.adx > 30) return 'BREAKOUT';
            return 'VOLATILE';
        }

        // 2. Check for Strong Trends
        if (analysis.trend === 'STRONG_UPTREND') return 'BULL_TREND';
        if (analysis.trend === 'STRONG_DOWNTREND') return 'BEAR_TREND';

        // 3. Range Check
        if (analysis.trend === 'SIDEWAYS' && analysis.volatility === 'LOW') return 'RANGING';

        return 'UNCERTAIN';
    }

    private selectStrategy(regime: MarketRegime, analysis: MarketAnalysis): StrategicDecision {
        let strategy = 'MULTI_SIGNAL';
        let riskMultiplier = 1.0;
        let tpMultiplier = 1.0;
        let confidence = 0.5;
        let reasoning = '';

        switch (regime) {
            case 'BULL_TREND':
                strategy = 'MA_CROSSOVER'; // Ride the trend with moving averages
                confidence = 0.85;
                riskMultiplier = 1.2; // Increase size in strong trend
                reasoning = "Bull Market Detected. Riding Trend (MA Crossover).";
                break;

            case 'BEAR_TREND':
                strategy = 'MACD'; // Momentum usually leads drops
                confidence = 0.80;
                riskMultiplier = 1.0;
                reasoning = "Bear Market Detected. Using Momentum (MACD) to catch drops.";
                break;

            case 'RANGING':
                strategy = 'GRID'; // King of sideways
                confidence = 0.90;
                riskMultiplier = 1.0; // Standard risk
                reasoning = "Market is Ranging. Grid Trading is optimal to harvest volatility.";
                break;

            case 'VOLATILE':
                strategy = 'MEAN_REVERSION'; // Fade the extremes
                confidence = 0.70;
                riskMultiplier = 0.5; // HALVE RISK in chop
                reasoning = "High Volatility Choppiness. Fading extremes with reduced risk.";
                break;

            case 'BREAKOUT':
                strategy = 'MULTI_SIGNAL'; // Use ensemble for breakout confirmation
                confidence = 0.80;
                riskMultiplier = 1.5; // Aggressive sizing for breakouts
                reasoning = "Volatility Expansion Detected! Aggressive Breakout Entry (Ensemble).";
                break;

            case 'UNCERTAIN':
            default:
                strategy = 'MULTI_SIGNAL'; // Safety in numbers
                confidence = 0.5;
                riskMultiplier = 0.5; // Minimal risk
                reasoning = "Market Undefined. Using voting mechanism (Multi-Signal) for safety.";
                break;
        }

        // VOLATILITY SCALING (Sentiment Driven)
        // High Fear -> Wider price swings. Decrease risk size, but WIDEN take profit to catch the giant snap-backs.
        if (analysis.marketMood === 'EXTREME_FEAR') {
            riskMultiplier *= 0.7; // Reduce risk
            tpMultiplier = 2.0;    // Double the Take Profit distance
            reasoning += " [Volatility Scaling Active: Risk -30%, TP x2]";
        } else if (analysis.marketMood === 'EXTREME_GREED') {
            riskMultiplier *= 1.2; // Press the gas
            tpMultiplier = 1.5;    // Higher targets in blow-off tops
            reasoning += " [Volatility Scaling Active: Risk +20%, TP x1.5]";
        }

        return {
            regime,
            recommendedStrategy: strategy,
            confidence,
            riskMultiplier,
            tpMultiplier,
            reasoning,
            shouldSwitch: false // Calculated in main logic
        };
    }

    /**
     * Record the result of a closed trade to update the Feedback Loop
     */
    public recordTradeResult(strategy: string, pnl: number) {
        const stats = this.strategyPerformance.get(strategy) || { consecutiveLosses: 0, lastLossTime: 0 };

        if (pnl < 0) {
            stats.consecutiveLosses++;
            stats.lastLossTime = Date.now();
        } else {
            stats.consecutiveLosses = 0; // Reset on win!
        }

        this.strategyPerformance.set(strategy, stats);
        // Only log significantly
        if (stats.consecutiveLosses > 0) {
            console.log(`[Orchestrator] Feedback for ${strategy}: Loss (Consecutive: ${stats.consecutiveLosses})`);
        }
    }
}
