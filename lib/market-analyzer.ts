
import { IndicatorData } from './technical-indicators';

export type MarketTrend = 'STRONG_UPTREND' | 'WEAK_UPTREND' | 'SIDEWAYS' | 'WEAK_DOWNTREND' | 'STRONG_DOWNTREND';
export type VolatilityState = 'HIGH' | 'NORMAL' | 'LOW';
export type MarketMood = 'EXTREME_FEAR' | 'FEAR' | 'NEUTRAL' | 'GREED' | 'EXTREME_GREED';

export interface MarketAnalysis {
    trend: MarketTrend;
    volatility: VolatilityState;
    marketMood: MarketMood;
    adx: number;
    atr: number;
    recommendedStrategy: string;
    reasoning: string;
    confidence: number;
}

export class MarketAnalyzer {
    static analyze(data: IndicatorData): MarketAnalysis {
        // --- 1. Enhanced Trend Analysis ---
        // ADX: Strength (>25 is strong)
        const adx = data.adx;

        // SMA Alignment: Long-term Direction
        const longTermBullish = data.sma50 > data.sma200;
        const shortTermBullish = data.sma20 > data.sma50;

        // VWAP: Institutional Bias (Price above VWAP = Bullish)
        const vwapBullish = data.close > (data.vwap || data.sma50); // Fallback to SMA50 if VWAP missing

        // Ichimoku Cloud: Trend Filter (Price above Cloud = Bullish)
        let cloudBullish = false;
        let cloudBearish = false;
        if (data.ichimoku) {
            const cloudTop = Math.max(data.ichimoku.senkouA, data.ichimoku.senkouB);
            const cloudBottom = Math.min(data.ichimoku.senkouA, data.ichimoku.senkouB);
            cloudBullish = data.close > cloudTop;
            cloudBearish = data.close < cloudBottom;
        }

        // MACD: Momentum (Histogram > 0 = Bullish Momentum)
        const macdBullish = data.histogram > 0;

        // Combine for Robust Trend Determination
        let trend: MarketTrend = 'SIDEWAYS';

        // Score: -5 (Strong Bear) to +5 (Strong Bull)
        let trendScore = 0;
        if (longTermBullish) trendScore += 1; else trendScore -= 1;
        if (shortTermBullish) trendScore += 1; else trendScore -= 1;
        if (vwapBullish) trendScore += 1; else trendScore -= 1;
        if (macdBullish) trendScore += 1; else trendScore -= 1;
        if (cloudBullish) trendScore += 1;
        if (cloudBearish) trendScore -= 1;

        // ADX Amplifier: Only if trend is present
        if (adx > 25) {
            if (trendScore >= 3) trend = 'STRONG_UPTREND';
            else if (trendScore <= -3) trend = 'STRONG_DOWNTREND';
            else if (trendScore > 0) trend = 'WEAK_UPTREND'; // Weak b/c ADX supports but signals mixed
            else trend = 'WEAK_DOWNTREND';
        } else {
            // Low ADX = Sideways usually, unless momentum is extreme
            if (trendScore >= 4) trend = 'WEAK_UPTREND'; // Early breakout?
            else if (trendScore <= -4) trend = 'WEAK_DOWNTREND'; // Early breakdown?
            else trend = 'SIDEWAYS';
        }

        // --- 2. Enhanced Volatility Analysis ---
        const bbWidth = (data.bollingerUpper - data.bollingerLower) / data.bollingerMiddle;
        // ATR relative to price (normalized volatility)
        const atrPercent = (data.atr14 / data.close) * 100;

        let volatility: VolatilityState = 'NORMAL';
        if (bbWidth > 0.05 || atrPercent > 1.5) volatility = 'HIGH'; // ATR > 1.5% is volatile 
        else if (bbWidth < 0.015 || atrPercent < 0.5) volatility = 'LOW'; // Low volatility

        // --- 2.5 Simulated Sentiment Engine (Market Mood) ---
        // Combining RSI, SMA distance, AND volatility for robust mood detection
        let marketMood: MarketMood = 'NEUTRAL';
        const distanceToSMA50 = data.sma50 !== 0 ? (data.close - data.sma50) / data.sma50 : 0;

        if (data.rsi14 > 70 && distanceToSMA50 > 0.05) marketMood = 'EXTREME_GREED';
        else if (data.rsi14 > 60 && distanceToSMA50 > 0.02) marketMood = 'GREED';
        else if (data.rsi14 < 30 && distanceToSMA50 < -0.05) marketMood = 'EXTREME_FEAR';
        else if (data.rsi14 < 40 && distanceToSMA50 < -0.02) marketMood = 'FEAR';

        // --- 3. Strategy Selection (Robust Matrix) ---
        let strategy = 'MULTI_SIGNAL';
        let reasoning = 'Market analysis inconclusive. Using Multi-Signal Ensemble.';
        let confidence = 0.5;

        // Sentiment Overrides
        if (marketMood === 'EXTREME_FEAR') {
            strategy = 'RSI_DIVERGENCE';
            reasoning = `EXTREME FEAR detected. Buying the blood (Mean Reversion) for bounce.`;
            confidence = 0.90;
        } else if (marketMood === 'EXTREME_GREED' && trend.includes('UP')) {
            strategy = 'BOLLINGER_BREAKOUT';
            reasoning = `EXTREME GREED detected in Uptrend. Riding the FOMO breakout.`;
            confidence = 0.90;
        }
        // A. TRENDING UP (Success Probability: High for Trend Following)
        else if (trend === 'STRONG_UPTREND' || trend === 'WEAK_UPTREND') {
            if (volatility === 'HIGH') {
                if (data.rsi14 > 80) {
                    strategy = 'RSI_DIVERGENCE';
                    reasoning = `Strong Uptrend but Overbought (RSI ${data.rsi14.toFixed(0)}). Expecting Pullback/Reversal.`;
                    confidence = 0.75;
                } else {
                    strategy = 'BOLLINGER_BREAKOUT';
                    reasoning = `Strong Momentum (MACD+VWAP) + High Volatility. Riding the breakout.`;
                    confidence = 0.85;
                }
            } else {
                if (cloudBullish) {
                    strategy = 'ICHIMOKU';
                    reasoning = 'Price above Cloud with steady momentum. Classic Ichimoku Trend Follow.';
                    confidence = 0.90;
                } else {
                    strategy = 'VWAP_TREND';
                    reasoning = 'Price respecting VWAP support. Institutional accumulation detected.';
                    confidence = 0.85;
                }
            }
        }
        // B. TRENDING DOWN
        else if (trend === 'STRONG_DOWNTREND' || trend === 'WEAK_DOWNTREND') {
            if (data.rsi14 < 20) {
                strategy = 'RSI_DIVERGENCE';
                reasoning = `Strong Downtrend but Oversold (RSI ${data.rsi14.toFixed(0)}). Creating long limit orders for bounce.`;
                confidence = 0.75;
            } else if (volatility === 'HIGH') {
                strategy = 'BOLLINGER_BREAKOUT'; // Downside
                reasoning = 'Bearish expansion (High Volatility). Shorting the breakdown.';
                confidence = 0.85;
            } else {
                strategy = 'VWAP_TREND'; // Shorting below VWAP
                reasoning = 'Price below VWAP and Cloud. Institutional distribution active.';
                confidence = 0.90;
            }
        }
        // C. SIDEWAYS / CHOP
        else {
            if (volatility === 'LOW') {
                strategy = 'GRID';
                reasoning = `Low Volatility (${(bbWidth * 100).toFixed(2)}% Width). Range-bound conditions ideal for Grid Trading.`;
                confidence = 0.85;
            } else {
                strategy = 'MEAN_REVERSION';
                reasoning = 'High Volatility but no Trend (Chop). Trading Bollinger Band edges (Mean Reversion).';
                confidence = 0.80;
            }
        }

        return {
            trend,
            volatility,
            marketMood,
            adx,
            atr: data.atr14,
            recommendedStrategy: strategy,
            reasoning,
            confidence
        };
    }
}
