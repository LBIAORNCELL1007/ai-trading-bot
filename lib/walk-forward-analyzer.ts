import { BinanceKline } from "./binance-websocket";

// ════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════

export interface ParameterSet {
    id: string;
    strategy: string;
    params: Record<string, number>;
}

export interface BacktestResult {
    parameterSet: ParameterSet;
    inSamplePnL: number;
    outOfSamplePnL: number;
    inSamplePnLPercent: number;
    outOfSamplePnLPercent: number;
    tradesInSample: number;
    tradesOutOfSample: number;
    winRateInSample: number;
    winRateOutOfSample: number;
    isRobust: boolean;
    plateauScore: number;
}

export interface WFAReport {
    symbol: string;
    timeframe: string;
    totalCandlesUsed: number;
    inSampleSize: number;
    outOfSampleSize: number;
    strategiesTested: number;
    paramCombinationsTested: number;
    bestResult: BacktestResult | null;
    topResults: BacktestResult[]; // Top 5
    allResults: BacktestResult[];
    timestamp: Date;
}

// ════════════════════════════════════════════════════════════════════════════
// WALK-FORWARD ANALYZER
// ════════════════════════════════════════════════════════════════════════════

export class WalkForwardAnalyzer {

    /**
     * Split data into in-sample (train) and out-of-sample (validate).
     */
    static splitData(data: BinanceKline[], trainRatio: number = 0.8) {
        const splitIndex = Math.floor(data.length * trainRatio);
        return {
            inSample: data.slice(0, splitIndex),
            outOfSample: data.slice(splitIndex)
        };
    }

    /**
     * Main entry point — runs WFA across multiple strategies and parameters.
     * Returns a rich report with ranked results.
     */
    static runAnalysis(
        klines: BinanceKline[],
        symbol: string = 'UNKNOWN',
        timeframe: string = '15m'
    ): WFAReport {
        const { inSample, outOfSample } = this.splitData(klines, 0.8);

        const report: WFAReport = {
            symbol,
            timeframe,
            totalCandlesUsed: klines.length,
            inSampleSize: inSample.length,
            outOfSampleSize: outOfSample.length,
            strategiesTested: 0,
            paramCombinationsTested: 0,
            bestResult: null,
            topResults: [],
            allResults: [],
            timestamp: new Date(),
        };

        if (inSample.length < 50 || outOfSample.length < 20) {
            return report;
        }

        const allResults: BacktestResult[] = [];

        // ── Strategy 1: Mean Reversion (Bollinger Band std dev) ──
        for (let dev = 1.5; dev <= 3.0; dev += 0.25) {
            const paramSet: ParameterSet = {
                id: `MeanRev-BB(${dev.toFixed(2)})`,
                strategy: 'MEAN_REVERSION',
                params: { stdDev: Number(dev.toFixed(2)), period: 20 }
            };
            const is = this.simulateMeanReversion(inSample, dev, 20);
            const os = this.simulateMeanReversion(outOfSample, dev, 20);
            allResults.push(this.buildResult(paramSet, is, os, inSample, outOfSample));
        }
        report.strategiesTested++;

        // ── Strategy 2: MA Crossover (fast/slow period combos) ──
        const fastPeriods = [5, 10, 15, 20];
        const slowPeriods = [30, 50, 100];
        for (const fast of fastPeriods) {
            for (const slow of slowPeriods) {
                if (fast >= slow) continue;
                const paramSet: ParameterSet = {
                    id: `MA-Cross(${fast}/${slow})`,
                    strategy: 'MA_CROSSOVER',
                    params: { fastPeriod: fast, slowPeriod: slow }
                };
                const is = this.simulateMovingAvgCross(inSample, fast, slow);
                const os = this.simulateMovingAvgCross(outOfSample, fast, slow);
                allResults.push(this.buildResult(paramSet, is, os, inSample, outOfSample));
            }
        }
        report.strategiesTested++;

        // ── Strategy 3: RSI Reversal (overbought/oversold thresholds) ──
        for (let ob = 65; ob <= 80; ob += 5) {
            for (let os_thresh = 20; os_thresh <= 35; os_thresh += 5) {
                const paramSet: ParameterSet = {
                    id: `RSI(${os_thresh}/${ob})`,
                    strategy: 'RSI_DIVERGENCE',
                    params: { oversold: os_thresh, overbought: ob, period: 14 }
                };
                const is = this.simulateRSI(inSample, os_thresh, ob, 14);
                const os = this.simulateRSI(outOfSample, os_thresh, ob, 14);
                allResults.push(this.buildResult(paramSet, is, os, inSample, outOfSample));
            }
        }
        report.strategiesTested++;

        // ── Strategy 4: MACD (signal period variations) ──
        const macdConfigs = [
            { fast: 8, slow: 21, signal: 5 },
            { fast: 12, slow: 26, signal: 9 },
            { fast: 5, slow: 35, signal: 5 },
            { fast: 10, slow: 20, signal: 7 },
        ];
        for (const cfg of macdConfigs) {
            const paramSet: ParameterSet = {
                id: `MACD(${cfg.fast}/${cfg.slow}/${cfg.signal})`,
                strategy: 'MACD',
                params: { fastPeriod: cfg.fast, slowPeriod: cfg.slow, signalPeriod: cfg.signal }
            };
            const is = this.simulateMACD(inSample, cfg.fast, cfg.slow, cfg.signal);
            const os = this.simulateMACD(outOfSample, cfg.fast, cfg.slow, cfg.signal);
            allResults.push(this.buildResult(paramSet, is, os, inSample, outOfSample));
        }
        report.strategiesTested++;

        // ── Strategy 5: Breakout (ATR multiplier for channel) ──
        for (let mult = 1.0; mult <= 3.0; mult += 0.5) {
            const paramSet: ParameterSet = {
                id: `Breakout-ATR(${mult.toFixed(1)}x)`,
                strategy: 'BOLLINGER_BREAKOUT',
                params: { atrMultiplier: mult, period: 14 }
            };
            const is = this.simulateBreakout(inSample, mult, 14);
            const os = this.simulateBreakout(outOfSample, mult, 14);
            allResults.push(this.buildResult(paramSet, is, os, inSample, outOfSample));
        }
        report.strategiesTested++;

        report.paramCombinationsTested = allResults.length;
        report.allResults = allResults;

        // Calculate plateau scores for robustness
        this.calculatePlateauScores(allResults);

        // Sort by composite score: OOS PnL% + plateau score (favors consistent + profitable)
        const sorted = [...allResults]
            .filter(r => r.outOfSamplePnL > 0 && r.inSamplePnL > 0)
            .sort((a, b) => {
                const scoreA = a.outOfSamplePnLPercent + a.plateauScore * 0.5;
                const scoreB = b.outOfSamplePnLPercent + b.plateauScore * 0.5;
                return scoreB - scoreA;
            });

        report.topResults = sorted.slice(0, 5);
        report.bestResult = sorted[0] || null;

        return report;
    }

    // ════════════════════════════════════════════════════════════════
    // STRATEGY SIMULATORS
    // ════════════════════════════════════════════════════════════════

    private static buildResult(
        paramSet: ParameterSet,
        is: SimResult,
        os: SimResult,
        inSample: BinanceKline[],
        outOfSample: BinanceKline[]
    ): BacktestResult {
        const isStartPrice = inSample[0]?.close || 1;
        const osStartPrice = outOfSample[0]?.close || 1;
        return {
            parameterSet: paramSet,
            inSamplePnL: is.pnl,
            outOfSamplePnL: os.pnl,
            inSamplePnLPercent: (is.pnl / isStartPrice) * 100,
            outOfSamplePnLPercent: (os.pnl / osStartPrice) * 100,
            tradesInSample: is.trades,
            tradesOutOfSample: os.trades,
            winRateInSample: is.trades > 0 ? (is.wins / is.trades) * 100 : 0,
            winRateOutOfSample: os.trades > 0 ? (os.wins / os.trades) * 100 : 0,
            isRobust: os.pnl > 0 && is.pnl > 0,
            plateauScore: 0,
        };
    }

    /**
     * Mean Reversion using Bollinger Bands.
     */
    private static simulateMeanReversion(
        klines: BinanceKline[], stdDevMultiplier: number, period: number = 20
    ): SimResult {
        if (klines.length < period) return { pnl: 0, trades: 0, wins: 0 };
        const closes = klines.map(k => k.close);
        let pnl = 0, trades = 0, wins = 0;
        let position: number | null = null;

        for (let i = period; i < closes.length; i++) {
            const window = closes.slice(i - period, i);
            const sma = window.reduce((s, v) => s + v, 0) / period;
            const variance = window.reduce((s, v) => s + (v - sma) ** 2, 0) / period;
            const std = Math.sqrt(variance);
            const upper = sma + stdDevMultiplier * std;
            const lower = sma - stdDevMultiplier * std;
            const price = closes[i];

            if (position === null && price < lower) {
                position = price;
            } else if (position !== null && price > upper) {
                const tradePnl = price - position;
                pnl += tradePnl;
                trades++;
                if (tradePnl > 0) wins++;
                position = null;
            }
        }
        if (position !== null) {
            const tradePnl = closes[closes.length - 1] - position;
            pnl += tradePnl; trades++;
            if (tradePnl > 0) wins++;
        }
        return { pnl, trades, wins };
    }

    /**
     * Moving Average Crossover.
     */
    private static simulateMovingAvgCross(
        klines: BinanceKline[], fastPeriod: number, slowPeriod: number
    ): SimResult {
        if (klines.length < slowPeriod + 1) return { pnl: 0, trades: 0, wins: 0 };
        const closes = klines.map(k => k.close);
        let pnl = 0, trades = 0, wins = 0;
        let position: number | null = null;
        let positionSide: 'long' | null = null;

        const sma = (arr: number[], p: number, end: number) => {
            const slice = arr.slice(end - p, end);
            return slice.reduce((s, v) => s + v, 0) / p;
        };

        let prevFast = sma(closes, fastPeriod, slowPeriod);
        let prevSlow = sma(closes, slowPeriod, slowPeriod);

        for (let i = slowPeriod + 1; i < closes.length; i++) {
            const fast = sma(closes, fastPeriod, i);
            const slow = sma(closes, slowPeriod, i);
            const price = closes[i];

            // Golden cross: fast crosses above slow
            if (prevFast <= prevSlow && fast > slow && position === null) {
                position = price;
                positionSide = 'long';
            }
            // Death cross: fast crosses below slow — exit long
            else if (prevFast >= prevSlow && fast < slow && position !== null) {
                const tradePnl = price - position;
                pnl += tradePnl;
                trades++;
                if (tradePnl > 0) wins++;
                position = null;
            }

            prevFast = fast;
            prevSlow = slow;
        }
        if (position !== null) {
            const tradePnl = closes[closes.length - 1] - position;
            pnl += tradePnl; trades++;
            if (tradePnl > 0) wins++;
        }
        return { pnl, trades, wins };
    }

    /**
     * RSI Reversal strategy.
     */
    private static simulateRSI(
        klines: BinanceKline[], oversold: number, overbought: number, period: number
    ): SimResult {
        if (klines.length < period + 2) return { pnl: 0, trades: 0, wins: 0 };
        const closes = klines.map(k => k.close);
        let pnl = 0, trades = 0, wins = 0;
        let position: number | null = null;

        for (let i = period + 1; i < closes.length; i++) {
            const rsi = this.calcRSI(closes, period, i);
            const price = closes[i];

            if (position === null && rsi < oversold) {
                position = price;
            } else if (position !== null && rsi > overbought) {
                const tradePnl = price - position;
                pnl += tradePnl;
                trades++;
                if (tradePnl > 0) wins++;
                position = null;
            }
        }
        if (position !== null) {
            const tradePnl = closes[closes.length - 1] - position;
            pnl += tradePnl; trades++;
            if (tradePnl > 0) wins++;
        }
        return { pnl, trades, wins };
    }

    /**
     * MACD crossover strategy.
     */
    private static simulateMACD(
        klines: BinanceKline[], fastP: number, slowP: number, signalP: number
    ): SimResult {
        if (klines.length < slowP + signalP + 1) return { pnl: 0, trades: 0, wins: 0 };
        const closes = klines.map(k => k.close);
        let pnl = 0, trades = 0, wins = 0;
        let position: number | null = null;

        // Build MACD line
        const macdLine: number[] = [];
        for (let i = 0; i < closes.length; i++) {
            const emaFast = this.calcEMA(closes, fastP, i + 1);
            const emaSlow = this.calcEMA(closes, slowP, i + 1);
            macdLine.push(emaFast - emaSlow);
        }

        // Build signal line (EMA of MACD)
        const signalLine: number[] = [];
        for (let i = 0; i < macdLine.length; i++) {
            signalLine.push(this.calcEMA(macdLine, signalP, i + 1));
        }

        let prevHist = macdLine[slowP] - signalLine[slowP];

        for (let i = slowP + 1; i < closes.length; i++) {
            const hist = macdLine[i] - signalLine[i];
            const price = closes[i];

            // Crossover: histogram flips positive
            if (prevHist <= 0 && hist > 0 && position === null) {
                position = price;
            }
            // Exit: histogram flips negative
            else if (prevHist >= 0 && hist < 0 && position !== null) {
                const tradePnl = price - position;
                pnl += tradePnl;
                trades++;
                if (tradePnl > 0) wins++;
                position = null;
            }
            prevHist = hist;
        }
        if (position !== null) {
            const tradePnl = closes[closes.length - 1] - position;
            pnl += tradePnl; trades++;
            if (tradePnl > 0) wins++;
        }
        return { pnl, trades, wins };
    }

    /**
     * ATR Breakout strategy.
     */
    private static simulateBreakout(
        klines: BinanceKline[], atrMult: number, period: number
    ): SimResult {
        if (klines.length < period + 2) return { pnl: 0, trades: 0, wins: 0 };
        let pnl = 0, trades = 0, wins = 0;
        let position: number | null = null;

        for (let i = period + 1; i < klines.length; i++) {
            // ATR
            let atrSum = 0;
            for (let j = i - period; j < i; j++) {
                const tr = Math.max(
                    klines[j].high - klines[j].low,
                    Math.abs(klines[j].high - klines[j - 1].close),
                    Math.abs(klines[j].low - klines[j - 1].close)
                );
                atrSum += tr;
            }
            const atr = atrSum / period;
            const prevClose = klines[i - 1].close;
            const price = klines[i].close;
            const upper = prevClose + atr * atrMult;
            const lower = prevClose - atr * atrMult;

            if (position === null && price > upper) {
                position = price;
            } else if (position !== null && price < lower) {
                const tradePnl = price - position;
                pnl += tradePnl;
                trades++;
                if (tradePnl > 0) wins++;
                position = null;
            }
        }
        if (position !== null) {
            const tradePnl = klines[klines.length - 1].close - position;
            pnl += tradePnl; trades++;
            if (tradePnl > 0) wins++;
        }
        return { pnl, trades, wins };
    }

    // ════════════════════════════════════════════════════════════════
    // HELPERS
    // ════════════════════════════════════════════════════════════════

    private static calcRSI(prices: number[], period: number, end: number): number {
        if (end < period + 1) return 50;
        let gains = 0, losses = 0;
        for (let i = end - period; i < end; i++) {
            const change = prices[i] - prices[i - 1];
            if (change > 0) gains += change;
            else losses += Math.abs(change);
        }
        const avgGain = gains / period;
        const avgLoss = losses / period;
        if (avgLoss === 0) return 100;
        const rs = avgGain / avgLoss;
        return 100 - 100 / (1 + rs);
    }

    private static calcEMA(prices: number[], period: number, end: number): number {
        const slice = prices.slice(0, end);
        if (slice.length < period) return slice[slice.length - 1] || 0;
        let ema = slice.slice(0, period).reduce((s, v) => s + v, 0) / period;
        const mult = 2 / (period + 1);
        for (let i = period; i < slice.length; i++) {
            ema = slice[i] * mult + ema * (1 - mult);
        }
        return ema;
    }

    /**
     * Calculate plateau scores — how stable a result is relative to its neighbors.
     */
    private static calculatePlateauScores(results: BacktestResult[]) {
        // Group by strategy
        const groups = new Map<string, BacktestResult[]>();
        for (const r of results) {
            const key = r.parameterSet.strategy;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key)!.push(r);
        }

        for (const [, group] of groups) {
            for (let i = 0; i < group.length; i++) {
                const current = group[i];
                let neighborSum = current.outOfSamplePnLPercent;
                let count = 1;

                if (i > 0) {
                    neighborSum += group[i - 1].outOfSamplePnLPercent;
                    count++;
                }
                if (i < group.length - 1) {
                    neighborSum += group[i + 1].outOfSamplePnLPercent;
                    count++;
                }
                current.plateauScore = neighborSum / count;
            }
        }
    }
}

interface SimResult {
    pnl: number;
    trades: number;
    wins: number;
}
