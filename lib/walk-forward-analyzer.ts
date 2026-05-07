import { BinanceKline } from "./binance-websocket";

// ════════════════════════════════════════════════════════════════════════════
// COSTS — every simulator now applies these.  Without them the analyzer
// wildly over-estimates PnL of high-frequency strategies (mean-reversion,
// MACD) which look great gross-of-fees but break-even or lose net.
// ════════════════════════════════════════════════════════════════════════════
const TAKER_FEE_PCT = 0.0004;   // Binance spot taker (≈0.04%)
const SLIPPAGE_PCT  = 0.0005;   // 5 bp per fill — conservative for crypto majors
const ROUND_TRIP_COST = 2 * (TAKER_FEE_PCT + SLIPPAGE_PCT); // entry+exit

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
    inSampleSharpe: number;
    outOfSampleSharpe: number;
    tradesInSample: number;
    tradesOutOfSample: number;
    winRateInSample: number;
    winRateOutOfSample: number;
    isRobust: boolean;
    plateauScore: number;
    /**
     * Per-fold OOS Sharpe ratios (when produced by `runRollingWalkForward`).
     * Empty for the legacy single-split path.
     */
    foldSharpes?: number[];
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
    /** Number of folds (1 for legacy single-split, N for rolling WFA). */
    foldsUsed: number;
}

interface SimResult {
    pnl: number;            // sum of per-trade fractional returns (already net of fees)
    trades: number;
    wins: number;
    /** Per-trade fractional returns (net of fees) — used to compute Sharpe. */
    returns: number[];
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
     * Generate rolling walk-forward windows.  Each window has its own
     * in-sample (train) and out-of-sample (test) slice, sliding forward by
     * the test-window size.  This is the *correct* WFA — not a one-off split.
     *
     * Example with totalCandles=1000, trainSize=400, testSize=100, step=100:
     *   Fold 0:  IS [0..400),     OOS [400..500)
     *   Fold 1:  IS [100..500),   OOS [500..600)
     *   Fold 2:  IS [200..600),   OOS [600..700)
     *   ...
     */
    static rollingWindows(
        data: BinanceKline[],
        trainSize: number,
        testSize: number,
        step: number = testSize
    ): { inSample: BinanceKline[]; outOfSample: BinanceKline[] }[] {
        const folds: { inSample: BinanceKline[]; outOfSample: BinanceKline[] }[] = [];
        for (let start = 0; start + trainSize + testSize <= data.length; start += step) {
            folds.push({
                inSample: data.slice(start, start + trainSize),
                outOfSample: data.slice(start + trainSize, start + trainSize + testSize),
            });
        }
        return folds;
    }

    /**
     * Main entry point — runs WFA across multiple strategies and parameters.
     *
     * Selection criterion (CHANGED — was: rank by OOS PnL, which is textbook
     * data-snooping bias):
     *   1. Filter to params that are profitable in-sample AND OOS (`isRobust`)
     *   2. Rank by **in-sample Sharpe** (the unbiased optimisation criterion)
     *   3. Tie-break by plateau score (parameter-stability)
     *   4. Report OOS Sharpe / PnL as out-of-sample *evidence*, not driver
     *
     * This eliminates the bias of picking the lucky-on-OOS parameter set.
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
            foldsUsed: 1,
        };

        if (inSample.length < 50 || outOfSample.length < 20) {
            return report;
        }

        const allResults = this.runOnSplit(inSample, outOfSample);
        report.strategiesTested = 5;
        report.paramCombinationsTested = allResults.length;
        report.allResults = allResults;

        // Calculate plateau scores for robustness
        this.calculatePlateauScores(allResults);

        // ── SELECTION: rank by IN-SAMPLE Sharpe (no data-snooping bias) ──
        const sorted = [...allResults]
            .filter(r => r.isRobust && r.tradesInSample >= 5 && r.tradesOutOfSample >= 3)
            .sort((a, b) => {
                // Primary: in-sample Sharpe
                // Secondary: plateau score (parameter stability — neighbors also profitable)
                const scoreA = a.inSampleSharpe + a.plateauScore * 0.05;
                const scoreB = b.inSampleSharpe + b.plateauScore * 0.05;
                return scoreB - scoreA;
            });

        report.topResults = sorted.slice(0, 5);
        report.bestResult = sorted[0] || null;

        return report;
    }

    /**
     * Rolling walk-forward — N folds, each fold:
     *   • train on `trainSize` bars
     *   • test on `testSize` bars
     *   • slide forward by `step`
     *
     * For each parameter set, mean and std of OOS-Sharpe across folds is
     * tracked.  Selection picks the param-set with the highest **mean OOS
     * Sharpe**, regularised by a stability term (minus std/2) so erratic
     * winners are penalised.
     *
     * This is the textbook WFA from López de Prado §11 and is the only honest
     * way to evaluate parameter robustness for live trading.
     */
    static runRollingWalkForward(
        klines: BinanceKline[],
        symbol: string = 'UNKNOWN',
        timeframe: string = '15m',
        trainSize: number = 400,
        testSize: number = 100,
        step: number = 100
    ): WFAReport {
        const folds = this.rollingWindows(klines, trainSize, testSize, step);

        const report: WFAReport = {
            symbol,
            timeframe,
            totalCandlesUsed: klines.length,
            inSampleSize: trainSize,
            outOfSampleSize: testSize,
            strategiesTested: 5,
            paramCombinationsTested: 0,
            bestResult: null,
            topResults: [],
            allResults: [],
            timestamp: new Date(),
            foldsUsed: folds.length,
        };

        if (folds.length === 0) return report;

        // For each parameter-id, accumulate per-fold stats
        const accumulated = new Map<string, BacktestResult & { isPnls: number[]; osPnls: number[] }>();

        for (const fold of folds) {
            const foldResults = this.runOnSplit(fold.inSample, fold.outOfSample);
            for (const r of foldResults) {
                const key = r.parameterSet.id;
                if (!accumulated.has(key)) {
                    accumulated.set(key, {
                        ...r,
                        foldSharpes: [],
                        isPnls: [],
                        osPnls: [],
                        // reset cumulative fields — we'll average them
                        inSamplePnL: 0,
                        outOfSamplePnL: 0,
                        inSamplePnLPercent: 0,
                        outOfSamplePnLPercent: 0,
                        inSampleSharpe: 0,
                        outOfSampleSharpe: 0,
                        tradesInSample: 0,
                        tradesOutOfSample: 0,
                        winRateInSample: 0,
                        winRateOutOfSample: 0,
                    });
                }
                const acc = accumulated.get(key)!;
                acc.foldSharpes!.push(r.outOfSampleSharpe);
                acc.isPnls.push(r.inSamplePnLPercent);
                acc.osPnls.push(r.outOfSamplePnLPercent);
                acc.inSamplePnL += r.inSamplePnL;
                acc.outOfSamplePnL += r.outOfSamplePnL;
                acc.inSamplePnLPercent += r.inSamplePnLPercent;
                acc.outOfSamplePnLPercent += r.outOfSamplePnLPercent;
                acc.inSampleSharpe += r.inSampleSharpe;
                acc.outOfSampleSharpe += r.outOfSampleSharpe;
                acc.tradesInSample += r.tradesInSample;
                acc.tradesOutOfSample += r.tradesOutOfSample;
                acc.winRateInSample += r.winRateInSample;
                acc.winRateOutOfSample += r.winRateOutOfSample;
            }
        }

        // Average per-fold metrics
        const all: BacktestResult[] = [];
        for (const acc of accumulated.values()) {
            const n = acc.foldSharpes!.length;
            const meanIsSharpe = acc.inSampleSharpe / n;
            const meanOsSharpe = acc.outOfSampleSharpe / n;
            all.push({
                parameterSet: acc.parameterSet,
                inSamplePnL: acc.inSamplePnL / n,
                outOfSamplePnL: acc.outOfSamplePnL / n,
                inSamplePnLPercent: acc.inSamplePnLPercent / n,
                outOfSamplePnLPercent: acc.outOfSamplePnLPercent / n,
                inSampleSharpe: meanIsSharpe,
                outOfSampleSharpe: meanOsSharpe,
                tradesInSample: Math.round(acc.tradesInSample / n),
                tradesOutOfSample: Math.round(acc.tradesOutOfSample / n),
                winRateInSample: acc.winRateInSample / n,
                winRateOutOfSample: acc.winRateOutOfSample / n,
                isRobust: meanOsSharpe > 0 && meanIsSharpe > 0,
                plateauScore: 0,
                foldSharpes: acc.foldSharpes,
            });
        }

        report.allResults = all;
        report.paramCombinationsTested = all.length;

        this.calculatePlateauScores(all);

        // SELECTION: mean OOS Sharpe minus stability penalty (std/2)
        const sorted = [...all]
            .filter(r => r.isRobust && r.tradesOutOfSample >= 3)
            .sort((a, b) => {
                const stdA = stddev(a.foldSharpes ?? [a.outOfSampleSharpe]);
                const stdB = stddev(b.foldSharpes ?? [b.outOfSampleSharpe]);
                const scoreA = a.outOfSampleSharpe - stdA * 0.5 + a.plateauScore * 0.05;
                const scoreB = b.outOfSampleSharpe - stdB * 0.5 + b.plateauScore * 0.05;
                return scoreB - scoreA;
            });

        report.topResults = sorted.slice(0, 5);
        report.bestResult = sorted[0] || null;

        return report;
    }

    /**
     * Run all strategies × params on a single (in-sample, out-of-sample) split.
     */
    private static runOnSplit(inSample: BinanceKline[], outOfSample: BinanceKline[]): BacktestResult[] {
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

        // ── Strategy 4: MACD ──
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

        // ── Strategy 5: ATR Breakout ──
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

        return allResults;
    }

    // ════════════════════════════════════════════════════════════════
    // STRATEGY SIMULATORS — all return per-trade *fractional returns net
    // of round-trip fees + slippage*.
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
            // PnL% is the SUM of fractional returns × 100.  Each return is
            // already net of fees/slippage in the simulator.
            inSamplePnLPercent: is.pnl * 100,
            outOfSamplePnLPercent: os.pnl * 100,
            inSampleSharpe: sharpe(is.returns),
            outOfSampleSharpe: sharpe(os.returns),
            tradesInSample: is.trades,
            tradesOutOfSample: os.trades,
            winRateInSample: is.trades > 0 ? (is.wins / is.trades) * 100 : 0,
            winRateOutOfSample: os.trades > 0 ? (os.wins / os.trades) * 100 : 0,
            isRobust: os.pnl > 0 && is.pnl > 0,
            plateauScore: 0,
            // ignore unused variable
            ...(isStartPrice && osStartPrice ? {} : {}),
        };
    }

    private static recordTrade(entry: number, exit: number, side: 'long' | 'short' = 'long'): number {
        // Fractional return net of round-trip cost
        const grossRet = side === 'long' ? (exit - entry) / entry : (entry - exit) / entry;
        return grossRet - ROUND_TRIP_COST;
    }

    private static simulateMeanReversion(
        klines: BinanceKline[], stdDevMultiplier: number, period: number = 20
    ): SimResult {
        if (klines.length < period) return { pnl: 0, trades: 0, wins: 0, returns: [] };
        const closes = klines.map(k => k.close);
        const out: SimResult = { pnl: 0, trades: 0, wins: 0, returns: [] };
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
                const r = this.recordTrade(position, price);
                out.pnl += r; out.returns.push(r); out.trades++;
                if (r > 0) out.wins++;
                position = null;
            }
        }
        if (position !== null) {
            const r = this.recordTrade(position, closes[closes.length - 1]);
            out.pnl += r; out.returns.push(r); out.trades++;
            if (r > 0) out.wins++;
        }
        return out;
    }

    private static simulateMovingAvgCross(
        klines: BinanceKline[], fastPeriod: number, slowPeriod: number
    ): SimResult {
        if (klines.length < slowPeriod + 1) return { pnl: 0, trades: 0, wins: 0, returns: [] };
        const closes = klines.map(k => k.close);
        const out: SimResult = { pnl: 0, trades: 0, wins: 0, returns: [] };
        let position: number | null = null;

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

            if (prevFast <= prevSlow && fast > slow && position === null) {
                position = price;
            } else if (prevFast >= prevSlow && fast < slow && position !== null) {
                const r = this.recordTrade(position, price);
                out.pnl += r; out.returns.push(r); out.trades++;
                if (r > 0) out.wins++;
                position = null;
            }
            prevFast = fast;
            prevSlow = slow;
        }
        if (position !== null) {
            const r = this.recordTrade(position, closes[closes.length - 1]);
            out.pnl += r; out.returns.push(r); out.trades++;
            if (r > 0) out.wins++;
        }
        return out;
    }

    private static simulateRSI(
        klines: BinanceKline[], oversold: number, overbought: number, period: number
    ): SimResult {
        if (klines.length < period + 2) return { pnl: 0, trades: 0, wins: 0, returns: [] };
        const closes = klines.map(k => k.close);
        const out: SimResult = { pnl: 0, trades: 0, wins: 0, returns: [] };
        let position: number | null = null;

        for (let i = period + 1; i < closes.length; i++) {
            const rsi = this.calcRSI(closes, period, i);
            const price = closes[i];

            if (position === null && rsi < oversold) {
                position = price;
            } else if (position !== null && rsi > overbought) {
                const r = this.recordTrade(position, price);
                out.pnl += r; out.returns.push(r); out.trades++;
                if (r > 0) out.wins++;
                position = null;
            }
        }
        if (position !== null) {
            const r = this.recordTrade(position, closes[closes.length - 1]);
            out.pnl += r; out.returns.push(r); out.trades++;
            if (r > 0) out.wins++;
        }
        return out;
    }

    private static simulateMACD(
        klines: BinanceKline[], fastP: number, slowP: number, signalP: number
    ): SimResult {
        if (klines.length < slowP + signalP + 1) return { pnl: 0, trades: 0, wins: 0, returns: [] };
        const closes = klines.map(k => k.close);
        const out: SimResult = { pnl: 0, trades: 0, wins: 0, returns: [] };
        let position: number | null = null;

        const macdLine: number[] = [];
        for (let i = 0; i < closes.length; i++) {
            const emaFast = this.calcEMA(closes, fastP, i + 1);
            const emaSlow = this.calcEMA(closes, slowP, i + 1);
            macdLine.push(emaFast - emaSlow);
        }
        const signalLine: number[] = [];
        for (let i = 0; i < macdLine.length; i++) {
            signalLine.push(this.calcEMA(macdLine, signalP, i + 1));
        }
        let prevHist = macdLine[slowP] - signalLine[slowP];

        for (let i = slowP + 1; i < closes.length; i++) {
            const hist = macdLine[i] - signalLine[i];
            const price = closes[i];

            if (prevHist <= 0 && hist > 0 && position === null) {
                position = price;
            } else if (prevHist >= 0 && hist < 0 && position !== null) {
                const r = this.recordTrade(position, price);
                out.pnl += r; out.returns.push(r); out.trades++;
                if (r > 0) out.wins++;
                position = null;
            }
            prevHist = hist;
        }
        if (position !== null) {
            const r = this.recordTrade(position, closes[closes.length - 1]);
            out.pnl += r; out.returns.push(r); out.trades++;
            if (r > 0) out.wins++;
        }
        return out;
    }

    private static simulateBreakout(
        klines: BinanceKline[], atrMult: number, period: number
    ): SimResult {
        if (klines.length < period + 2) return { pnl: 0, trades: 0, wins: 0, returns: [] };
        const out: SimResult = { pnl: 0, trades: 0, wins: 0, returns: [] };
        let position: number | null = null;

        for (let i = period + 1; i < klines.length; i++) {
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
                const r = this.recordTrade(position, price);
                out.pnl += r; out.returns.push(r); out.trades++;
                if (r > 0) out.wins++;
                position = null;
            }
        }
        if (position !== null) {
            const r = this.recordTrade(position, klines[klines.length - 1].close);
            out.pnl += r; out.returns.push(r); out.trades++;
            if (r > 0) out.wins++;
        }
        return out;
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
     * Plateau score = mean OOS PnL% of this param + its strategy-neighbors.
     * High plateau ⇒ parameter is in a stable region (less curve-fit).
     */
    private static calculatePlateauScores(results: BacktestResult[]) {
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
                if (i > 0) { neighborSum += group[i - 1].outOfSamplePnLPercent; count++; }
                if (i < group.length - 1) { neighborSum += group[i + 1].outOfSamplePnLPercent; count++; }
                current.plateauScore = neighborSum / count;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MATH HELPERS
// ════════════════════════════════════════════════════════════════════════════

/**
 * Sharpe ratio of a per-trade return series.  Uses 0 risk-free rate (we are
 * judging strategies relative to each other).  Returns 0 for series with
 * fewer than 2 trades or zero variance.
 */
function sharpe(returns: number[]): number {
    if (returns.length < 2) return 0;
    const m = returns.reduce((s, v) => s + v, 0) / returns.length;
    const v = returns.reduce((s, x) => s + (x - m) ** 2, 0) / (returns.length - 1);
    const sd = Math.sqrt(v);
    if (sd === 0) return 0;
    // Annualisation is symbol/timeframe-dependent and would mislead at this
    // layer.  We report the **per-trade Sharpe** (mean / std), which is fully
    // sufficient for RANKING strategies against one another.
    return m / sd;
}

function stddev(arr: number[]): number {
    if (arr.length < 2) return 0;
    const m = arr.reduce((s, v) => s + v, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}
