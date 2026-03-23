import { BinanceKline } from "./binance-websocket";
import { IndicatorData } from "./technical-indicators";

export interface ParameterSet {
    id: string;
    params: Record<string, number>;
}

export interface BacktestResult {
    parameterSet: ParameterSet;
    inSamplePnL: number;
    outOfSamplePnL: number;
    tradesInSample: number;
    tradesOutOfSample: number;
    isRobust: boolean; // Passed OOS check
    plateauScore: number; // How stable it is compared to neighbors
}

export class WalkForwardAnalyzer {

    /**
     * Splits data into Train (In-Sample) and Validate (Out-Of-Sample)
     * Default split is 80/20.
     */
    static splitData(data: BinanceKline[], trainRatio: number = 0.8) {
        const splitIndex = Math.floor(data.length * trainRatio);
        return {
            inSample: data.slice(0, splitIndex),
            outOfSample: data.slice(splitIndex)
        };
    }

    /**
     * Finds the 'Plateau' parameter. 
     * Instead of returning the highest absolute PnL, it returns the parameter 
     * that sits in the middle of a cluster of winning parameters.
     */
    static findMostRobustParameter(results: BacktestResult[], targetParam: string): BacktestResult | null {
        // Filter out those that failed Out-Of-Sample validation
        const robustCandidates = results.filter(r => r.outOfSamplePnL > 0 && r.inSamplePnL > 0);

        if (robustCandidates.length === 0) return null;

        // Sort candidates by the target parameter value to find neighbors
        robustCandidates.sort((a, b) => a.parameterSet.params[targetParam] - b.parameterSet.params[targetParam]);

        // Calculate Plateauness (average PnL of neighbors)
        for (let i = 0; i < robustCandidates.length; i++) {
            const current = robustCandidates[i];
            let neighborPnLSum = current.inSamplePnL + current.outOfSamplePnL;
            let count = 1;

            // Check left neighbor
            if (i > 0) {
                neighborPnLSum += robustCandidates[i - 1].inSamplePnL + robustCandidates[i - 1].outOfSamplePnL;
                count++;
            }
            // Check right neighbor
            if (i < robustCandidates.length - 1) {
                neighborPnLSum += robustCandidates[i + 1].inSamplePnL + robustCandidates[i + 1].outOfSamplePnL;
                count++;
            }

            current.plateauScore = neighborPnLSum / count;
        }

        // Return the one with the highest plateau score (most stable neighborhood)
        robustCandidates.sort((a, b) => b.plateauScore - a.plateauScore);

        return robustCandidates[0];
    }

    /**
     * Simulates a basic Grid or Mean Reversion strategy over historical data mapped to indicators.
     * This is a stub for the full backtest engine capability.
     */
    static runAnalysis(klines: BinanceKline[]): ParameterSet | null {
        // 1. Split Data
        const { inSample, outOfSample } = this.splitData(klines, 0.8);

        if (inSample.length < 50 || outOfSample.length < 20) {
            console.warn("Not enough data for Walk Forward Analysis. Need at least 100 candles.");
            return null;
        }

        // 2. Generate Parameter Combinations (e.g., Bollinger Band Standard Deviations)
        const paramSets: ParameterSet[] = [];
        for (let dev = 1.5; dev <= 3.0; dev += 0.1) {
            paramSets.push({ id: `bb-${dev.toFixed(1)}`, params: { stdDev: Number(dev.toFixed(1)) } });
        }

        const results: BacktestResult[] = [];

        // 3. Run a real mean-reversion backtest for each parameter set
        for (const set of paramSets) {
            const dev = set.params.stdDev;

            const isPnL = this.simulateMeanReversion(inSample, dev);
            const osPnL = this.simulateMeanReversion(outOfSample, dev);

            results.push({
                parameterSet: set,
                inSamplePnL: isPnL.pnl,
                outOfSamplePnL: osPnL.pnl,
                tradesInSample: isPnL.trades,
                tradesOutOfSample: osPnL.trades,
                isRobust: osPnL.pnl > 0,
                plateauScore: 0
            });
        }

        // 4. Find most robust parameter
        const best = this.findMostRobustParameter(results, 'stdDev');

        if (best) {
            console.log(`[WFA] Most Robust Parameter Found: stdDev = ${best.parameterSet.params.stdDev} (Plateau Score: ${best.plateauScore})`);
            return best.parameterSet;
        }

        return null;
    }

    /**
     * Simulate a mean-reversion strategy using Bollinger Bands.
     * Buy when price crosses below lower band, sell when above upper band.
     */
    private static simulateMeanReversion(
        klines: BinanceKline[],
        stdDevMultiplier: number,
        period: number = 20
    ): { pnl: number; trades: number } {
        if (klines.length < period) return { pnl: 0, trades: 0 };

        const closes = klines.map(k => k.close);
        let pnl = 0;
        let trades = 0;
        let position: number | null = null; // entry price when in a trade

        for (let i = period; i < closes.length; i++) {
            // Calculate SMA and standard deviation over rolling window
            const window = closes.slice(i - period, i);
            const sma = window.reduce((s, v) => s + v, 0) / period;
            const variance = window.reduce((s, v) => s + (v - sma) ** 2, 0) / period;
            const std = Math.sqrt(variance);

            const upper = sma + stdDevMultiplier * std;
            const lower = sma - stdDevMultiplier * std;
            const price = closes[i];

            if (position === null && price < lower) {
                // Buy signal
                position = price;
            } else if (position !== null && price > upper) {
                // Sell signal — close the trade
                pnl += price - position;
                trades++;
                position = null;
            }
        }

        // Close any open position at last price
        if (position !== null) {
            pnl += closes[closes.length - 1] - position;
            trades++;
        }

        return { pnl, trades };
    }
}
