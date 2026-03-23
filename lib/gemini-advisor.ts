
import { GoogleGenerativeAI } from "@google/generative-ai";
import { IndicatorData } from "./technical-indicators";
import { MarketAnalysis } from "./market-analyzer";

export class GeminiAdvisor {
    private genAI: GoogleGenerativeAI;
    private model: any;

    constructor(apiKey: string) {
        // Upgraded to Gemini 2.5 Flash as per user access
        this.genAI = new GoogleGenerativeAI(apiKey);
        this.model = this.genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    }

    async getMarketAdvice(indicators: IndicatorData, analysis: MarketAnalysis): Promise<string> {
        try {
            const prompt = `
            Act as a Senior Crypto Trading Strategist.
            Analyze the following Technical Indicators for a crypto asset:

            **Market Data:**
            - Price (Close): ${indicators.close}
            - RSI (14): ${indicators.rsi14.toFixed(2)}
            - ADX (14): ${indicators.adx.toFixed(2)}
            - ATR (14): ${indicators.atr14.toFixed(4)}
            - SMA(20): ${indicators.sma20.toFixed(2)}
            - SMA(50): ${indicators.sma50.toFixed(2)}
            - SMA(200): ${indicators.sma200.toFixed(2)}
            - Bollinger Width: ${((indicators.bollingerUpper - indicators.bollingerLower) / indicators.bollingerMiddle).toFixed(4)}

            **Automated Analysis:**
            - Trend: ${analysis.trend}
            - Volatility: ${analysis.volatility}
            - Current Strategy: ${analysis.recommendedStrategy}

            **Task:**
            1. Confirm if the "Current Strategy" is appropriate for these conditions.
            2. Identify any major risks (e.g., divergence, overbought RSI).
            3. Provide a concise, actionable recommendation (Buy/Sell/Hold/Wait).

            Keep the response under 100 words. Use bullet points.
            `;

            const result = await this.model.generateContent(prompt);
            const response = await result.response;
            const rawAdvice = response.text();

            // Guardrails: Validate LLM output
            return this.applyGuardrails(rawAdvice, indicators, analysis);

        } catch (error: any) {
            console.error("Gemini API Error:", error);
            // Fallback to local rule-based analysis
            return this.generateFallbackAdvice(indicators, analysis);
        }
    }

    private generateFallbackAdvice(indicators: IndicatorData, analysis: MarketAnalysis): string {
        const isBullish = analysis.trend === 'STRONG_UPTREND' || analysis.trend === 'WEAK_UPTREND';
        const isBearish = analysis.trend === 'STRONG_DOWNTREND' || analysis.trend === 'WEAK_DOWNTREND';
        const rsiStatus = indicators.rsi14 > 70 ? "Overbought" : indicators.rsi14 < 30 ? "Oversold" : "Neutral";

        // Determine action based on simple logic
        let action = "HOLD";
        if (isBullish && rsiStatus !== "Overbought") action = "BUY";
        if (isBearish && rsiStatus !== "Oversold") action = "SELL";
        if (rsiStatus === "Overbought" && isBullish) action = "TAKE PROFIT / WAIT";

        return `**🔮 Offline Mode (API Unavailable)**
        
*   **Analysis**: The local system detects a **${analysis.trend.replace('_', ' ')}** market structure with **${analysis.volatility}** volatility.
*   **Indicators**: RSI is **${indicators.rsi14.toFixed(1)}** (${rsiStatus}) and ADX is ${indicators.adx.toFixed(1)} (Trend Strength).
*   **Strategy**: The current **${analysis.recommendedStrategy}** strategy is well-suited for these conditions.
*   **Recommendation**: **${action}**. Monitor the ${indicators.sma20.toFixed(2)} support/resistance level.

_(Note: Enable 'Generative Language API' in Google Cloud to unlock full AI analysis)_`;
    }

    /**
     * Guardrails: Validate and sanitize LLM output before displaying to user.
     * - Strips hallucinated price targets and percentage claims
     * - Constrains recommended action to known values
     * - Adds disclaimer prefix
     * - Truncates overly long responses
     */
    private applyGuardrails(rawAdvice: string, indicators: IndicatorData, analysis: MarketAnalysis): string {
        const VALID_ACTIONS = ['BUY', 'SELL', 'HOLD', 'WAIT', 'TAKE PROFIT'];
        const MAX_LENGTH = 800;

        // Truncate overly long responses
        let advice = rawAdvice.length > MAX_LENGTH
            ? rawAdvice.substring(0, MAX_LENGTH) + '...'
            : rawAdvice;

        // Strip hallucinated specific price targets (e.g. "price will reach $75,000")
        advice = advice.replace(
            /(?:price|target|will reach|should hit|expect)\s*(?:of\s*)?\$[\d,.]+/gi,
            '[price target removed]'
        );

        // Strip hallucinated percentage predictions (e.g. "90% chance of...")
        advice = advice.replace(
            /\b\d{1,3}%\s*(?:chance|probability|likelihood|certainty|confident)\b/gi,
            '[probability claim removed]'
        );

        // Validate that the recommendation matches known actions
        const upperAdvice = advice.toUpperCase();
        const hasValidAction = VALID_ACTIONS.some(action => upperAdvice.includes(action));
        if (!hasValidAction) {
            // If LLM didn't provide a clear recommendation, append fallback
            advice += '\n\n**⚠️ AI did not provide a clear action. Defaulting to HOLD.**';
        }

        // Always prepend disclaimer
        const disclaimer = '**⚠️ AI Advisory (Not Financial Advice)** — Review with your own analysis.\n\n';
        return disclaimer + advice;
    }
}

