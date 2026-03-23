"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { GeminiAdvisor } from "@/lib/gemini-advisor"
import { RiskManager } from "@/lib/risk-management"
import { AgenticOrchestrator, StrategicDecision } from "@/lib/agentic-orchestrator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Cell,
} from "recharts"
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Settings,
  Play,
  Pause,
  RotateCcw,
  AlertCircle,
  CheckCircle,
  Clock,
  DollarSign,
  Zap,
  Wifi,
  WifiOff,
  Brain,
  Sparkles,
  Skull,
  Unlock,
} from "lucide-react"
import {
  MovingAverageCrossoverStrategy,
  MACDStrategy,
  MeanReversionStrategy,
  GridTradingStrategy,
  MultiSignalStrategy,
  RSIDivergenceStrategy,
  BollingerBreakoutStrategy,
  VWAPTrendStrategy,
  IchimokuStrategy,
  PivotReversalStrategy,
  type StrategySignal
} from "@/lib/trading-strategies"
import {
  binanceWS,
  getHistoricalKlines,
  getCurrentPrice,
  type BinanceKline
} from "@/lib/binance-websocket"
import { MarketAnalyzer, type MarketAnalysis } from "@/lib/market-analyzer"
import { WalkForwardAnalyzer } from "@/lib/walk-forward-analyzer"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { FinancialChart } from "@/components/financial-chart"

interface TradeEntry {
  id: string
  symbol: string
  side: "BUY" | "SELL"
  entryPrice: number
  exitPrice?: number
  quantity: number
  entryTime: number
  exitTime?: number
  status: "OPEN" | "CLOSED" | "PENDING" | "CANCELLED"
  pnl?: number
  pnlPercent?: number
  strategy?: string
  leverage?: number
  stopLossPrice?: number
  takeProfitPrice?: number
  highestPrice?: number // For trailing stop
  lowestPrice?: number  // For trailing stop (short)
  limitPrice?: number   // For Limit-Chase entries
  signalPrice?: number  // The original price when the signal fired
}

interface PortfolioData {
  totalBalance: number
  usedBalance: number
  availableBalance: number
  equity: number
  unrealizedPnL: number
  realizedPnL: number
  totalPnL: number
  winRate: number
  profitFactor: number
  maxDrawdown: number
}

interface TradingConfig {
  symbol: string
  timeframe: string
  strategy: string
  riskPerTrade: number
  maxPositions: number
  stopLoss: number
  takeProfit: number
  trailingStop: number
  leverage: number
}

export default function BinanceTradingDashboard() {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState("overview")
  const [isRunning, setIsRunning] = useState(false)
  const [trades, setTrades] = useState<TradeEntry[]>([])
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    totalBalance: 10000,
    usedBalance: 0,
    availableBalance: 10000,
    equity: 10000,
    unrealizedPnL: 0,
    realizedPnL: 0,
    totalPnL: 0,
    winRate: 0,
    profitFactor: 0,
    maxDrawdown: 0,
  })
  const [balanceHistory, setBalanceHistory] = useState<Array<{ time: string; balance: number }>>([
    { time: new Date().toLocaleTimeString(), balance: 10000 },
  ])
  const [selectedSymbol, setSelectedSymbol] = useState("BTCUSDT")
  const [selectedTimeframe, setSelectedTimeframe] = useState("15m") // Default to 15m
  const [selectedStrategy, setSelectedStrategy] = useState("MULTI_SIGNAL")
  const [riskPerTrade, setRiskPerTrade] = useState(2)
  const [maxPositions, setMaxPositions] = useState(5)
  const [stopLoss, setStopLoss] = useState(2.0)
  const [takeProfit, setTakeProfit] = useState(4.0)
  const [trailingStop, setTrailingStop] = useState(1.0)
  const [leverage, setLeverage] = useState(1)
  const [showPairSearch, setShowPairSearch] = useState(false)
  const [strategyInstance, setStrategyInstance] = useState<any>(null)
  const [presetName, setPresetName] = useState("")
  const [savedPresets, setSavedPresets] = useState<{ [key: string]: TradingConfig }>({})
  const [showPresets, setShowPresets] = useState(false)
  const [currentPrice, setCurrentPrice] = useState<number>(0)
  const [priceLoading, setPriceLoading] = useState(false)
  const [candlestickData, setCandlestickData] = useState<BinanceKline[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const [searchTerm, setSearchTerm] = useState(selectedSymbol) // Decoupled input state

  // AI Mode State
  const [isAIEnabled, setIsAIEnabled] = useState(false)
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null)
  const [lastError, setLastError] = useState<string | null>(null)
  const [debugIndicators, setDebugIndicators] = useState<any>(null)
  const lastSwitchTime = useRef<number>(0)

  // Safety Layer State
  const [isSafetyLocked, setIsSafetyLocked] = useState(false)
  const [safetyReason, setSafetyReason] = useState<string | null>(null)
  /* AI State & Orchestration */
  const riskManagerRef = useRef(new RiskManager(10000))
  // Update RiskManager whenever riskPerTrade changes
  useEffect(() => {
    riskManagerRef.current.setMaxRiskPerTrade(riskPerTrade);
  }, [riskPerTrade]);

  // We need to persist the Orchestrator's decision (Risk Multiplier) to the trading loop
  // The Orchestrator runs in the AI Loop, but the Trade Execution happens in the Strategy Loop
  const lastRiskMultiplier = useRef<number>(1.0)
  const lastTpMultiplier = useRef<number>(1.0)
  const orchestratorRef = useRef(new AgenticOrchestrator())

  // Gemini state
  const [geminiAdvice, setGeminiAdvice] = useState<string | null>(null)
  const [isLoadingAdvice, setIsLoadingAdvice] = useState(false)
  const [isAdviceOpen, setIsAdviceOpen] = useState(false)

  // WFA Auto-Tune State
  const [isTuning, setIsTuning] = useState(false)
  const [tunedParams, setTunedParams] = useState<string | null>(null)

  const handleAskGemini = async () => {
    // Check for API Key in environment (safe client-side check for local app)
    // Note: Ideally this is a server action, but for local prototype direct call is faster.
    const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY;

    if (!apiKey || apiKey.length < 5) {
      toast({
        variant: "destructive",
        title: "API Key Missing",
        description: "Please add NEXT_PUBLIC_GEMINI_API_KEY to .env.local"
      })
      return
    }

    setIsLoadingAdvice(true)
    setIsAdviceOpen(true)
    setGeminiAdvice(null) // Reset

    try {
      if (typeof strategyInstance.getIndicators === 'function') {
        const indicators = strategyInstance.getIndicators();

        // Run quick analysis if not fresh
        const currentAnalysis = marketAnalysis || MarketAnalyzer.analyze(indicators);

        const advisor = new GeminiAdvisor(apiKey);
        const advice = await advisor.getMarketAdvice(indicators, currentAnalysis);
        setGeminiAdvice(advice);
      } else {
        setGeminiAdvice("Error: Strategy indicators not available.");
      }
    } catch (e: any) {
      console.error(e)
      setGeminiAdvice("Error reaching Gemini Service: " + e.message)
    } finally {
      setIsLoadingAdvice(false)
    }
  }

  // Handle WFA Auto-Tune
  const handleAutoTune = async () => {
    setIsTuning(true);
    setTunedParams(null);
    try {
      // Fetch 500 candles for WFA
      const data = await getHistoricalKlines(selectedSymbol, selectedTimeframe, 500);
      if (data.length < 100) {
        toast({ title: "Auto-Tune Failed", description: "Not enough historical data.", variant: "destructive" });
        return;
      }

      const bestParams = WalkForwardAnalyzer.runAnalysis(data);
      if (bestParams) {
        setTunedParams(`Robust Params Found: ${bestParams.id}`);
        toast({ title: "🛠️ Auto-Tune Complete", description: `Found stable plateau: ${bestParams.id}`, duration: 5000 });
      } else {
        toast({ title: "Auto-Tune Complete", description: "No robust parameters found in recent history.", variant: "destructive" });
      }
    } catch (e: any) {
      console.error(e);
      toast({ title: "Auto-Tune Error", description: e.message, variant: "destructive" });
    } finally {
      setIsTuning(false);
    }
  }

  // Sync searchTerm when selectedSymbol changes externally (e.g. loading preset)
  useEffect(() => {
    setSearchTerm(selectedSymbol)
  }, [selectedSymbol])

  // Helper to get readable strategy name
  const getStrategyName = (key: string) => {
    return {
      'MA_CROSSOVER': 'MA Crossover',
      'MACD': 'MACD',
      'MEAN_REVERSION': 'Mean Reversion',
      'GRID': 'Grid Trading',
      'MULTI_SIGNAL': 'Multi-Signal',
      'RSI_DIVERGENCE': 'RSI Divergence',
      'BOLLINGER_BREAKOUT': 'Bollinger Breakout',
      'VWAP_TREND': 'VWAP Trend',
      'ICHIMOKU': 'Ichimoku Cloud',
      'PIVOT_REVERSAL': 'Pivot Reversal'
    }[key] || key
  }

  // Popular trading pairs
  const TRADING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT"
  ]

  const filteredPairs = TRADING_PAIRS.filter(pair =>
    pair.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // Load configuration and presets from localStorage on mount
  useEffect(() => {
    const savedConfig = localStorage.getItem('tradingConfig')
    if (savedConfig) {
      try {
        const config: TradingConfig = JSON.parse(savedConfig)
        setSelectedSymbol(config.symbol)
        setSelectedTimeframe(config.timeframe)
        setSelectedStrategy(config.strategy)
        setRiskPerTrade(config.riskPerTrade)

        setMaxPositions(config.maxPositions)
        setStopLoss(config.stopLoss ?? 2.0)
        setTakeProfit(config.takeProfit ?? 4.0)
        setTrailingStop(config.trailingStop ?? 1.0)
        setLeverage(config.leverage ?? 1)
      } catch (e) {
        console.error('Failed to load config:', e)
      }
    }

    // Load saved presets
    const presets = localStorage.getItem('tradingPresets')
    if (presets) {
      try {
        setSavedPresets(JSON.parse(presets))
      } catch (e) {
        console.error('Failed to load presets:', e)
      }
    }
  }, [])




  // Refs for stable access in strategy effect
  const candlestickDataRef = useRef(candlestickData)
  const currentPriceRef = useRef(currentPrice)
  const portfolioRef = useRef(portfolio)
  const tradesRef = useRef(trades)

  useEffect(() => {
    candlestickDataRef.current = candlestickData
    currentPriceRef.current = currentPrice
    portfolioRef.current = portfolio
    tradesRef.current = trades
  }, [candlestickData, currentPrice, portfolio, trades])

  // Initialize strategy instance based on selected strategy
  // ONLY recreate when STRATEGY changes. NOT when data updates.
  useEffect(() => {
    let newStrategy: any
    const price = currentPriceRef.current || 0
    const history = candlestickDataRef.current

    switch (selectedStrategy) {
      case 'MA_CROSSOVER':
        newStrategy = new MovingAverageCrossoverStrategy()
        break
      case 'MACD':
        newStrategy = new MACDStrategy()
        break
      case 'MEAN_REVERSION':
        newStrategy = new MeanReversionStrategy()
        break
      case 'GRID':
        newStrategy = new GridTradingStrategy()
        newStrategy.initializeGrid(price > 0 ? price : 100, 10, 0.02)
        break
      case 'MULTI_SIGNAL':
        newStrategy = new MultiSignalStrategy(price > 0 ? price : 100)
        break
      case 'RSI_DIVERGENCE':
        newStrategy = new RSIDivergenceStrategy()
        break
      case 'BOLLINGER_BREAKOUT':
        newStrategy = new BollingerBreakoutStrategy()
        break
      case 'VWAP_TREND':
        newStrategy = new VWAPTrendStrategy()
        break
      case 'ICHIMOKU':
        newStrategy = new IchimokuStrategy()
        break
      case 'PIVOT_REVERSAL':
        newStrategy = new PivotReversalStrategy()
        break
      default:
        newStrategy = new MultiSignalStrategy(price > 0 ? price : 100)
    }

    // Initialize with historical data if available
    if (history.length > 0 && typeof newStrategy.initialize === 'function') {
      console.log(`[Strategy] Initializing ${selectedStrategy} with ${history.length} candles`)
      newStrategy.initialize(history)
    }

    setStrategyInstance(newStrategy)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStrategy]) // Critical: Only run on strategy change!

  // Initialize Grid Strategy when bot starts
  useEffect(() => {
    if (isRunning && currentPrice > 0 && strategyInstance) {
      if (selectedStrategy === 'GRID' && typeof strategyInstance.initializeGrid === 'function') {
        console.log(`[Dashboard] Initializing Grid Strategy at ${currentPrice}`)
        // DEMO MODE: 0.5% grid for faster trades (was 2%)
        strategyInstance.initializeGrid(currentPrice, 10, 0.005)
      } else if (selectedStrategy === 'MULTI_SIGNAL' && strategyInstance.gridStrategy) {
        // Best effort for multi-signal if accessible, otherwise it uses default
        // strategyInstance.gridStrategy.initializeGrid(currentPrice)
      }
    }
  }, [isRunning]) // Only run when start/stop is toggled

  // Save configuration handler
  const handleSaveConfiguration = () => {
    const config: TradingConfig = {
      symbol: selectedSymbol,
      timeframe: selectedTimeframe,
      strategy: selectedStrategy,
      riskPerTrade: riskPerTrade,
      maxPositions: maxPositions,
      stopLoss,
      takeProfit,
      trailingStop,
      leverage
    }

    localStorage.setItem('tradingConfig', JSON.stringify(config))

    const strategyName = getStrategyName(selectedStrategy)

    toast({
      title: "✅ Configuration Saved Successfully",
      description: `Your trading configuration has been saved. All future trades will be executed using:\n\n• Strategy: ${strategyName}\n• Trading Pair: ${selectedSymbol}\n• Timeframe: ${selectedTimeframe}\n• Risk: ${riskPerTrade}% per trade\n• Leverage: ${leverage}x\n• SL/TP: ${stopLoss}% / ${takeProfit}%`,
    })
  }

  // Save as preset
  const handleSaveAsPreset = () => {
    if (!presetName.trim()) {
      toast({
        title: "Preset Name Required",
        description: "Please enter a name for this preset.",
        variant: "destructive"
      })
      return
    }

    const config: TradingConfig = {
      symbol: selectedSymbol,
      timeframe: selectedTimeframe,
      strategy: selectedStrategy,
      riskPerTrade: riskPerTrade,
      maxPositions: maxPositions,
      stopLoss,
      takeProfit,
      trailingStop,
      leverage
    }

    const newPresets = { ...savedPresets, [presetName]: config }
    setSavedPresets(newPresets)
    localStorage.setItem('tradingPresets', JSON.stringify(newPresets))

    toast({
      title: "Preset Saved",
      description: `"${presetName}" has been saved to your presets.`,
    })

    setPresetName("")
  }

  // Load preset
  const handleLoadPreset = (name: string) => {
    const preset = savedPresets[name]
    if (preset) {
      setSelectedSymbol(preset.symbol)
      setSelectedTimeframe(preset.timeframe)
      setSelectedStrategy(preset.strategy)
      setRiskPerTrade(preset.riskPerTrade)
      setRiskPerTrade(preset.riskPerTrade)
      setMaxPositions(preset.maxPositions)
      setStopLoss(preset.stopLoss ?? 2.0)
      setTakeProfit(preset.takeProfit ?? 4.0)
      setTrailingStop(preset.trailingStop ?? 1.0)
      setLeverage(preset.leverage ?? 1)

      toast({
        title: "Preset Loaded",
        description: `"${name}" configuration has been loaded. Click "Save Configuration" to apply it.`,
      })
    }
  }

  // Delete preset
  const handleDeletePreset = (name: string) => {
    const newPresets = { ...savedPresets }
    delete newPresets[name]
    setSavedPresets(newPresets)
    localStorage.setItem('tradingPresets', JSON.stringify(newPresets))

    toast({
      title: "Preset Deleted",
      description: `"${name}" has been removed from your presets.`,
    })
  }

  // Calculate position size based on risk% and leverage
  const calculatePositionSize = (price: number) => {
    const riskAmount = (portfolio.availableBalance * riskPerTrade) / 100
    // Position size = Risk Amount / (Stop Loss distance per unit)
    // Adjusted for leverage: We want to risk X% of account, so if SL is hit, we lose X%.
    // Leverage increases position size, but risk management keeps valid risk.
    const stopLossDistance = price * (stopLoss / 100)
    const positionSize = riskAmount / stopLossDistance
    return positionSize
  }
  const [priceData, setPriceData] = useState<Array<{ time: string; price: number; volume: number }>>([])

  // Convert trading pair to CoinGecko ID
  const getCoinIdFromSymbol = (symbol: string): string => {
    const mapping: { [key: string]: string } = {
      'BTCUSDT': 'bitcoin',
      'ETHUSDT': 'ethereum',
      'BNBUSDT': 'binancecoin',
      'ADAUSDT': 'cardano',
      'SOLUSDT': 'solana',
      'XRPUSDT': 'ripple',
      'DOGEUSDT': 'dogecoin',
      'MATICUSDT': 'matic-network',
      'DOTUSDT': 'polkadot',
      'LTCUSDT': 'litecoin',
      'AVAXUSDT': 'avalanche-2',
      'LINKUSDT': 'chainlink',
      'UNIUSDT': 'uniswap',
      'ATOMUSDT': 'cosmos',
      'ETCUSDT': 'ethereum-classic'
    }
    return mapping[symbol] || 'bitcoin'
  }

  // REAL-TIME price data from Binance WebSocket
  useEffect(() => {
    let isMounted = true
    let unsubscribe: (() => void) | null = null

    const initializeChart = async () => {
      if (isMounted) setPriceLoading(true)

      try {
        // Fetch historical klines first
        const historicalData = await getHistoricalKlines(selectedSymbol, selectedTimeframe, 50)

        if (isMounted) {
          if (historicalData.length > 0) {
            setCandlestickData(historicalData)
            setCurrentPrice(historicalData[historicalData.length - 1].close)

            // ... (rest of success logic) ...

            // Convert to simple price data for the area chart
            const chartData = historicalData.map((kline) => {
              const date = new Date(kline.time)
              return {
                time: `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`,
                price: kline.close,
                volume: kline.volume
              }
            })
            setPriceData(chartData)
          } else {
            // Handle Invalid Symbol / No Data case
            setCandlestickData([])
            setWsConnected(false)
            toast({
              title: "❌ Invalid Trading Pair",
              description: `Could not load data for "${selectedSymbol}".\n\nDid you mean "${selectedSymbol}T"? (e.g. BTCUSDT instead of BTCUSD)`,
              variant: "destructive"
            })
          }
        }

        // Subscribe to real-time updates
        unsubscribe = binanceWS.subscribeKline(
          selectedSymbol,
          selectedTimeframe,
          (kline) => {
            if (!isMounted) return

            setWsConnected(true)
            setCurrentPrice(kline.close)

            // Update candlestick data
            setCandlestickData((prev) => {
              const newData = [...prev]
              const lastCandle = newData[newData.length - 1]

              if (lastCandle && lastCandle.time === kline.time) {
                // Update existing candle
                newData[newData.length - 1] = kline
              } else if (kline.isClosed || !lastCandle || kline.time > lastCandle.time) {
                // Add new candle
                newData.push(kline)
                if (newData.length > 50) newData.shift()
              }

              return newData
            })

            // Update price data for area chart
            setPriceData((prev) => {
              const date = new Date(kline.time)
              const timeStr = `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`
              const newData = [...prev]
              const lastPoint = newData[newData.length - 1]

              if (lastPoint && lastPoint.time === timeStr) {
                newData[newData.length - 1] = { time: timeStr, price: kline.close, volume: kline.volume }
              } else {
                newData.push({ time: timeStr, price: kline.close, volume: kline.volume })
                if (newData.length > 50) newData.shift()
              }

              return newData
            })
          }
        )

        console.log(`[Dashboard] Subscribed to ${selectedSymbol} ${selectedTimeframe} klines`)
      } catch (error) {
        console.error('Failed to initialize chart:', error)
        setWsConnected(false)
        setCandlestickData([]) // Clear invalid data

        toast({
          title: "❌ Invalid Trading Pair",
          description: `Could not load data for "${selectedSymbol}".\n\nDid you mean "${selectedSymbol}T"? (e.g. BTCUSDT instead of BTCUSD)`,
          variant: "destructive"
        })
      } finally {
        if (isMounted) setPriceLoading(false)
      }
    }

    initializeChart()

    return () => {
      isMounted = false
      if (unsubscribe) {
        unsubscribe()
        console.log(`[Dashboard] Unsubscribed from ${selectedSymbol} ${selectedTimeframe}`)
      }
      setWsConnected(false)
    }
  }, [selectedSymbol, selectedTimeframe])

  // Trading simulation - DOES NOT modify the price chart
  useEffect(() => {
    // Only run if strategy exists. AI runs even if paused.
    if (!strategyInstance) return

    const interval = setInterval(() => {
      // Use ref to avoid re-running effect on every price change
      const tradingPrice = currentPriceRef.current
      if (tradingPrice === 0) return

      // --- Safety Check (Circuit Breaker & Cooldown) ---
      // We check against the LIVE equity
      const safetyCheck = riskManagerRef.current.isTradingAllowed(portfolioRef.current.equity);

      if (!safetyCheck.allowed) {
        if (isRunning) { // Only stop if currently running
          console.warn(`[Safety] Force Stopping: ${safetyCheck.reason}`);
          setIsRunning(false);
          setIsSafetyLocked(true);
          setSafetyReason(safetyCheck.reason);
          toast({
            variant: "destructive",
            title: "🛡️ SAFETY SHUTDOWN TRIGGERED",
            description: safetyCheck.reason,
            duration: 10000
          });
        }
        return; // BLOCK TRADING
      }

      // Feed price data to strategy (simulating tick updates)
      if (strategyInstance) {
        // Pass High/Low as current price for tick simulation
        // In a real app, we would aggregate ticks into candles
        strategyInstance.addCandle(tradingPrice, Math.random() * 2000 + 1000, tradingPrice, tradingPrice)

        // --- AI Analysis Logic ---
        if (isAIEnabled) {
          try {
            if (typeof strategyInstance.getIndicators === 'function') {
              const indicators = strategyInstance.getIndicators();
              const analysis = MarketAnalyzer.analyze(indicators);

              // 1. Get Strategic Decision from The Brain
              const decision = orchestratorRef.current.decide(
                analysis,
                indicators,
                selectedStrategy,
                portfolioRef.current.totalPnL
              );

              // Update UI
              setMarketAnalysis({
                ...analysis,
                recommendedStrategy: decision.recommendedStrategy,
                reasoning: decision.reasoning,
                confidence: decision.confidence
              });

              setDebugIndicators({
                adx: indicators.adx,
                regime: decision.regime, // Add Regime to debug
                riskMult: decision.riskMultiplier,
                rsi: indicators.rsi14,
              });

              // STORE RISK & TP MULTIPLIERS FOR EXECUTION LOOP
              lastRiskMultiplier.current = decision.riskMultiplier;
              lastTpMultiplier.current = decision.tpMultiplier || 1.0;

              setLastError(null);

              // Auto-Switch Strategy (Throttled to 30s)
              if (Date.now() - lastSwitchTime.current > 30000 && decision.shouldSwitch) {
                console.log(`[AI] Orchestrator switching to ${decision.recommendedStrategy} (${decision.reasoning})`);

                // --- GHOST ORDER CLEANUP ---
                // Close all OPEN positions instantly at market value to avoid orphan trades on the old strategy.
                if (isRunning) {
                  setTrades(prev => prev.map(trade => {
                    if (trade.status === 'OPEN') {
                      const exitPrice = tradingPrice;
                      const rawPnl = trade.side === "BUY"
                        ? (exitPrice - trade.entryPrice) * trade.quantity
                        : (trade.entryPrice - exitPrice) * trade.quantity;

                      const currentLev = trade.leverage || 1;
                      const pnlPercent = ((exitPrice - trade.entryPrice) / trade.entryPrice) * 100 * (trade.side === "SELL" ? -1 : 1) * currentLev;

                      riskManagerRef.current.recordTrade(trade.symbol, trade.side, trade.entryPrice, exitPrice, trade.quantity);
                      const attributedStrategy = trade.strategy || selectedStrategy;
                      orchestratorRef.current.recordTradeResult(attributedStrategy, rawPnl);

                      return {
                        ...trade,
                        status: 'CLOSED',
                        exitPrice,
                        exitTime: Date.now(),
                        pnl: rawPnl,
                        pnlPercent
                      };
                    }
                    return trade;
                  }));

                  toast({
                    title: "🧹 Ghost Orders Cleared",
                    description: `All open ${selectedStrategy} positions closed prior to switch.`,
                    variant: "default",
                  });
                }

                setSelectedStrategy(decision.recommendedStrategy);
                lastSwitchTime.current = Date.now();

                toast({
                  title: "🤖 Agentic Brain Switch",
                  description: `Regime: ${decision.regime}\nStrategy: ${getStrategyName(decision.recommendedStrategy)}\nConfidence: ${(decision.confidence * 100).toFixed(0)}%`,
                  variant: "default",
                  className: "border-purple-500 bg-purple-50/90 dark:bg-purple-900/20"
                });
              }
            }
          } catch (e: any) {
            console.warn("[AI] Analysis failed temporarily", e);
            setLastError(e.toString());
          }
        }
        // -------------------------

        // Get signal from strategy - ONLY IF RUNNING
        if (isRunning) {
          let signal: StrategySignal | null = null

          if (selectedStrategy === 'MULTI_SIGNAL') {
            const signals = strategyInstance.generateSignals()
            signal = signals.combined
          } else {
            signal = strategyInstance.generateSignal()
          }

          // Count open positions
          const openPositions = trades.filter(t => t.status === "OPEN").length

          // Only create new trade if signal is BUY/SELL and within position limits
          if (signal && signal.action !== 'HOLD' && openPositions < maxPositions) {

            // DYNAMIC POSITION SIZING (The Gas Pedal)
            // If AI is enabled, use the Orchestrator's multiplier. Else 1.0.
            const multiplier = isAIEnabled ? lastRiskMultiplier.current : 1.0;

            const positionSize = riskManagerRef.current.calculateDynamicPositionSize(
              tradingPrice,
              stopLoss, // Stop Loss % from Settings
              multiplier
            );

            // Log if multiplier is active
            if (multiplier !== 1.0) {
              console.log(`[Risk] Dynamic Sizing Active: ${multiplier}x | Size: ${positionSize.toFixed(4)}`);
            }

            // Calculate SL/TP prices
            const slPrice = signal.action === 'BUY'
              ? tradingPrice * (1 - stopLoss / 100)
              : tradingPrice * (1 + stopLoss / 100)

            // DYNAMIC TAKE PROFIT (Volatility Scaling)
            const activeTpMultiplier = isAIEnabled ? lastTpMultiplier.current : 1.0;
            const dynamicTakeProfit = takeProfit * activeTpMultiplier;

            if (activeTpMultiplier !== 1.0) {
              console.log(`[Risk] Volatility Scaling Active: TP widened by ${activeTpMultiplier}x to ${dynamicTakeProfit}%`);
            }

            const tpPrice = signal.action === 'BUY'
              ? tradingPrice * (1 + dynamicTakeProfit / 100)
              : tradingPrice * (1 - dynamicTakeProfit / 100)

            // Limit-Chase: Start by placing order slightly better than current price
            const limitBuffer = 0.0005; // 0.05%
            const initialLimitPrice = signal.action === 'BUY'
              ? tradingPrice * (1 - limitBuffer)
              : tradingPrice * (1 + limitBuffer);

            const newTrade: TradeEntry = {
              id: `trade-${Date.now()}`,
              symbol: selectedSymbol,
              side: signal.action,
              entryPrice: initialLimitPrice, // Placeholder until filled
              limitPrice: initialLimitPrice,
              signalPrice: tradingPrice,
              quantity: positionSize,
              entryTime: Date.now(),
              status: "PENDING", // Wait for fill
              strategy: selectedStrategy,
              leverage: leverage,
              stopLossPrice: slPrice,
              takeProfitPrice: tpPrice,
              highestPrice: initialLimitPrice,
              lowestPrice: initialLimitPrice
            }
            setTrades((prev) => [newTrade, ...prev].slice(0, 50))

            toast({
              title: `⏳ ${signal.action} Limit Order Placed`,
              description: `Signal: $${tradingPrice.toFixed(2)} | Limit: $${initialLimitPrice.toFixed(2)}`,
              duration: 3000
            })
          }
        }
        // Simulate Limit Order Processing (Chase & Fill)
        setTrades((prev) =>
          prev.map((trade) => {
            if (trade.status !== "PENDING" || trade.symbol !== selectedSymbol) return trade;

            const timeActive = Date.now() - trade.entryTime;
            let newLimitPrice = trade.limitPrice || trade.entryPrice;

            // 1. Check if Filled
            if (trade.side === "BUY" && tradingPrice <= newLimitPrice) {
              toast({ title: "✅ Buy Limit Filled", description: `Filled at $${newLimitPrice.toFixed(2)}` });
              return { ...trade, status: "OPEN", entryPrice: newLimitPrice, entryTime: Date.now(), highestPrice: newLimitPrice, lowestPrice: newLimitPrice };
            } else if (trade.side === "SELL" && tradingPrice >= newLimitPrice) {
              toast({ title: "✅ Sell Limit Filled", description: `Filled at $${newLimitPrice.toFixed(2)}` });
              return { ...trade, status: "OPEN", entryPrice: newLimitPrice, entryTime: Date.now(), highestPrice: newLimitPrice, lowestPrice: newLimitPrice };
            }

            // 2. Check if needs Chase (after 30s)
            if (timeActive > 30000) {
              const fairValueSpread = 0.005; // 0.5% max chase from original signal
              const signalPrice = trade.signalPrice || trade.entryPrice;

              // 3. Check Fair Value cutoff
              if (trade.side === "BUY" && tradingPrice > signalPrice * (1 + fairValueSpread)) {
                toast({ title: "❌ Buy Cancelled", description: "Price exceeded Fair Value. Aborting chase.", variant: "destructive" });
                return { ...trade, status: "CANCELLED" };
              } else if (trade.side === "SELL" && tradingPrice < signalPrice * (1 - fairValueSpread)) {
                toast({ title: "❌ Sell Cancelled", description: "Price dropped below Fair Value. Aborting chase.", variant: "destructive" });
                return { ...trade, status: "CANCELLED" };
              }

              // 4. Chase the price (simulate bumping order up/down)
              newLimitPrice = trade.side === "BUY" ? tradingPrice * 0.9999 : tradingPrice * 1.0001;
              toast({ title: "🏃 Order Chasing", description: `Moving ${trade.side} limit to $${newLimitPrice.toFixed(2)}` });

              return { ...trade, limitPrice: newLimitPrice, entryTime: Date.now() }; // Reset timer
            }

            return trade;
          })
        );

        // Simulate closing trades and PnL updates
        setTrades((prev) =>
          prev.map((trade) => {
            // Only process active trades for the current symbol
            if (trade.status !== "OPEN" || trade.symbol !== selectedSymbol) return trade

            const currentLev = trade.leverage || 1
            let exitReason: string | null = null
            let exitPrice = 0

            // Update trailing high/low
            const newHighest = Math.max(trade.highestPrice || 0, tradingPrice)
            const newLowest = Math.min(trade.lowestPrice || Infinity, tradingPrice)

            if (trade.side === "BUY") {
              // Check Stop Loss (Urgent Market Exit)
              if (tradingPrice <= (trade.stopLossPrice || 0)) {
                exitReason = "🛑 Urgent Exit (Stop Loss)"
                // FOK Simulation: We take whatever liquidity is at the current price. We do NOT get the perfect stop price if we gapped down.
                // Apply a tiny 0.05% slippage penalty to simulate eating the order book to guarantee the fill.
                exitPrice = tradingPrice * 0.9995;
              }
              // Check Take Profit (Limit Order Simulation)
              else if (tradingPrice >= (trade.takeProfitPrice || Infinity)) {
                exitReason = "Take Profit 🎯"
                // For TP, we assume it rested on the book and filled exactly at the requested price, or better.
                exitPrice = Math.max(trade.takeProfitPrice || tradingPrice, tradingPrice);
              }
              // Check Trailing Stop (Urgent Market Exit)
              else if (trailingStop > 0 && tradingPrice <= newHighest * (1 - trailingStop / 100)) {
                exitReason = "📉 Urgent Exit (Trailing Stop)"
                exitPrice = tradingPrice * 0.9995;
              }
            } else { // SELL
              // Check Stop Loss (Urgent Market Exit)
              if (tradingPrice >= (trade.stopLossPrice || Infinity)) {
                exitReason = "🛑 Urgent Exit (Stop Loss)"
                // FOK Simulation: Pay slightly more (worse price) to guarantee exit
                exitPrice = tradingPrice * 1.0005;
              }
              // Check Take Profit (Limit Order Simulation)
              else if (tradingPrice <= (trade.takeProfitPrice || 0)) {
                exitReason = "Take Profit 🎯"
                exitPrice = Math.min(trade.takeProfitPrice || tradingPrice, tradingPrice);
              }
              // Check Trailing Stop (Urgent Market Exit)
              else if (trailingStop > 0 && tradingPrice >= newLowest * (1 + trailingStop / 100)) {
                exitReason = "📈 Urgent Exit (Trailing Stop)"
                exitPrice = tradingPrice * 1.0005;
              }
            }

            // If triggered, close trade
            if (exitReason) {
              const rawPnl = trade.side === "BUY"
                ? (exitPrice - trade.entryPrice) * trade.quantity
                : (trade.entryPrice - exitPrice) * trade.quantity

              // Apply Leverage
              const pnl = rawPnl

              const pnlPercent = ((exitPrice - trade.entryPrice) / trade.entryPrice) * 100 * (trade.side === "SELL" ? -1 : 1) * currentLev

              // Notify Risk Manager
              riskManagerRef.current.recordTrade(trade.symbol, trade.side, trade.entryPrice, exitPrice, trade.quantity);

              // Notify Orchestrator (Feedback Loop)
              if (isAIEnabled) {
                const attributedStrategy = trade.strategy || selectedStrategy;
                orchestratorRef.current.recordTradeResult(attributedStrategy, pnl);
              }

              toast({
                title: `${exitReason} Triggered`,
                description: `Closed at $${exitPrice.toFixed(2)} | PnL: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`,
              })

              return {
                ...trade,
                exitPrice,
                exitTime: Date.now(),
                status: "CLOSED",
                pnl,
                pnlPercent,
                highestPrice: newHighest, // Update final state
                lowestPrice: newLowest
              }
            }

            // Return with updated trailing values if not closed
            return {
              ...trade,
              highestPrice: newHighest,
              lowestPrice: newLowest
            }
          })
        )

        // Update portfolio metrics
        const closedTrades = trades.filter((t) => t.status === "CLOSED")
        const totalPnL = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0)
        const wins = closedTrades.filter((t) => (t.pnl || 0) > 0).length
        const winRate = closedTrades.length > 0 ? wins / closedTrades.length : 0

        setPortfolio((prev) => {
          // Recalculate Unrealized PnL
          const unrealizedPnL = trades
            .filter(t => t.status === "OPEN" && t.symbol === selectedSymbol)
            .reduce((sum, t) => {
              const priceDiff = t.side === "BUY" ? tradingPrice - t.entryPrice : t.entryPrice - tradingPrice
              return sum + (priceDiff * t.quantity)
            }, 0)

          return {
            ...prev,
            equity: prev.totalBalance + totalPnL + unrealizedPnL,
            realizedPnL: totalPnL,
            unrealizedPnL,
            totalPnL,
            winRate,
          }
        })

        setBalanceHistory((prev) => [
          ...prev,
          {
            time: new Date().toLocaleTimeString(),
            balance: portfolioRef.current.equity, // Use current equity from Ref
          },
        ].slice(-100))
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, strategyInstance, isAIEnabled, selectedStrategy])

  const openTrades = trades.filter((t) => t.status === "OPEN")
  const closedTrades = trades.filter((t) => t.status === "CLOSED")
  const totalPnL = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0)

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Binance Trading Bot</h1>
            <p className="text-muted-foreground mt-1">Advanced multi-strategy trading engine</p>
          </div>
          <div className="flex gap-4 items-center">
            {/* AI Toggle */}
            <div className="flex items-center space-x-2 bg-card border px-3 py-2 rounded-lg">
              <Switch
                id="ai-mode"
                checked={isAIEnabled}
                onCheckedChange={(checked) => {
                  setIsAIEnabled(checked);
                  if (checked) toast({ title: "🧠 AI Smart Mode Activated", description: "The bot will now automatically switch strategies based on market conditions." });
                }}
              />
              <Label htmlFor="ai-mode" className="cursor-pointer font-semibold flex items-center gap-1">
                {isAIEnabled ? "🧠 AI Active" : "🧠 AI Off"}
              </Label>
            </div>

            <Button
              variant="outline"
              className="gap-2 border-purple-500/50 hover:bg-purple-500/10 text-purple-400"
              onClick={handleAskGemini}
              disabled={isLoadingAdvice}
            >
              {isLoadingAdvice ? <div className="animate-spin">⏳</div> : <Sparkles className="h-4 w-4" />}
              Ask AI Advisor
            </Button>

            <Button
              variant="outline"
              className="gap-2 border-blue-500/50 hover:bg-blue-500/10 text-blue-400"
              onClick={handleAutoTune}
              disabled={isTuning}
            >
              {isTuning ? <div className="animate-spin">⏳</div> : <Settings className="h-4 w-4" />}
              Auto-Tune (WFA)
            </Button>

            <Button
              onClick={() => setIsRunning(!isRunning)}
              disabled={isSafetyLocked} // Disable start if locked
              className={isRunning ? "bg-destructive hover:bg-destructive/90" : "bg-primary"}
              size="lg"
            >
              {isRunning ? (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Stop Bot
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Start Bot
                </>
              )}
            </Button>

            {/* Safety Controls */}
            {isSafetyLocked ? (
              <Button
                variant="outline"
                className="border-yellow-500 text-yellow-500 hover:bg-yellow-500/10"
                onClick={() => {
                  riskManagerRef.current.resetSafety();
                  setIsSafetyLocked(false);
                  setSafetyReason(null);
                  toast({ title: "Safety Override", description: "System safety locks have been reset manually." });
                }}
              >
                <Unlock className="w-4 h-4 mr-2" />
                Reset Safety
              </Button>
            ) : (
              <Button
                variant="destructive"
                className="bg-red-900/80 hover:bg-red-900 border border-red-700"
                onClick={() => {
                  riskManagerRef.current.triggerKillSwitch();
                  setIsSafetyLocked(true);
                  setSafetyReason("Manual Kill Switch Triggered");
                  setIsRunning(false);

                  // LIQUIDATE ALL POSITIONS IMMEDIATELY
                  const exitPrice = currentPrice;
                  let totalLiquidationPnL = 0;

                  setTrades(prevTrades => prevTrades.map(t => {
                    if (t.status === 'OPEN') {
                      const pnl = t.side === "BUY"
                        ? (exitPrice - t.entryPrice) * t.quantity
                        : (t.entryPrice - exitPrice) * t.quantity;

                      totalLiquidationPnL += pnl;

                      return {
                        ...t,
                        status: 'CLOSED',
                        exitPrice,
                        exitTime: Date.now(),
                        pnl,
                        exitReason: "KILL SWITCH 💀"
                      };
                    }
                    return t;
                  }));

                  // Update Portfolio immediately since loop stops
                  setPortfolio(prev => ({
                    ...prev,
                    equity: prev.equity + totalLiquidationPnL,
                    totalPnL: prev.totalPnL + totalLiquidationPnL,
                    realizedPnL: prev.realizedPnL + totalLiquidationPnL,
                    unrealizedPnL: 0
                  }));

                  toast({
                    title: "🟥 KILL SWITCH ACTIVATED",
                    description: `Trading halted. All positions liquidated at $${exitPrice}.`,
                    duration: 5000
                  });
                }}
              >
                <Skull className="w-4 h-4 mr-2" />
                KILL SWITCH
              </Button>
            )}
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Account Balance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">${portfolio.equity.toFixed(2)}</div>
              <p className={`text-xs mt-1 ${portfolio.totalPnL > 0 ? "text-green-500" : "text-red-500"}`}>
                {portfolio.totalPnL > 0 ? "+" : ""}
                ${portfolio.totalPnL.toFixed(2)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{(portfolio.winRate * 100).toFixed(1)}%</div>
              <p className="text-xs mt-1 text-muted-foreground">{closedTrades.length} trades closed</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Open Positions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{openTrades.length}</div>
              <p className="text-xs mt-1 text-muted-foreground">
                Unrealized: ${portfolio.unrealizedPnL.toFixed(2)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Max Drawdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{(portfolio.maxDrawdown * 100).toFixed(2)}%</div>
              <p className="text-xs mt-1 text-muted-foreground">Risk metric</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="flex items-center">
              <Activity className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="trades" className="flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Trades
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center">
              <BarChart className="w-4 h-4 mr-2" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              Settings
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Price Chart - REAL-TIME from Binance */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      Price Action - {selectedSymbol}
                      {wsConnected ? (
                        <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30 text-xs">
                          <span className="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse" />
                          LIVE
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/30 text-xs">
                          Connecting...
                        </Badge>
                      )}
                    </span>
                    {currentPrice > 0 && (
                      <span className="text-lg font-normal text-primary">
                        ${currentPrice < 1 ? currentPrice.toFixed(4) : currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                    )}
                  </CardTitle>
                  <CardDescription>{selectedTimeframe} timeframe • Binance Real-Time Data</CardDescription>
                </CardHeader>
                <CardContent>
                  {priceLoading ? (
                    <div className="h-[400px] flex items-center justify-center">
                      <div className="text-muted-foreground">Loading price data...</div>
                    </div>
                  ) : candlestickData.length === 0 ? (
                    <div className="h-[400px] flex items-center justify-center">
                      <div className="text-muted-foreground">No price data available</div>
                    </div>
                  ) : (
                    <div className="w-full h-[400px] relative group">
                      <FinancialChart data={candlestickData} />
                      {/* Strategy Overlay Badge */}
                      <div className="absolute top-4 left-4 z-10 bg-background/90 backdrop-blur-md border border-border/50 rounded-lg px-3 py-2 shadow-lg flex items-center gap-3 transition-opacity">
                        <div className={`w-2.5 h-2.5 rounded-full shadow-sm ${isRunning ? "bg-emerald-500 animate-pulse" : "bg-yellow-500"}`} />
                        <div className="flex flex-col">
                          <span className="text-xs font-bold text-foreground tracking-tight leading-none">
                            {getStrategyName(selectedStrategy)}
                          </span>
                          <span className="text-[10px] font-medium text-muted-foreground leading-none mt-1 uppercase tracking-wider">
                            {isRunning ? "Running" : "Paused"}
                          </span>
                        </div>
                      </div>

                      {/* AI Analysis Overlay */}
                      {isAIEnabled && marketAnalysis && (
                        <div className="absolute top-4 right-4 z-10 bg-purple-950/80 backdrop-blur-md border border-purple-500/30 rounded-lg px-3 py-2 shadow-lg max-w-[220px]">
                          <div className="text-[10px] uppercase text-purple-300 font-bold mb-1 flex justify-between items-center">
                            <span>{debugIndicators?.regime?.replace('_', ' ') || 'ANALYZING'}</span>
                            <Badge variant="outline" className="text-[9px] h-4 px-1 border-purple-400 text-purple-200">
                              {marketAnalysis.confidence * 100}% Conf
                            </Badge>
                          </div>
                          <div className="text-xs font-medium text-white mb-1">
                            {getStrategyName(marketAnalysis.recommendedStrategy || selectedStrategy)}
                          </div>
                          <div className="text-[10px] text-gray-300 leading-tight">
                            {marketAnalysis.reasoning}
                          </div>
                          {debugIndicators?.riskMult !== 1 && (
                            <div className="mt-1 pt-1 border-t border-purple-800 text-[9px] text-yellow-300 flex justify-between">
                              <span>Dynamic Size:</span>
                              <span>{debugIndicators?.riskMult}x</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Equity Curve */}
              <Card>
                <CardHeader>
                  <CardTitle>Equity Curve</CardTitle>
                  <CardDescription>Account balance over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={balanceHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="time" stroke="#000000" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#000000" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Line
                        type="monotone"
                        dataKey="balance"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Active Trades */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="w-4 h-4 mr-2 text-yellow-500" />
                  Active Positions ({openTrades.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                {openTrades.length === 0 ? (
                  <p className="text-muted-foreground text-center py-8">No open positions</p>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {openTrades.map((trade) => (
                      <div key={trade.id} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                        <div className="flex items-center gap-3">
                          <Badge variant={trade.side === "BUY" ? "default" : "destructive"}>
                            {trade.side}
                          </Badge>
                          <div>
                            <p className="font-medium">{trade.symbol}</p>
                            <p className="text-sm text-muted-foreground">${trade.entryPrice.toFixed(2)}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-medium">{trade.quantity.toFixed(4)}</p>
                          <Badge variant="outline" className="text-yellow-600">
                            <Clock className="w-3 h-3 mr-1" />
                            Open
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Trades Tab */}
          <TabsContent value="trades" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                  Closed Trades ({closedTrades.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-2 text-left">Symbol</th>
                        <th className="px-4 py-2 text-left">Side</th>
                        <th className="px-4 py-2 text-left">Entry</th>
                        <th className="px-4 py-2 text-left">Exit</th>
                        <th className="px-4 py-2 text-left">Qty</th>
                        <th className="px-4 py-2 text-right">PnL</th>
                        <th className="px-4 py-2 text-right">%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {closedTrades.slice(0, 20).map((trade) => (
                        <tr key={trade.id} className="border-t hover:bg-muted">
                          <td className="px-4 py-2">{trade.symbol}</td>
                          <td className="px-4 py-2">
                            <Badge variant={trade.side === "BUY" ? "default" : "destructive"}>
                              {trade.side}
                            </Badge>
                          </td>
                          <td className="px-4 py-2">${trade.entryPrice.toFixed(2)}</td>
                          <td className="px-4 py-2">${(trade.exitPrice || 0).toFixed(2)}</td>
                          <td className="px-4 py-2">{trade.quantity.toFixed(4)}</td>
                          <td className={`px-4 py-2 text-right ${(trade.pnl || 0) > 0 ? "text-green-500" : "text-red-500"}`}>
                            ${(trade.pnl || 0).toFixed(2)}
                          </td>
                          <td className={`px-4 py-2 text-right ${(trade.pnlPercent || 0) > 0 ? "text-green-500" : "text-red-500"}`}>
                            {(trade.pnlPercent || 0).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Trade Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={[
                        { name: "Wins", value: closedTrades.filter((t) => (t.pnl || 0) > 0).length },
                        { name: "Losses", value: closedTrades.filter((t) => (t.pnl || 0) < 0).length },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="name" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>PnL Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={closedTrades
                        .slice(0, 10)
                        .map((t, i) => ({
                          name: `Trade ${i + 1}`,
                          pnl: t.pnl || 0,
                        }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="name" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip contentStyle={{ backgroundColor: "var(--card)", border: "1px solid var(--border)" }} />
                      <Bar
                        dataKey="pnl"
                        fill="#3b82f6"
                        style={{ background: "#3b82f6" }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Trading Configuration</CardTitle>
                <CardDescription>Configure your trading strategy and parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Strategy Selection */}
                <div>
                  <label className="text-sm font-medium mb-2 block">Trading Strategy</label>
                  <select
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md bg-background"
                  >
                    <option value="MA_CROSSOVER">Moving Average Crossover - SMA 20/50/200 (Golden/Death Cross)</option>
                    <option value="MACD">MACD Strategy - Momentum-based crossovers</option>
                    <option value="MEAN_REVERSION">Mean Reversion - Bollinger Bands + RSI (Oversold/Overbought)</option>
                    <option value="GRID">Grid Trading - Fixed intervals for oscillations</option>
                    <option value="MULTI_SIGNAL">Multi-Signal Ensemble - Voting system combining all strategies</option>
                    <option disabled>--- New Strategies ---</option>
                    <option value="RSI_DIVERGENCE">RSI Divergence - High Win-Rate Reversals</option>
                    <option value="BOLLINGER_BREAKOUT">Bollinger Breakout - Explosive Volatility Moves</option>
                    <option value="VWAP_TREND">VWAP Trend - Institutional Trend Following</option>
                    <option value="ICHIMOKU">Ichimoku Cloud - Japanese Trend Reliability</option>
                    <option value="PIVOT_REVERSAL">Pivot Reversal - Daily Support/Resistance</option>
                  </select>
                  <p className="text-xs text-muted-foreground mt-1">
                    {selectedStrategy === "MA_CROSSOVER" && "Best for: Trending markets with clear direction"}
                    {selectedStrategy === "MACD" && "Best for: Momentum trading and trend changes"}
                    {selectedStrategy === "MEAN_REVERSION" && "Best for: Range-bound and ranging markets"}
                    {selectedStrategy === "GRID" && "Best for: Volatile and range-bound markets"}
                    {selectedStrategy === "MULTI_SIGNAL" && "Best for: All market conditions (recommended)"}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {/* Trading Pair with Search */}
                  <div className="relative">
                    <label className="text-sm font-medium mb-2 block">Trading Pair</label>
                    <input
                      type="text"
                      value={searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value.toUpperCase())
                        setShowPairSearch(true)
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          setSelectedSymbol(searchTerm)
                          setShowPairSearch(false)
                        }
                      }}
                      onFocus={() => setShowPairSearch(true)}
                      onBlur={() => setTimeout(() => setShowPairSearch(false), 200)}
                      placeholder="Search pairs (e.g., BTCUSDT)"
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    />
                    {showPairSearch && filteredPairs.length > 0 && (
                      <div className="absolute z-10 w-full mt-1 bg-background border rounded-md shadow-lg max-h-60 overflow-y-auto">
                        {filteredPairs.map((pair) => (
                          <button
                            key={pair}
                            onClick={() => {
                              setSelectedSymbol(pair)
                              setSearchTerm(pair)
                              setShowPairSearch(false)
                            }}
                            className="w-full px-3 py-2 text-left hover:bg-muted transition-colors"
                          >
                            {pair}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Timeframe */}
                  <div>
                    <label className="text-sm font-medium mb-2 block">Timeframe</label>
                    <select
                      value={selectedTimeframe}
                      onChange={(e) => setSelectedTimeframe(e.target.value)}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    >
                      <option value="1m">1 Minute</option>
                      <option value="5m">5 Minutes</option>
                      <option value="15m">15 Minutes</option>
                      <option value="1h">1 Hour</option>
                      <option value="4h">4 Hours</option>
                      <option value="1d">1 Day</option>
                    </select>
                  </div>
                </div>

                {/* Risk Settings */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Risk Per Trade (%)</label>
                    <input
                      type="number"
                      min="0.5"
                      max="10"
                      step="0.5"
                      value={riskPerTrade}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        setRiskPerTrade(isNaN(val) ? 0 : val)
                      }}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Recommended: 1-3%</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">Max Positions</label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={maxPositions}
                      onChange={(e) => {
                        const val = parseInt(e.target.value)
                        setMaxPositions(isNaN(val) ? 1 : val)
                      }}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    />
                  </div>
                </div>

                {/* Advanced Risk Management */}
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-semibold mb-3">Advanced Risk Management</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Stop Loss (%)</label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={stopLoss}
                        onChange={(e) => setStopLoss(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">Take Profit (%)</label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={takeProfit}
                        onChange={(e) => setTakeProfit(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">Trailing Stop (%)</label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={trailingStop}
                        onChange={(e) => setTrailingStop(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Locks in profit as price rises</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">Simulated Leverage (x)</label>
                      <select
                        value={leverage}
                        onChange={(e) => setLeverage(parseInt(e.target.value))}
                        className="w-full px-3 py-2 border rounded-md bg-background"
                      >
                        <option value="1">1x (Spot)</option>
                        <option value="2">2x</option>
                        <option value="5">5x</option>
                        <option value="10">10x (High Risk)</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Current Configuration Summary */}
                <div className="pt-4 border-t bg-muted/30 rounded-lg p-4">
                  <h4 className="text-sm font-semibold mb-2">Current Configuration</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">Strategy:</span>
                      <span className="ml-2 font-medium">
                        {getStrategyName(selectedStrategy)}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Pair:</span>
                      <span className="ml-2 font-medium">{selectedSymbol}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Timeframe:</span>
                      <span className="ml-2 font-medium">{selectedTimeframe}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Risk:</span>
                      <span className="ml-2 font-medium">{riskPerTrade}%</span>
                    </div>
                  </div>
                </div>

                {/* Configuration Presets */}
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-semibold">Configuration Presets</h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowPresets(!showPresets)}
                    >
                      {showPresets ? "Hide" : "Show"} Presets ({Object.keys(savedPresets).length})
                    </Button>
                  </div>

                  {showPresets && (
                    <div className="space-y-4">
                      {/* Save as Preset */}
                      <div className="flex gap-2">
                        <input
                          type="text"
                          placeholder="Enter preset name..."
                          value={presetName}
                          onChange={(e) => setPresetName(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSaveAsPreset()}
                          className="flex-1 px-3 py-2 text-sm border rounded-md bg-background"
                        />
                        <Button size="sm" onClick={handleSaveAsPreset}>
                          Save as Preset
                        </Button>
                      </div>

                      {/* Saved Presets List */}
                      {Object.keys(savedPresets).length > 0 ? (
                        <div className="space-y-2">
                          <p className="text-xs text-muted-foreground">Saved Presets:</p>
                          {Object.entries(savedPresets).map(([name, config]) => (
                            <div
                              key={name}
                              className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
                            >
                              <div className="flex-1">
                                <p className="font-medium text-sm">{name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {config.symbol} • {config.timeframe} • {config.strategy.replace('_', ' ')} • {config.riskPerTrade}%
                                </p>
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleLoadPreset(name)}
                                >
                                  Load
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleDeletePreset(name)}
                                  className="text-red-600 hover:text-red-700"
                                >
                                  Delete
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground text-center py-4">
                          No saved presets yet. Create one above!
                        </p>
                      )}
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t flex gap-2">
                  <Button variant="outline" className="flex-1 bg-transparent">
                    Reset to Defaults
                  </Button>
                  <Button className="flex-1" onClick={handleSaveConfiguration}>Save Configuration</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      <Dialog open={isAdviceOpen} onOpenChange={setIsAdviceOpen}>
        <DialogContent className="sm:max-w-[500px] bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-purple-400">
              <Brain className="h-5 w-5" />
              Gemini Strategic Advisor
            </DialogTitle>
            <DialogDescription>
              Analysis of current market structure based on technical indicators.
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4 p-4 bg-slate-950 rounded-lg border border-slate-800 text-sm whitespace-pre-wrap leading-relaxed text-slate-300">
            {geminiAdvice || "Analyzing market data..."}
          </div>
        </DialogContent>
      </Dialog>


    </div>
  )

}
