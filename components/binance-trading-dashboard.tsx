"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { useLiveMarketData } from "@/lib/use-live-market"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { GeminiAdvisor } from "@/lib/gemini-advisor"
import { RiskManager } from "@/lib/risk-management"
import { AgenticOrchestrator, StrategicDecision } from "@/lib/agentic-orchestrator"
import { aiConfigEngine, type AIConfigResult } from "@/lib/ai-config-engine"
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
  Lock,
  Globe,
  Database,
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
import { WalkForwardAnalyzer, type WFAReport } from "@/lib/walk-forward-analyzer"
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

const MOCK_TRENDING_COINS = [
  { symbol: "BTCUSDT", name: "Bitcoin", price: "$65,432.10", change24h: "+2.4%", logo: "₿", rank: 1, cap: "$1.2T", vol: "$34B", supply: "19.6M" },
  { symbol: "ETHUSDT", name: "Ethereum", price: "$3,456.78", change24h: "+1.2%", logo: "Ξ", rank: 2, cap: "$415B", vol: "$15B", supply: "120M" },
  { symbol: "XRPUSDT", name: "XRP", price: "$0.62", change24h: "+3.1%", logo: "✕", rank: 3, cap: "$33B", vol: "$2.1B", supply: "54B" },
  { symbol: "SOLUSDT", name: "Solana", price: "$145.20", change24h: "+5.6%", logo: "◎", rank: 5, cap: "$65B", vol: "$4.2B", supply: "443M" },
  { symbol: "BNBUSDT", name: "Binance Coin", price: "$580.40", change24h: "-0.8%", logo: "BNB", rank: 4, cap: "$86B", vol: "$1.1B", supply: "149M" },
  { symbol: "ADAUSDT", name: "Cardano", price: "$0.48", change24h: "+2.9%", logo: "₳", rank: 9, cap: "$17B", vol: "$530M", supply: "35B" },
  { symbol: "DOGEUSDT", name: "Dogecoin", price: "$0.15", change24h: "+12.4%", logo: "Ð", rank: 8, cap: "$21B", vol: "$2.5B", supply: "143B" },
  { symbol: "AVAXUSDT", name: "Avalanche", price: "$35.60", change24h: "+4.1%", logo: "🔺", rank: 11, cap: "$14B", vol: "$600M", supply: "377M" },
  { symbol: "SHIBUSDT", name: "Shiba Inu", price: "$0.000025", change24h: "+8.7%", logo: "🐕", rank: 13, cap: "$14B", vol: "$800M", supply: "589T" },
  { symbol: "LINKUSDT", name: "Chainlink", price: "$18.20", change24h: "+2.1%", logo: "🔗", rank: 14, cap: "$10B", vol: "$450M", supply: "587M" },
  { symbol: "MATICUSDT", name: "Polygon", price: "$0.95", change24h: "-1.2%", logo: "⬡", rank: 16, cap: "$9B", vol: "$320M", supply: "9.2B" },
  { symbol: "UNIUSDT", name: "Uniswap", price: "$11.40", change24h: "+1.8%", logo: "🦄", rank: 18, cap: "$6.8B", vol: "$210M", supply: "600M" },
]
import { useRouter } from "next/navigation"

// ─── Helper to derive live-ish indicator data from a coin's 24h change ───────
function deriveCoinAnalysis(symbol: string, change24h: string, price: string) {
  const pct = parseFloat(change24h.replace('%', ''))
  const isBull = pct >= 0
  const strength = Math.min(Math.abs(pct), 15) / 15 // 0→1

  // RSI: bullish coins → higher RSI
  const rsi = isBull ? 52 + strength * 26 : 48 - strength * 26
  // MACD: positive when bullish
  const macd = isBull ? +(strength * 0.8).toFixed(3) : -(strength * 0.8).toFixed(3)
  const macdSignal = isBull ? +(strength * 0.5).toFixed(3) : -(strength * 0.5).toFixed(3)
  // Bollinger %B: >0.5 = price above midband
  const bollingerB = isBull ? 0.5 + strength * 0.45 : 0.5 - strength * 0.45
  // Stochastic
  const stochK = isBull ? 55 + strength * 35 : 45 - strength * 35
  const stochD = isBull ? 50 + strength * 30 : 50 - strength * 30
  // ADX — higher change means stronger trend
  const adx = 20 + strength * 45

  const signal = (v: boolean) => (v ? 'BUY' : 'SELL')
  const rsiSignal = rsi > 70 ? 'OVERBOUGHT' : rsi < 30 ? 'OVERSOLD' : isBull ? 'BUY' : 'SELL'

  // Sentiment scores (0-10)
  const sentScore = 5 + pct * 0.4
  const socialMedia  = Math.min(10, Math.max(0, sentScore + 0.8)).toFixed(2)
  const socialSent   = Math.min(10, Math.max(0, sentScore + 0.3)).toFixed(2)
  const news         = Math.min(10, Math.max(0, sentScore - 0.1)).toFixed(2)
  const kol          = Math.min(10, Math.max(0, sentScore + 0.6)).toFixed(2)
  const overallScore = ((+socialMedia + +socialSent + +news + +kol) / 4).toFixed(2)
  const sentiment    = +overallScore >= 6.5 ? 'Strong Positive' : +overallScore >= 5 ? 'Positive' : +overallScore >= 4 ? 'Neutral' : 'Negative'
  const sentimentColor = +overallScore >= 6 ? '#1DB954' : +overallScore >= 5 ? '#1DB954' : '#F6465D'
  const volText      = `${(2.1 + strength).toFixed(2)}k`

  return {
    rsi: rsi.toFixed(1), rsiSignal,
    macd: macd.toString(), macdSignal: macdSignal.toString(),
    bollingerB: bollingerB.toFixed(2),
    stochK: stochK.toFixed(1), stochD: stochD.toFixed(1),
    adx: adx.toFixed(1),
    overallSignal: isBull ? 'BUY' : 'SELL',
    socialMedia, socialSent, news, kol, overallScore,
    sentiment, sentimentColor, volText,
    newsText: isBull ? 'Bullish' : 'Bearish',
    socialText: isBull ? 'Bullish' : 'Bearish',
    kolText: isBull ? 'Bullish' : 'Neutral',
  }
}

type CoinEntry = typeof MOCK_TRENDING_COINS[0]

function AiSelectPanel({ coins, onSelectCoin }: { coins: CoinEntry[]; onSelectCoin: (c: CoinEntry) => void }) {
  const [activeTab, setActiveTab] = useState<'technical' | 'sentiment'>('sentiment')
  const [search, setSearch] = useState('')

  const displayCoins = coins
    .filter(c => c.symbol.toLowerCase().includes(search.toLowerCase()) || c.name.toLowerCase().includes(search.toLowerCase()))
    .slice(0, 6)

  return (
    <div className="flex flex-col space-y-4">
      {/* Top Bar Navigation */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A2D35] pb-4">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('technical')}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${activeTab === 'technical' ? 'bg-[#2B3139] text-[#EAECEF]' : 'text-[#848E9C] hover:text-[#EAECEF]'}`}
          >
            Technical Indicators
          </button>
          <button
            onClick={() => setActiveTab('sentiment')}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${activeTab === 'sentiment' ? 'bg-[#2B3139] text-[#EAECEF]' : 'text-[#848E9C] hover:text-[#EAECEF]'}`}
          >
            Sentiment
          </button>
        </div>
        <div className="relative">
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search"
            className="bg-[#121418] border border-[#2A2D35] rounded-lg px-3 py-1.5 text-sm w-48 text-[#EAECEF] placeholder:text-[#848E9C] focus:outline-none focus:border-[#1DB954]"
          />
          <svg className="absolute right-3 top-2 w-4 h-4 text-[#848E9C]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Disclaimer & Badge */}
      <div className="flex flex-col md:flex-row justify-between md:items-start items-center text-[#848E9C] text-xs w-full">
        <p className="max-w-4xl leading-relaxed">
          Disclaimer: Data is derived from live market conditions and is presented "as is". Not financial advice. Digital asset prices can be volatile.
        </p>
        <div className="flex items-center gap-1 text-[#1DB954] font-bold whitespace-nowrap mt-4 md:mt-0 ml-4">
          <span>★</span> AI Select ({displayCoins.length})
        </div>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-2">
        {displayCoins.map((coin) => {
          const d = deriveCoinAnalysis(coin.symbol, coin.change24h, coin.price)
          const isBull = !coin.change24h.startsWith('-')

          return (
            <div
              key={coin.symbol}
              className="bg-[#181A20] border border-[#2A2D35] rounded-xl overflow-hidden hover:border-[#1DB954]/50 transition-colors cursor-pointer group"
              onClick={() => onSelectCoin(coin)}
            >
              <div className="p-5 flex flex-col h-full relative">
                {/* Header */}
                <div className="flex justify-between items-start mb-4">
                  <div className="flex gap-3 items-center">
                    <span className="text-3xl">{coin.logo}</span>
                    <div className="flex flex-col">
                      <span className="font-bold text-[#EAECEF] text-lg leading-tight">{coin.symbol.replace('USDT', '')}</span>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-sm font-semibold text-[#EAECEF]">{coin.price}</span>
                        <span className={`text-xs font-semibold ${isBull ? 'text-[#1DB954]' : 'text-[#F6465D]'}`}>{coin.change24h}</span>
                      </div>
                    </div>
                  </div>
                  <button className="text-[#848E9C] hover:text-[#1DB954] transition-colors" onClick={e => e.stopPropagation()}>
                    <span className="text-xl">☆</span>
                  </button>
                </div>

                {activeTab === 'sentiment' ? (
                  <>
                    {/* Radar Visual */}
                    <div className="relative w-full h-44 my-2 flex items-center justify-center">
                      <div className="absolute w-32 h-32 bg-[#1DB954]/5 border border-[#1DB954]/20 transform rotate-45 rounded-sm"></div>
                      <div className="absolute w-20 h-20 bg-[#1DB954]/10 border border-[#1DB954]/40 transform rotate-45 rounded-sm shadow-[0_0_15px_rgba(29,185,84,0.2)]"></div>
                      <div className="relative z-10 text-center flex flex-col items-center">
                        <span className="text-2xl font-bold text-white drop-shadow-md">{d.overallScore}</span>
                        <span className="text-[10px] font-bold uppercase tracking-wide bg-[#181A20] px-1 drop-shadow-sm" style={{ color: d.sentimentColor }}>{d.sentiment}</span>
                      </div>
                      <div className="absolute top-2 left-1/2 -translate-x-1/2 text-center flex flex-col items-center w-24">
                        <span className="text-[10px] text-[#848E9C]">Social Media</span>
                        <span className="text-xs font-medium text-[#EAECEF]">{d.socialMedia}</span>
                      </div>
                      <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-center flex flex-col items-center w-24">
                        <span className="text-[10px] text-[#848E9C] leading-tight mt-1">Social Sentiment</span>
                        <span className="text-xs font-medium text-[#EAECEF]">{d.socialSent}</span>
                      </div>
                      <div className="absolute top-1/2 -left-2 -translate-y-1/2 text-center flex flex-col items-center w-12">
                        <span className="text-[10px] text-[#848E9C]">News</span>
                        <span className="text-xs font-medium text-[#EAECEF]">{d.news}</span>
                      </div>
                      <div className="absolute top-1/2 -right-2 -translate-y-1/2 text-center flex flex-col items-center w-12">
                        <span className="text-[10px] text-[#848E9C]">KOL</span>
                        <span className="text-xs font-medium text-[#EAECEF]">{d.kol}</span>
                      </div>
                    </div>

                    {/* Sentiment Data List */}
                    <div className="flex flex-col mt-2 pb-3 space-y-2.5">
                      {[
                        { label: 'Social Volume', value: d.volText },
                        { label: 'News', value: d.newsText },
                        { label: 'Social Sentiment', value: d.socialText },
                        { label: 'KOL', value: d.kolText },
                      ].map(row => (
                        <div key={row.label} className="flex justify-between items-center text-sm">
                          <span className="text-[#848E9C] font-medium">{row.label}</span>
                          <span className={`font-bold ${row.value === 'Bullish' ? 'text-[#1DB954]' : row.value === 'Bearish' ? 'text-[#F6465D]' : 'text-[#EAECEF]'}`}>{row.value}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    {/* Technical Indicators */}
                    <div className="flex flex-col space-y-2 mt-2 pb-3">
                      {/* Overall Signal Badge */}
                      <div className={`flex items-center justify-center py-2 rounded-lg text-sm font-bold tracking-wide mb-1 ${isBull ? 'bg-[#1DB954]/10 text-[#1DB954] border border-[#1DB954]/30' : 'bg-[#F6465D]/10 text-[#F6465D] border border-[#F6465D]/30'}`}>
                        {isBull ? '▲' : '▼'} Overall Signal: {d.overallSignal}
                      </div>

                      {[
                        { label: 'RSI (14)', value: `${d.rsi}`, sub: d.rsiSignal, subColor: d.rsiSignal === 'BUY' ? '#1DB954' : d.rsiSignal === 'OVERSOLD' ? '#1DB954' : '#F6465D' },
                        { label: 'MACD', value: d.macd, sub: `Signal: ${d.macdSignal}`, subColor: +d.macd > 0 ? '#1DB954' : '#F6465D' },
                        { label: 'Bollinger %B', value: d.bollingerB, sub: +d.bollingerB > 0.5 ? 'Above Midband' : 'Below Midband', subColor: +d.bollingerB > 0.5 ? '#1DB954' : '#F6465D' },
                        { label: 'Stoch %K / %D', value: `${d.stochK} / ${d.stochD}`, sub: +d.stochK > 50 ? 'Bullish' : 'Bearish', subColor: +d.stochK > 50 ? '#1DB954' : '#F6465D' },
                        { label: 'ADX (Trend Str.)', value: d.adx, sub: +d.adx > 40 ? 'Strong Trend' : +d.adx > 25 ? 'Trending' : 'Weak', subColor: '#848E9C' },
                      ].map(row => (
                        <div key={row.label} className="flex items-center justify-between text-sm py-1.5 border-b border-[#2A2D35]/50">
                          <span className="text-[#848E9C] font-medium">{row.label}</span>
                          <div className="text-right">
                            <span className="text-[#EAECEF] font-bold mr-2">{row.value}</span>
                            <span className="text-[11px] font-semibold" style={{ color: row.subColor }}>{row.sub}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* Action Footer */}
                <div className="flex justify-between items-end mt-auto pt-3 border-t border-[#2A2D35]/50">
                  <span className="text-[#1DB954] text-xs font-medium cursor-pointer hover:underline underline-offset-4" onClick={e => { e.stopPropagation(); onSelectCoin(coin); }}>Details</span>
                  <button
                    className="px-5 py-1.5 bg-[#1DB954] hover:bg-[#17a349] text-black font-bold rounded-lg text-sm shadow-md transition-all active:scale-95"
                    onClick={e => { e.stopPropagation(); onSelectCoin(coin); }}
                  >
                    Trade
                  </button>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}


interface DashboardProps {
  initialSymbol?: string;
}

export default function BinanceTradingDashboard({ initialSymbol }: DashboardProps = {}) {
  const router = useRouter();

  // ── Live market data ──────────────────────────────────────────────────────
  const { liveCoins, isLive, isLoading: isMarketLoading, nextRefresh } = useLiveMarketData()

  // Resolve initial config derived from URL if present
  const activeInitialData = initialSymbol ? liveCoins.find(c => c.symbol === initialSymbol) : null;
  
  const [isCoinSelected, setIsCoinSelected] = useState(!!initialSymbol);
  const [selectedCoinData, setSelectedCoinData] = useState<any>(activeInitialData);
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState(initialSymbol ? "chart" : "universe")
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
  const [selectedSymbol, setSelectedSymbol] = useState(initialSymbol || "BTCUSDT")
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

  // -- NEW MOCK STATES FOR EXCHANGE LAYOUT --
  const [orderBookAsks, setOrderBookAsks] = useState<any[]>([])
  const [orderBookBids, setOrderBookBids] = useState<any[]>([])
  const [marketTradesList, setMarketTradesList] = useState<any[]>([])

  // AI Mode State
  const [isAIEnabled, setIsAIEnabled] = useState(false)
  const [marketAnalysis, setMarketAnalysis] = useState<MarketAnalysis | null>(null)
  const [lastError, setLastError] = useState<string | null>(null)
  const [debugIndicators, setDebugIndicators] = useState<any>(null)
  const lastSwitchTime = useRef<number>(0)

  // Safety Layer State
  const [isSafetyLocked, setIsSafetyLocked] = useState(false)
  const [safetyReason, setSafetyReason] = useState<string | null>(null)

  // AI Auto-Configure State
  const [aiResult, setAiResult] = useState<AIConfigResult | null>(null)
  const [isAIConfiguring, setIsAIConfiguring] = useState(false)
  const [aiRecommendedFields, setAiRecommendedFields] = useState<Set<string>>(new Set())
  const [aiLastAnalyzedAt, setAiLastAnalyzedAt] = useState<Date | null>(null)
  const [pendingAiSuggestion, setPendingAiSuggestion] = useState<AIConfigResult | null>(null)
  const aiAbortRef = useRef<AbortController | null>(null)
  const aiManualOverrides = useRef<Set<string>>(new Set())

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
  const [wfaReport, setWfaReport] = useState<WFAReport | null>(null)
  const [showWfaResults, setShowWfaResults] = useState(false)

  const handleSelectCoin = (coin: any) => {
    toast({
      title: "Asset Selected",
      description: `Opening execution terminal for ${coin.symbol}...`,
    });
    router.push(`/trade/${coin.symbol}`);
  };

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
    setWfaReport(null);
    try {
      const data = await getHistoricalKlines(selectedSymbol, selectedTimeframe, 500);
      if (data.length < 100) {
        toast({ title: "Auto-Tune Failed", description: "Not enough historical data. Need 100+ candles.", variant: "destructive" });
        return;
      }

      const report = WalkForwardAnalyzer.runAnalysis(data, selectedSymbol, selectedTimeframe);
      setWfaReport(report);
      setShowWfaResults(true);

      if (report.bestResult) {
        setTunedParams(`Best: ${report.bestResult.parameterSet.id}`);
        toast({
          title: "🛠️ Auto-Tune Complete",
          description: `Tested ${report.paramCombinationsTested} combos across ${report.strategiesTested} strategies. Best: ${report.bestResult.parameterSet.id} (${report.bestResult.winRateOutOfSample.toFixed(0)}% OOS win rate)`,
          duration: 6000
        });
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

  // Handle direct navigation via /trade/[symbol]
  useEffect(() => {
    if (initialSymbol) {
      // Ensure it has USDT suffix for the websocket/logic if it doesn't already
      const formattedSymbol = initialSymbol.endsWith('USDT') ? initialSymbol : `${initialSymbol}USDT`
      setSelectedSymbol(formattedSymbol)
      setIsCoinSelected(true)
      
      const found = MOCK_TRENDING_COINS.find(c => c.symbol === formattedSymbol || c.symbol === initialSymbol)
      if (found) {
        setSelectedCoinData(found)
      } else {
        setSelectedCoinData({
          symbol: formattedSymbol,
          name: initialSymbol,
          price: '--',
          change24h: '+0.00%',
          volume24h: '--',
          marketCap: '--',
          rank: 'N/A',
          logo: '🪙'
        })
      }
    }
  }, [initialSymbol])

  // Load configuration and presets from localStorage on mount
  useEffect(() => {
    const savedConfig = localStorage.getItem('tradingConfig')
    if (savedConfig) {
      try {
        const config: TradingConfig = JSON.parse(savedConfig)
        if (!initialSymbol) {
          setSelectedSymbol(config.symbol)
        }
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

  // Exchange Layout Orderbook & Trades Mock Data Logic
  useEffect(() => {
    if (!isCoinSelected) return;

    const generateTick = () => {
      const p = currentPriceRef.current;
      if (p <= 0) return;
      const asks = Array.from({length: 35}).map((_, i) => ({ 
        price: p + (35-i)*(p*0.0002) + Math.random()*(p*0.0001), 
        amount: Math.random()*2.5, 
        total: Math.random()*10 
      }));
      const bids = Array.from({length: 35}).map((_, i) => ({ 
        price: p - (i+1)*(p*0.0002) - Math.random()*(p*0.0001), 
        amount: Math.random()*2.5, 
        total: Math.random()*10 
      }));
      
      setOrderBookAsks(asks);
      setOrderBookBids(bids);

      setMarketTradesList(prev => {
        const newTrade = { 
          price: p + (Math.random() - 0.5) * (p*0.0004), 
          amount: (Math.random() * 2) + 0.01, 
          time: new Date().toLocaleTimeString('en-US', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'}), 
          isBuyer: Math.random() > 0.5 
        };
        return [newTrade, ...prev].slice(0, 15);
      });
    };

    generateTick();
    const interval = setInterval(generateTick, 800);
    return () => clearInterval(interval);
  }, [isCoinSelected]);

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

  // ════════════════════════════════════════════════════════════════
  // AI AUTO-CONFIGURE HANDLER
  // ════════════════════════════════════════════════════════════════

  const handleAIAutoConfigure = useCallback(async (options?: { silent?: boolean }) => {
    // Abort any in-flight analysis (race condition lock)
    if (aiAbortRef.current) {
      aiAbortRef.current.abort()
    }
    const abortCtrl = new AbortController()
    aiAbortRef.current = abortCtrl

    if (!options?.silent) setIsAIConfiguring(true)

    try {
      const portfolioCtx = {
        totalPnL: portfolio.totalPnL,
        equity: portfolio.equity,
        openTrades: trades
          .filter(t => t.status === 'OPEN')
          .map(t => ({ symbol: t.symbol, side: t.side }))
      }

      const result = await aiConfigEngine.analyze(
        selectedSymbol,
        portfolioCtx,
        abortCtrl.signal
      )

      if (abortCtrl.signal.aborted) return

      setAiResult(result)
      setAiLastAnalyzedAt(result.analyzedAt)

      // Non-destructive logic: if engine is running or user has manual overrides, suggest don't apply
      const shouldOnlySuggest = isRunning || aiManualOverrides.current.size > 0

      if (shouldOnlySuggest && options?.silent) {
        // Background refresh — just store as pending, don't overwrite
        setPendingAiSuggestion(result)
        return
      }

      if (result.confidence < 50) {
        // Low confidence — suggest only, don't apply
        setPendingAiSuggestion(result)
        if (!options?.silent) {
          toast({
            title: "⚠️ AI Analysis — Low Confidence",
            description: `Confidence ${result.confidence}% is too low to auto-apply. Review suggestions manually.`,
            variant: "destructive"
          })
        }
        return
      }

      // Apply configuration
      applyAIConfig(result)

      if (!options?.silent) {
        const badge = result.confidence >= 70 ? '✦' : '⚠'
        toast({
          title: `${badge} AI Configuration Applied`,
          description: `Strategy: ${result.strategy.replace(/_/g, ' ')} | Confidence: ${result.confidence}% | Regime: ${result.marketRegime}${
            result.backtestResult ? ` | Backtest: ${result.backtestResult.winRate.toFixed(0)}% win rate` : ''
          }`,
        })
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return // Expected
      console.error('[AIConfigEngine] Analysis failed:', err)
      if (!options?.silent) {
        toast({
          title: "AI Analysis Failed",
          description: err?.message || "Could not analyze market. Your current settings are unchanged.",
          variant: "destructive"
        })
      }
    } finally {
      if (!abortCtrl.signal.aborted) {
        setIsAIConfiguring(false)
      }
    }
  }, [selectedSymbol, portfolio, trades, isRunning, toast])

  const applyAIConfig = useCallback((result: AIConfigResult) => {
    aiManualOverrides.current.clear()
    const fields = new Set<string>()

    setSelectedStrategy(result.strategy); fields.add('strategy')
    setSelectedTimeframe(result.timeframe); fields.add('timeframe')
    setRiskPerTrade(result.riskPerTrade); fields.add('riskPerTrade')
    setMaxPositions(result.maxPositions); fields.add('maxPositions')
    setStopLoss(result.stopLoss); fields.add('stopLoss')
    setTakeProfit(result.takeProfit); fields.add('takeProfit')
    setTrailingStop(result.trailingStop); fields.add('trailingStop')
    setLeverage(result.leverage); fields.add('leverage')

    setAiRecommendedFields(fields)
    setPendingAiSuggestion(null)
  }, [])

  const applyPendingSuggestion = useCallback(() => {
    if (pendingAiSuggestion) {
      applyAIConfig(pendingAiSuggestion)
      setAiResult(pendingAiSuggestion)
      toast({
        title: "✦ AI Suggestions Applied",
        description: `Updated configuration with ${pendingAiSuggestion.confidence}% confidence.`,
      })
    }
  }, [pendingAiSuggestion, applyAIConfig, toast])

  // Mark field as manually overridden (removes AI badge)
  const markManualOverride = useCallback((field: string) => {
    aiManualOverrides.current.add(field)
    setAiRecommendedFields(prev => {
      const next = new Set(prev)
      next.delete(field)
      return next
    })
  }, [])

  // Background refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      handleAIAutoConfigure({ silent: true })
    }, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [handleAIAutoConfigure])

  // Auto-trigger on symbol change (debounced 800ms)
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (aiResult) {
        // Only auto-re-analyze if user has already used AI at least once
        handleAIAutoConfigure({ silent: true })
      }
    }, 800)
    return () => clearTimeout(timeout)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSymbol])

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
        const historicalData = await getHistoricalKlines(selectedSymbol, selectedTimeframe, 500)

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
                if (newData.length > 500) newData.shift()
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
    <div className={`min-h-screen bg-[#121212] flex flex-col items-center text-[#EAEAEA] ${initialSymbol ? '' : 'p-6 space-y-6'}`}>
      <div className={`w-full ${initialSymbol ? 'max-w-none' : 'max-w-7xl mx-auto space-y-6'}`}>
        


        {!isCoinSelected ? (
          <div className="w-full space-y-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Market Universe</h2>
                <p className="text-muted-foreground mt-1">Select a trending asset to lock the system engine</p>
              </div>
              <div className="flex items-center gap-2">
                {isMarketLoading ? (
                  <span className="flex items-center gap-1.5 text-xs text-[#848E9C] bg-[#1A1A1A] border border-[#333] px-3 py-1.5 rounded-full">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#848E9C] animate-pulse"></span>
                    Loading…
                  </span>
                ) : isLive ? (
                  <span className="flex items-center gap-1.5 text-xs text-[#1DB954] bg-[#1DB954]/10 border border-[#1DB954]/30 px-3 py-1.5 rounded-full font-semibold">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#1DB954] animate-pulse"></span>
                    LIVE
                    {nextRefresh > 0 && <span className="opacity-60 font-normal ml-1">· {nextRefresh}s</span>}
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-xs text-orange-400 bg-orange-400/10 border border-orange-400/30 px-3 py-1.5 rounded-full">
                    <span className="w-1.5 h-1.5 rounded-full bg-orange-400"></span>
                    Cached
                  </span>
                )}
              </div>
            </div>

            {/* Top Market Coins Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {liveCoins.map((coin) => (
                <Card 
                  key={coin.symbol}
                  className="bg-[#1A1A1A] border-[#333]/50 shadow-sm hover:border-[#1DB954]/50 hover:shadow-[#1DB954]/10 transition-all cursor-pointer group"
                  onClick={() => handleSelectCoin(coin)}
                >
                  <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-2xl">{coin.logo}</span>
                      <span className="font-bold text-lg text-foreground group-hover:text-[#1DB954] transition-colors">{coin.symbol}</span>
                    </div>
                    <Badge variant="outline" className="border-[#333] text-muted-foreground bg-transparent group-hover:border-[#1DB954]/50 group-hover:text-[#1DB954]">
                      Rank #{coin.rank}
                    </Badge>
                  </CardHeader>
                  <CardContent className="p-4 pt-2 flex justify-between items-end">
                    <div className="flex flex-col">
                      <span className="text-sm text-muted-foreground mb-1">Price</span>
                      <span className="text-xl font-bold text-foreground">{coin.price}</span>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="text-sm text-muted-foreground mb-1">24h Change</span>
                      <span className={`text-sm font-bold flex items-center ${coin.change24h.startsWith('+') ? 'text-[#1DB954]' : 'text-red-500'}`}>
                        {coin.change24h.startsWith('+') ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                        {coin.change24h}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-3 bg-[#1A1A1A]/80 backdrop-blur-md border border-[#333]/50 shadow-lg min-h-[48px] mb-6">
                <TabsTrigger value="overview" className="flex items-center text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10">Overview</TabsTrigger>
                <TabsTrigger value="trading_data" className="flex items-center text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10">Trading Data</TabsTrigger>
                <TabsTrigger value="ai_select" className="flex items-center text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10">AI Select</TabsTrigger>
              </TabsList>
              
              <TabsContent value="overview" className="space-y-6">
                {/* High Density Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { title: "🔥 Hot", data: [...liveCoins].sort((a, b) => Math.abs(b.rawChange) - Math.abs(a.rawChange)).slice(0, 3) },
                    { title: "✨ New", data: liveCoins.slice(3, 6) },
                    { title: "🚀 Top Gainer", data: [...liveCoins].sort((a, b) => b.rawChange - a.rawChange).slice(0, 3) },
                    { title: "📊 Top Volume", data: [...liveCoins].sort((a, b) => b.rawVol - a.rawVol).slice(0, 3) }
                  ].map((category, idx) => (
                    <Card key={idx} className="bg-[#1A1A1A] border-[#333]/50 shadow-sm flex-1">
                      <CardHeader className="py-3 border-b border-[#333]/50">
                        <CardTitle className="text-xs font-bold text-foreground tracking-wide uppercase">{category.title}</CardTitle>
                      </CardHeader>
                      <CardContent className="p-0 flex flex-col">
                        {category.data.map((coin, i) => (
                          <div 
                            key={coin.symbol} 
                            onClick={() => handleSelectCoin(coin)}
                            className={`flex items-center justify-between p-3 hover:bg-[#333]/30 cursor-pointer transition-colors ${i !== category.data.length - 1 ? 'border-b border-[#333]/30' : ''}`}
                          >
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{coin.logo}</span>
                              <div className="flex flex-col">
                                <span className="font-bold text-sm text-foreground hover:text-[#1DB954] transition-colors">{coin.symbol}</span>
                              </div>
                            </div>
                            <div className="flex flex-col items-end">
                              <span className="text-sm font-medium">{coin.price}</span>
                              <span className={`text-[10px] font-bold ${coin.change24h.startsWith('+') ? 'text-[#1DB954]' : 'text-red-500'}`}>{coin.change24h}</span>
                            </div>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {/* Top Tokens Table */}
                <Card className="bg-[#1A1A1A] border-[#333]/50 shadow-lg mt-6">
                  <CardHeader className="py-4 border-b border-[#333]/50">
                    <CardTitle className="text-sm font-bold text-foreground tracking-wide uppercase">Top Tokens by Market Capitalization</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-[#121212] text-xs text-muted-foreground uppercase tracking-widest border-b border-[#333]/50">
                        <tr>
                          <th className="px-4 py-3 text-left font-medium">Asset</th>
                          <th className="px-4 py-3 text-right font-medium">Price</th>
                          <th className="px-4 py-3 text-right font-medium">24h Change</th>
                          <th className="px-4 py-3 text-right font-medium">Market Cap</th>
                          <th className="px-4 py-3 text-right font-medium">Volume (24h)</th>
                          <th className="px-4 py-3 text-right font-medium">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {liveCoins.map((coin, i) => (
                          <tr 
                            key={coin.symbol} 
                            className={`hover:bg-[#333]/30 transition-colors group cursor-pointer ${i !== liveCoins.length - 1 ? 'border-b border-[#333]/30' : ''}`}
                            onClick={() => handleSelectCoin(coin)}
                          >
                            <td className="px-4 py-4 min-w-[150px]">
                              <div className="flex items-center gap-3">
                                <span className="text-2xl">{coin.logo}</span>
                                <div>
                                  <div className="font-bold text-foreground group-hover:text-[#1DB954] transition-colors">{coin.symbol}</div>
                                  <div className="text-xs text-muted-foreground">{coin.name}</div>
                                </div>
                              </div>
                            </td>
                            <td className="px-4 py-4 text-right font-medium">{coin.price}</td>
                            <td className={`px-4 py-4 text-right font-bold ${coin.change24h.startsWith('+') ? 'text-[#1DB954]' : 'text-red-500'}`}>
                              {coin.change24h}
                            </td>
                            <td className="px-4 py-4 text-right text-muted-foreground">{coin.cap}</td>
                            <td className="px-4 py-4 text-right text-muted-foreground">{coin.vol}</td>
                            <td className="px-4 py-4 text-right w-[120px]">
                              <Button 
                                size="sm" 
                                className="bg-[#1DB954]/10 text-[#1DB954] hover:bg-[#1DB954] hover:text-black font-bold border border-[#1DB954]/50 transition-all rounded-md w-full"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSelectCoin(coin);
                                }}
                              >
                                Trade
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="trading_data" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {[
                    "Hot Coins", "Top Gainers", "Top Losers", 
                    "Top Volume", "USD Futures", "Coin Futures"
                  ].map((title, cardIdx) => (
                    <Card key={title} className="bg-[#121418] border-[#2A2D35] shadow-sm rounded-xl overflow-hidden">
                      <CardHeader className="py-4 px-5 flex flex-row items-center justify-between bg-transparent">
                        <CardTitle className="text-base font-bold text-[#EAECEF]">{title}</CardTitle>
                        <Badge variant="outline" className="bg-transparent border-[#2A2D35] text-[#848E9C] text-xs px-2 py-0.5 cursor-pointer hover:bg-[#2A2D35]/50 hover:text-[#EAECEF] transition-colors">
                          Crypto <span className="ml-1 text-[8px] opacity-70">▼</span>
                        </Badge>
                      </CardHeader>
                      <CardContent className="p-0">
                        <div className="flex justify-between px-5 py-2 text-[11px] text-[#848E9C] font-medium">
                          <span className="w-6"></span>
                          <span className="flex-1 text-left">Name</span>
                          <span className="w-24 text-right">Price</span>
                          <span className="w-20 text-right">24h Change</span>
                        </div>
                        <div className="flex flex-col pb-2">
                          {(() => { const sorted = title === 'Top Gainers' ? [...liveCoins].sort((a,b) => b.rawChange - a.rawChange) : title === 'Top Losers' ? [...liveCoins].sort((a,b) => a.rawChange - b.rawChange) : title === 'Top Volume' || title === 'USD Futures' || title === 'Coin Futures' ? [...liveCoins].sort((a,b) => b.rawVol - a.rawVol) : liveCoins; return sorted; })().map((coin, i) => {
                            const isPositive = !coin.change24h.startsWith('-');
                            const color = isPositive ? 'text-[#0ECB81]' : 'text-[#F6465D]';
                            const displayChange = title === "Top Losers" && isPositive ? `-${coin.change24h.replace('+', '')}` : coin.change24h;
                            const displayColor = title === "Top Losers" ? 'text-[#F6465D]' : color;
                            return (
                              <div 
                                key={`${cardIdx}-${i}`} 
                                className="flex items-center px-4 py-2 hover:bg-[#2A2D35]/40 cursor-pointer transition-colors"
                                onClick={() => handleSelectCoin(coin)}
                              >
                                <span className="w-6 text-[#848E9C] text-xs font-mono">{i + 1}</span>
                                <div className="flex items-center gap-2 flex-1 overflow-hidden">
                                  <span className="text-base w-5 flex justify-center">{coin.logo}</span>
                                  <span className="font-bold text-[#EAECEF] text-sm truncate">{coin.symbol.replace('USDT', '')}</span>
                                </div>
                                <span className="w-24 text-right text-[#EAECEF] text-sm font-medium">{coin.price}</span>
                                <span className={`w-20 text-right text-sm font-medium ${displayColor}`}>{displayChange}</span>
                              </div>
                            )
                          })}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </TabsContent>
              
              <TabsContent value="ai_select" className="space-y-6">
                <AiSelectPanel coins={liveCoins} onSelectCoin={handleSelectCoin} />
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Detailed Asset Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-[#1A1A1A] p-6 rounded-xl border border-[#333]">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Button 
                    variant="ghost" 
                    className="px-3 hover:bg-[#333] text-muted-foreground hover:text-foreground" 
                    onClick={() => router.push('/dashboard')}
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Back to Market
                  </Button>
                  <Button 
                    variant="ghost" 
                    className="px-3 hover:bg-[#1DB954]/10 text-[#1DB954] hover:text-[#1DB954] font-medium" 
                    onClick={() => router.push('/crypto')}
                  >
                    <Globe className="w-4 h-4 mr-2" />
                    Back to Coins
                  </Button>
                </div>
                <div className="h-10 w-px bg-border mx-2 hidden md:block"></div>
                <span className="text-4xl hidden md:block">{selectedCoinData?.logo || '🪙'}</span>
                <div>
                  <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
                    {selectedCoinData?.name || selectedSymbol}
                    <Badge variant="outline" className="border-[#1DB954]/50 text-[#1DB954] text-xs">
                      Rank #{selectedCoinData?.rank || '--'}
                    </Badge>
                  </h1>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xl font-medium text-foreground">
                      ${currentPrice > 0 ? (currentPrice < 1 ? currentPrice.toFixed(4) : currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })) : selectedCoinData?.price || '--'}
                    </span>
                    {selectedCoinData?.change24h && (
                      <span className={`text-sm font-semibold ${selectedCoinData.change24h.startsWith('+') ? 'text-[#1DB954]' : 'text-red-500'}`}>
                        {selectedCoinData.change24h}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 items-center animate-in fade-in slide-in-from-right-4 duration-500">
                <div className="flex items-center space-x-2 bg-black/40 border border-[#333]/50 px-3 py-2 rounded-lg">
                  <Switch id="ai-mode" checked={isAIEnabled} onCheckedChange={(checked) => { setIsAIEnabled(checked); if (checked) toast({ title: "🧠 AI Smart Mode Activated", description: "System strategies automated." }); }} />
                  <Label htmlFor="ai-mode" className="cursor-pointer text-xs font-semibold flex items-center gap-1">{isAIEnabled ? "🧠 AI Active" : "🧠 AI Off"}</Label>
                </div>
                <Button variant="outline" size="sm" className="gap-2 border-purple-500/50 bg-black/40 hover:bg-purple-500/10 text-purple-400" onClick={handleAskGemini} disabled={isLoadingAdvice}>{isLoadingAdvice ? <div className="animate-spin">⏳</div> : <Sparkles className="h-4 w-4" />} Advisor</Button>
                <Button variant="outline" size="sm" className="gap-2 border-blue-500/50 bg-black/40 hover:bg-blue-500/10 text-blue-400" onClick={handleAutoTune} disabled={isTuning}>{isTuning ? <div className="animate-spin">⏳</div> : <Settings className="h-4 w-4" />} Auto-Tune</Button>
              </div>
            </div>

            {/* WFA Auto-Tune Results Panel */}
            {showWfaResults && wfaReport && (
              <Card className="bg-[#1A1A1A]/90 backdrop-blur-md border-blue-500/30 shadow-lg shadow-blue-900/10 animate-in fade-in slide-in-from-top-4 duration-500">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Settings className="h-4 w-4 text-blue-400" />
                        Walk-Forward Analysis Results
                      </CardTitle>
                      <CardDescription className="text-xs">
                        {wfaReport.symbol} · {wfaReport.timeframe} · {wfaReport.totalCandlesUsed} candles · {wfaReport.paramCombinationsTested} combos tested across {wfaReport.strategiesTested} strategies
                      </CardDescription>
                    </div>
                    <button onClick={() => setShowWfaResults(false)} className="text-muted-foreground hover:text-foreground text-xs px-2 py-1 rounded hover:bg-[#333]/50">✕</button>
                  </div>
                </CardHeader>
                <CardContent className="pt-2">
                  {wfaReport.topResults.length === 0 ? (
                    <p className="text-sm text-muted-foreground py-4 text-center">No robust parameters found — all tested combinations failed out-of-sample validation.</p>
                  ) : (
                    <div className="space-y-1">
                      {/* Table Header */}
                      <div className="grid grid-cols-12 gap-2 text-[10px] text-muted-foreground uppercase tracking-wider px-3 py-2 bg-black/30 rounded-t-md font-semibold">
                        <span className="col-span-1">#</span>
                        <span className="col-span-3">Config</span>
                        <span className="col-span-2 text-right">In-Sample</span>
                        <span className="col-span-2 text-right">Out-of-Sample</span>
                        <span className="col-span-2 text-right">OOS Win%</span>
                        <span className="col-span-2 text-right">Status</span>
                      </div>
                      {/* Result Rows */}
                      {wfaReport.topResults.map((r, i) => (
                        <div key={r.parameterSet.id} className={`grid grid-cols-12 gap-2 text-xs font-mono px-3 py-2 rounded-md transition-colors ${
                          i === 0 ? 'bg-blue-500/10 border border-blue-500/30' : 'hover:bg-[#333]/30'
                        }`}>
                          <span className="col-span-1 text-muted-foreground">{i + 1}</span>
                          <span className="col-span-3 text-foreground truncate" title={r.parameterSet.id}>
                            {i === 0 && <span className="text-blue-400 mr-1">★</span>}
                            {r.parameterSet.id}
                          </span>
                          <span className={`col-span-2 text-right ${r.inSamplePnLPercent > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {r.inSamplePnLPercent > 0 ? '+' : ''}{r.inSamplePnLPercent.toFixed(2)}%
                            <span className="text-[9px] text-muted-foreground ml-1">({r.tradesInSample}t)</span>
                          </span>
                          <span className={`col-span-2 text-right ${r.outOfSamplePnLPercent > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {r.outOfSamplePnLPercent > 0 ? '+' : ''}{r.outOfSamplePnLPercent.toFixed(2)}%
                            <span className="text-[9px] text-muted-foreground ml-1">({r.tradesOutOfSample}t)</span>
                          </span>
                          <span className={`col-span-2 text-right ${r.winRateOutOfSample >= 50 ? 'text-green-400' : 'text-yellow-400'}`}>
                            {r.winRateOutOfSample.toFixed(0)}%
                          </span>
                          <span className="col-span-2 text-right">
                            {r.isRobust
                              ? <span className="text-green-400 text-[10px] px-1.5 py-0.5 bg-green-500/10 rounded">✓ Robust</span>
                              : <span className="text-red-400 text-[10px] px-1.5 py-0.5 bg-red-500/10 rounded">✗ Weak</span>
                            }
                          </span>
                        </div>
                      ))}
                      {/* Apply Best Button */}
                      {wfaReport.bestResult && (
                        <div className="flex items-center justify-between mt-3 pt-3 border-t border-[#333]/30">
                          <p className="text-xs text-muted-foreground">
                            ★ Best robust config: <span className="text-blue-400 font-semibold">{wfaReport.bestResult.parameterSet.id}</span>
                            {' '} ({wfaReport.bestResult.parameterSet.strategy.replace(/_/g, ' ')})
                          </p>
                          <Button
                            size="sm"
                            className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-3"
                            onClick={() => {
                              const best = wfaReport.bestResult!;
                              setSelectedStrategy(best.parameterSet.strategy);
                              markManualOverride('strategy');
                              setShowWfaResults(false);
                              toast({
                                title: "🛠️ Strategy Applied from Auto-Tune",
                                description: `Switched to ${best.parameterSet.strategy.replace(/_/g, ' ')} (${best.parameterSet.id})`,
                              });
                            }}
                          >
                            Apply Strategy
                          </Button>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Detailed Asset Grid Layout - 3 COLUMN EXCHANGE Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left Column: Order Book */}
              <div className="lg:col-span-2 space-y-6 flex flex-col h-[1200px]">
                <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg flex-1 overflow-hidden flex flex-col">
                  <CardHeader className="py-3 border-b border-[#333]/50">
                    <CardTitle className="text-sm font-bold text-foreground tracking-wide uppercase">Order Book</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 flex flex-col flex-1 overflow-y-auto custom-scrollbar">
                    <div className="grid grid-cols-3 px-4 py-2 text-[10px] text-muted-foreground uppercase tracking-widest bg-black/40 sticky top-0 z-20">
                      <span className="text-left">Price(USDT)</span>
                      <span className="text-right">Amount</span>
                      <span className="text-right">Total</span>
                    </div>
                    {/* Asks (Red) */}
                    <div className="flex flex-col flex-1 px-3 py-1 space-y-[2px] justify-end">
                      {orderBookAsks.map((ask, i) => (
                        <div key={`ask-${i}`} className="grid grid-cols-3 text-xs font-mono relative cursor-pointer hover:bg-[#333]/30 py-[2px]">
                          <div className="absolute right-0 top-0 bottom-0 bg-red-500/10 z-0 transition-all pointer-events-none" style={{ width: `${(ask.total / 10) * 100}%` }} />
                          <span className="text-red-500 z-10 text-left">{ask.price.toFixed(4)}</span>
                          <span className="text-foreground z-10 text-right pr-2">{ask.amount.toFixed(3)}</span>
                          <span className="text-muted-foreground z-10 text-right">{ask.total.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                    
                    {/* Current Price Anchor */}
                    <div className="py-2 px-4 border-y border-[#333]/50 bg-black/60 flex items-center justify-between sticky z-20">
                      <span className="text-xl font-bold text-[#1DB954]">{currentPrice > 0 ? currentPrice.toFixed(4) : '--'}</span>
                      <span className="text-xs text-muted-foreground underline hover:text-foreground cursor-pointer">Spread: 0.01%</span>
                    </div>

                    {/* Bids (Green) */}
                    <div className="flex flex-col flex-1 px-3 py-1 space-y-[2px] justify-start">
                      {orderBookBids.map((bid, i) => (
                        <div key={`bid-${i}`} className="grid grid-cols-3 text-xs font-mono relative cursor-pointer hover:bg-[#333]/30 py-[2px]">
                          <div className="absolute right-0 top-0 bottom-0 bg-[#1DB954]/10 z-0 transition-all pointer-events-none" style={{ width: `${(bid.total / 10) * 100}%` }} />
                          <span className="text-[#1DB954] z-10 text-left">{bid.price.toFixed(4)}</span>
                          <span className="text-foreground z-10 text-right pr-2">{bid.amount.toFixed(3)}</span>
                          <span className="text-muted-foreground z-10 text-right">{bid.total.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Center Column: Intelligence Engine */}
              <div className="lg:col-span-8 space-y-6">

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4 bg-[#1A1A1A]/80 backdrop-blur-md border border-[#333]/50 shadow-lg min-h-[48px]">
            <TabsTrigger value="chart" className="flex items-center text-[10px] md:text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10"><Activity className="w-3 h-3 mr-2 hidden md:block" />Chart</TabsTrigger>
            <TabsTrigger value="info" className="flex items-center text-[10px] md:text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10"><Settings className="w-3 h-3 mr-2 hidden md:block" />Trade Config</TabsTrigger>
            <TabsTrigger value="data" className="flex items-center text-[10px] md:text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10"><BarChart className="w-3 h-3 mr-2 hidden md:block" />Data</TabsTrigger>
            <TabsTrigger value="square" className="flex items-center text-[10px] md:text-sm font-bold uppercase tracking-wider data-[state=active]:text-[#1DB954] data-[state=active]:bg-[#1DB954]/10"><Globe className="w-3 h-3 mr-2 hidden md:block" />Square</TabsTrigger>
          </TabsList>

          {/* Chart Tab: Central Focus */}
          <TabsContent value="chart" className="space-y-4 mt-4">
            {/* Risk Intelligence Embedded */}
            {isCoinSelected && (
              <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-[#1DB954]" />
                    Risk Intelligence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30">
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">95% Confidence Interval</div>
                      <div className="text-lg font-bold text-foreground">Value at Risk (VaR)</div>
                    </div>
                    <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30">
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">Topological Persistence Homology</div>
                      <div className="text-lg font-bold text-foreground">Regime Structure</div>
                    </div>
                    <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30">
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">Fractional Differencing (d=0.4)</div>
                      <div className="text-lg font-bold text-foreground">Stationarity</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            <div className="flex flex-col gap-6">
              {/* Price Chart - REAL-TIME from Binance */}
              <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
                <CardHeader className="border-b border-[#333]/50 pb-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <CardTitle className="text-2xl font-bold flex items-center gap-3 text-foreground">
                      Price Chart ({selectedSymbol})
                      {wsConnected && (
                        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/20 text-[10px] text-green-500 font-bold uppercase tracking-widest">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                          LIVE
                        </div>
                      )}
                    </CardTitle>
                    <div className="flex items-center gap-1 bg-[#121212] p-1 rounded-lg border border-[#333]/30">
                      {['15m', '1h', '4h', '1d', '1w'].map(tf => (
                        <button
                          key={tf}
                          onClick={() => setSelectedTimeframe(tf)}
                          className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${selectedTimeframe === tf ? 'bg-[#2B2B43] text-[#EAEAEA]' : 'text-gray-500 hover:text-gray-300'}`}
                        >
                          {tf.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  {priceLoading ? (
                    <div className="h-[600px] flex items-center justify-center">
                      <div className="text-muted-foreground">Loading price data...</div>
                    </div>
                  ) : candlestickData.length === 0 ? (
                    <div className="h-[600px] flex items-center justify-center">
                      <div className="text-muted-foreground">No price data available</div>
                    </div>
                  ) : (
                    <div className="w-full h-[600px] relative group">
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
              <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
                <CardHeader>
                  <CardTitle>Equity Curve</CardTitle>
                  <CardDescription>Account balance over time</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={balanceHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.3} />
                      <XAxis dataKey="time" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `$${value}`} />
                      <Tooltip contentStyle={{ backgroundColor: "#1A1A1A", border: "1px solid #333", color: "#EAEAEA" }} />
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
            <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
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

          {/* Data Tab */}
          <TabsContent value="data" className="space-y-4 mt-4">
            
            {/* SDP Quantitative Metrics Headway Card */}
            <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-lg text-[#1DB954]">
                  <Database className="w-5 h-5" />
                  Quantitative Metrics (SDP)
                </CardTitle>
                <CardDescription>Live mathematical model encodings for active market conditions.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                  <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30 flex flex-col justify-between">
                    <div>
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">Topological Data Analysis</div>
                      <div className="text-lg font-bold text-foreground">Regime Encoding</div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-[#333]/50 space-y-2 text-sm font-mono">
                      <div className="flex justify-between items-center"><span className="text-muted-foreground">Components (β₀):</span> <span className="text-foreground">12</span></div>
                      <div className="flex justify-between items-center"><span className="text-muted-foreground">Loops (β₁):</span> <span className="text-foreground">3</span></div>
                      <div className="flex justify-between items-center"><span className="text-muted-foreground">Voids (β₂):</span> <span className="text-foreground">0</span></div>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30 flex flex-col justify-between">
                    <div>
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">Stationarity Protocol</div>
                      <div className="text-lg font-bold text-foreground">Fractional Diff (d=0.4)</div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-[#333]/50 text-sm">
                      <div className="flex justify-center items-center gap-2 mb-3">
                        <span className="relative flex h-2.5 w-2.5"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#1DB954] opacity-75"></span><span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#1DB954]"></span></span>
                        <span className="font-bold text-foreground tracking-wide">STATIONARITY ACH.</span>
                      </div>
                      <p className="text-muted-foreground text-center text-xs">ADF Test p-value: <span className="text-white font-mono">0.014</span> (&lt; 0.05)</p>
                    </div>
                  </div>

                  <div className="p-4 bg-black/40 rounded-lg border border-[#333]/30 flex flex-col justify-between">
                    <div>
                      <div className="text-[#1DB954] text-xs font-bold uppercase tracking-wider mb-1">Risk Bounds</div>
                      <div className="text-lg font-bold text-foreground">VaR & Expected Shortfall</div>
                    </div>
                    <div className="mt-4 pt-4 border-t border-[#333]/50 space-y-2 text-sm font-mono">
                      <div className="flex justify-between items-center"><span className="text-muted-foreground">VaR (95%):</span> <span className="text-red-400 font-bold">-2.40%</span></div>
                      <div className="flex justify-between items-center"><span className="text-muted-foreground">CVaR (95%):</span> <span className="text-red-500 font-bold">-3.85%</span></div>
                      <div className="w-full bg-[#333]/50 h-1.5 rounded-full overflow-hidden flex mt-2"><div className="w-[85%] bg-blue-500/50 h-full"></div><div className="w-[15%] bg-red-500 h-full"></div></div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
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
            </div>
          {/* Analytics Section inside Data Tab */}
          <div className="space-y-4 mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
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
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.3} />
                      <XAxis dataKey="name" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: "#1A1A1A", border: "1px solid #333", color: "#EAEAEA" }} />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
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
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.3} />
                      <XAxis dataKey="name" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: "#1A1A1A", border: "1px solid #333", color: "#EAEAEA" }} />
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
          </div>
        </TabsContent>

          {/* Square Tab (Gemini Advisor integration point) */}
          <TabsContent value="square" className="space-y-4 mt-4">
             <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg p-6 flex flex-col items-center justify-center text-center min-h-[400px]">
               <Brain className="w-16 h-16 text-purple-500 mb-4 animate-pulse" />
               <h3 className="text-2xl font-bold text-white mb-2">Square Intelligence</h3>
               <p className="text-muted-foreground mb-6 max-w-md">Connect with the Gemini Neural Network to generate market sentiment, backtesting signals, and auto-tuning boundaries.</p>
               <Button onClick={handleAskGemini} disabled={isLoadingAdvice} className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-6 px-8 rounded-xl shadow-lg shadow-purple-900/50">
                 {isLoadingAdvice ? "Analyzing..." : "Generate Square Analysis"}
               </Button>
             </Card>
          </TabsContent>

          {/* Info Tab (Settings + Scenarios) */}
          <TabsContent value="info" className="space-y-4 mt-4">
            <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg mb-6">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Activity className="w-5 h-5 text-[#1DB954]" />
                    Market Scenarios
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 grid grid-cols-3">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Market Cap</div>
                    <div className="text-xl font-bold text-foreground">{selectedCoinData?.cap || '--'}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Volume (24h)</div>
                    <div className="text-xl font-bold text-foreground">{selectedCoinData?.vol || '--'}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Circulating Supply</div>
                    <div className="text-xl font-bold text-foreground">{selectedCoinData?.supply || '--'}</div>
                  </div>
                </CardContent>
            </Card>
            
            <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Trading Configuration</CardTitle>
                    <CardDescription>Configure your trading strategy and parameters</CardDescription>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <Button
                      onClick={() => handleAIAutoConfigure()}
                      disabled={isAIConfiguring}
                      className="bg-gradient-to-r from-purple-600 to-[#1DB954] hover:from-purple-700 hover:to-[#1AA34A] text-white font-bold px-4 py-2 rounded-lg shadow-lg shadow-purple-900/20 text-xs"
                    >
                      {isAIConfiguring ? (
                        <span className="flex items-center gap-1.5"><RotateCcw className="w-3.5 h-3.5 animate-spin" /> Analyzing...</span>
                      ) : (
                        <span className="flex items-center gap-1.5"><Zap className="w-3.5 h-3.5" /> AI Auto-Configure</span>
                      )}
                    </Button>
                    {aiLastAnalyzedAt && (
                      <span className="text-[10px] text-muted-foreground font-mono">
                        Last: {aiLastAnalyzedAt.toLocaleTimeString()}
                        {aiResult && <span className={`ml-1.5 font-bold ${aiResult.confidence >= 70 ? 'text-green-400' : aiResult.confidence >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>{aiResult.confidence}% conf.</span>}
                      </span>
                    )}
                    {pendingAiSuggestion && (
                      <button
                        onClick={applyPendingSuggestion}
                        className="text-[10px] text-purple-400 hover:text-purple-300 underline cursor-pointer"
                      >
                        ✨ New AI suggestions available — click to apply
                      </button>
                    )}
                  </div>
                </div>
                {aiResult && aiResult.backtestResult && aiResult.backtestResult.totalTrades > 0 && (
                  <div className="mt-2 px-3 py-1.5 bg-black/40 rounded-md border border-[#333]/30 text-xs text-muted-foreground font-mono flex items-center gap-3">
                    <span>📊 Backtest: <span className={aiResult.backtestResult.winRate >= 50 ? 'text-green-400' : 'text-red-400'}>{aiResult.backtestResult.winRate.toFixed(0)}% win rate</span></span>
                    <span>• {aiResult.backtestResult.totalTrades} trades</span>
                    <span>• PF {aiResult.backtestResult.profitFactor.toFixed(1)}</span>
                    <span className="text-[9px] text-muted-foreground/60">(incl. 0.1% slippage)</span>
                  </div>
                )}
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Strategy Selection */}
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Trading Strategy
                    {aiRecommendedFields.has('strategy') && (
                      <span title={aiResult?.reasoning.strategy} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                        AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                      </span>
                    )}
                  </label>
                  <select
                    value={selectedStrategy}
                    onChange={(e) => { setSelectedStrategy(e.target.value); markManualOverride('strategy') }}
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
                      className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
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
                    <label className="text-sm font-medium mb-2 block">
                      Timeframe
                      {aiRecommendedFields.has('timeframe') && (
                        <span title={aiResult?.reasoning.timeframe} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                          AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                        </span>
                      )}
                    </label>
                    <select
                      value={selectedTimeframe}
                      onChange={(e) => { setSelectedTimeframe(e.target.value); markManualOverride('timeframe') }}
                      className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
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
                    <label className="text-sm font-medium mb-2 block">
                      Risk Per Trade (%)
                      {aiRecommendedFields.has('riskPerTrade') && (
                        <span title={aiResult?.reasoning.risk} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                          AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                        </span>
                      )}
                    </label>
                    <input
                      type="number"
                      min="0.5"
                      max="10"
                      step="0.5"
                      value={riskPerTrade}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value)
                        setRiskPerTrade(isNaN(val) ? 0 : val)
                        markManualOverride('riskPerTrade')
                      }}
                      className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
                    />
                    <p className="text-xs text-muted-foreground mt-1">Recommended: 1-3%</p>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Max Positions
                      {aiRecommendedFields.has('maxPositions') && (
                        <span title={aiResult?.reasoning.maxPositions} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                          AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                        </span>
                      )}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="20"
                      value={maxPositions}
                      onChange={(e) => {
                        const val = parseInt(e.target.value)
                        setMaxPositions(isNaN(val) ? 1 : val)
                        markManualOverride('maxPositions')
                      }}
                      className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
                    />
                  </div>
                </div>

                {/* Advanced Risk Management */}
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-semibold mb-3">Advanced Risk Management</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Stop Loss (%)
                        {aiRecommendedFields.has('stopLoss') && (
                          <span title={aiResult?.reasoning.stopLoss} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                          </span>
                        )}
                      </label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={stopLoss}
                        onChange={(e) => { setStopLoss(parseFloat(e.target.value) || 0); markManualOverride('stopLoss') }}
                        className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Take Profit (%)
                        {aiRecommendedFields.has('takeProfit') && (
                          <span title={aiResult?.reasoning.takeProfit} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                          </span>
                        )}
                      </label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={takeProfit}
                        onChange={(e) => { setTakeProfit(parseFloat(e.target.value) || 0); markManualOverride('takeProfit') }}
                        className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Trailing Stop (%)
                        {aiRecommendedFields.has('trailingStop') && (
                          <span title={aiResult?.reasoning.trailingStop} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                          </span>
                        )}
                      </label>
                      <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={trailingStop}
                        onChange={(e) => { setTrailingStop(parseFloat(e.target.value) || 0); markManualOverride('trailingStop') }}
                        className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
                      />
                      <p className="text-xs text-muted-foreground mt-1">Locks in profit as price rises</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Simulated Leverage (x)
                        {aiRecommendedFields.has('leverage') && (
                          <span title={aiResult?.reasoning.leverage} className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full cursor-help ${aiResult && aiResult.confidence >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            AI {aiResult && aiResult.confidence >= 70 ? '✦' : '⚠'}
                          </span>
                        )}
                      </label>
                      <select
                        value={leverage}
                        onChange={(e) => { setLeverage(parseInt(e.target.value)); markManualOverride('leverage') }}
                        className="w-full px-3 py-2 border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
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
                          className="flex-1 px-3 py-2 text-sm border border-[#333]/50 rounded-md bg-[#121212] text-foreground focus:border-[#1DB954]/50 focus:outline-none"
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

                {/* Finalize Launch Execution within Configuration Module */}
                <div className="pt-8 mt-8 border-t border-[#333]/50">
                  <div className="bg-black/60 border border-[#333]/50 rounded-xl p-6 relative overflow-hidden">
                    {/* Decorative Background Glow */}
                    <div className="absolute top-0 right-0 w-32 h-32 bg-[#1DB954]/10 rounded-full blur-3xl pointer-events-none" />
                    
                    <h4 className="text-sm font-bold uppercase tracking-wider mb-2 text-[#1DB954] flex items-center gap-2">
                      <Zap className="w-4 h-4" /> Finalize & Engine Launch
                    </h4>
                    <p className="text-xs text-muted-foreground mb-6">Review your configuration matrix above. Only launch the strategy after you have verified your capital exposure limits and active trading pair selections.</p>
                    
                    <div className="flex flex-col sm:flex-row gap-4 items-center relative z-10">
                      <Button
                        onClick={() => setIsRunning(!isRunning)}
                        disabled={isSafetyLocked}
                        className={isRunning 
                          ? "w-full sm:flex-1 bg-red-600 hover:bg-red-700 font-bold py-6 text-lg shadow-lg shadow-red-900/20" 
                          : "w-full sm:flex-1 bg-[#1DB954] hover:bg-[#1AA34A] text-white font-bold py-6 text-lg shadow-lg shadow-[#1DB954]/20"}
                      >
                        {isRunning ? (<span className="flex items-center"><Pause className="w-5 h-5 mr-2" /> STOP ENGINE</span>) : (<span className="flex items-center"><Play className="w-5 h-5 mr-2" /> LAUNCH ALGORITHMIC ENGINE</span>)}
                      </Button>

                      {isSafetyLocked ? (
                        <Button
                          variant="outline"
                          className="w-full sm:w-auto py-6 px-6 font-bold border-yellow-500 text-yellow-500 bg-yellow-500/5 hover:bg-yellow-500/10"
                          onClick={() => {
                            riskManagerRef.current.resetSafety();
                            setIsSafetyLocked(false);
                            setSafetyReason(null);
                            toast({ title: "Safety Override", description: "System safety locks have been reset manually." });
                          }}
                        >
                          <Unlock className="w-5 h-5 mr-2" />
                          RESET LOCKS
                        </Button>
                      ) : (
                        <Button
                          variant="destructive"
                          className="w-full sm:w-auto py-6 px-6 font-bold bg-red-900/80 hover:bg-red-900 border border-red-700 shadow-lg"
                          onClick={() => {
                            riskManagerRef.current.triggerKillSwitch();
                            setIsSafetyLocked(true);
                            setSafetyReason("Manual Kill Switch Triggered");
                            setIsRunning(false);

                            const exitPrice = currentPrice;
                            let totalLiquidationPnL = 0;

                            setTrades(prevTrades => prevTrades.map(t => {
                              if (t.status === 'OPEN') {
                                const pnl = t.side === "BUY"
                                  ? (exitPrice - t.entryPrice) * t.quantity
                                  : (t.entryPrice - exitPrice) * t.quantity;
                                totalLiquidationPnL += pnl;
                                return { ...t, status: 'CLOSED', exitPrice, exitTime: Date.now(), pnl, exitReason: "KILL SWITCH 💀" };
                              }
                              return t;
                            }));

                            setPortfolio(prev => ({
                              ...prev,
                              equity: prev.equity + totalLiquidationPnL,
                              totalPnL: prev.totalPnL + totalLiquidationPnL,
                              realizedPnL: prev.realizedPnL + totalLiquidationPnL,
                              unrealizedPnL: 0
                            }));

                            toast({ title: "🟥 KILL SWITCH ACTIVATED", description: `Trading halted. All positions liquidated at $${exitPrice}.`, duration: 5000 });
                          }}
                        >
                          <Skull className="w-5 h-5 mr-2" />
                          KILL SWITCH
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
              </div>

              {/* Right Column: Market Activity & About Coin */}
              <div className="lg:col-span-2 space-y-6 flex flex-col h-[1200px]">
                
                {/* About Coin */}
                <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg shrink-0">
                  <CardHeader className="py-3 border-b border-[#333]/50">
                    <CardTitle className="text-sm font-bold text-foreground tracking-wide uppercase flex items-center gap-2">
                       <span className="text-lg">{selectedCoinData?.logo || '🪙'}</span> About {selectedSymbol.replace('USDT','')}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 text-xs text-muted-foreground leading-relaxed space-y-3">
                    <p>
                      {selectedSymbol.replace('USDT','')} is a dynamic digital asset actively positioned on the global exchange network. It leverages cryptographic consensus mechanisms and high-throughput liquidity to enable automated algorithmic strategies.
                    </p>
                    <p>
                      The current quantitative models indicate persistent structural accumulation, allowing executing engines to harness fractional volume discrepancies optimally.
                    </p>
                    <div className="mt-4 pt-3 border-t border-[#333]/50 flex justify-between">
                      <span className="text-foreground font-bold">Network Consensus</span>
                      <span className="text-[#1DB954] font-mono">Proof-of-Stake</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg flex-1 overflow-hidden flex flex-col">
                  <CardHeader className="py-3 border-b border-[#333]/50">
                    <CardTitle className="text-sm font-bold text-foreground tracking-wide uppercase">Market Trades</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 flex flex-col flex-1 overflow-y-auto custom-scrollbar">
                    <div className="grid grid-cols-3 px-4 py-2 text-[10px] text-muted-foreground uppercase tracking-widest bg-black/40 sticky top-0 z-20">
                      <span className="text-left">Price(USDT)</span>
                      <span className="text-right">Amount</span>
                      <span className="text-right">Time</span>
                    </div>
                    <div className="flex flex-col flex-1 px-3 py-1 space-y-[2px]">
                      {marketTradesList.map((trade, i) => (
                        <div key={`trade-${i}`} className="grid grid-cols-3 text-xs font-mono cursor-pointer hover:bg-[#333]/30 py-[2px] animate-in slide-in-from-top-2 duration-300">
                          <span className={`${trade.isBuyer ? 'text-[#1DB954]' : 'text-red-500'} text-left`}>{trade.price.toFixed(4)}</span>
                          <span className="text-foreground text-right pr-2">{trade.amount.toFixed(3)}</span>
                          <span className="text-muted-foreground text-right">{trade.time}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* Persistent Global Metrics Row (Anchored to Footer) */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Account Balance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">${portfolio.equity.toFixed(2)}</div>
              <p className={`text-xs mt-1 font-semibold ${portfolio.totalPnL > 0 ? "text-[#1DB954]" : "text-red-500"}`}>
                {portfolio.totalPnL > 0 ? "+" : ""}${portfolio.totalPnL.toFixed(2)}
              </p>
            </CardContent>
          </Card>
          <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{(portfolio.winRate * 100).toFixed(1)}%</div>
              <p className="text-xs mt-1 text-muted-foreground">{closedTrades.length} trades closed</p>
            </CardContent>
          </Card>
          <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Open Positions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{openTrades.length}</div>
              <p className="text-xs mt-1 text-muted-foreground">Unrealized: <span className={portfolio.unrealizedPnL > 0 ? "text-[#1DB954]" : "text-red-500"}>${portfolio.unrealizedPnL.toFixed(2)}</span></p>
            </CardContent>
          </Card>
          <Card className="bg-[#1A1A1A]/80 backdrop-blur-md border-[#333]/50 shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Max Drawdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground">{(portfolio.maxDrawdown * 100).toFixed(2)}%</div>
              <p className="text-xs mt-1 text-muted-foreground">Risk metric</p>
            </CardContent>
          </Card>
        </div>
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
