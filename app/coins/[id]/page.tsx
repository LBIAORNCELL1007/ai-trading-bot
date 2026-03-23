'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { ArrowLeft, ExternalLink, Globe, Github } from 'lucide-react'
import { useCoinDetail } from '@/hooks/use-crypto-data'
import { FinancialChart } from '@/components/financial-chart'
import { CryptoSearch } from '@/components/crypto-search'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { binanceWS, getHistoricalKlines, type BinanceKline } from '@/lib/binance-websocket'

interface CoinDetailPageProps {
  params: Promise<{ id: string }>
}

// Helper to map CoinGecko ID to Binance Symbol
const getBinanceSymbol = (id: string, symbol: string): string => {
  const manualMap: Record<string, string> = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'binancecoin': 'BNBUSDT',
    'solana': 'SOLUSDT',
    'ripple': 'XRPUSDT',
    'cardano': 'ADAUSDT',
    'dogecoin': 'DOGEUSDT',
    'avalanche-2': 'AVAXUSDT',
    'polkadot': 'DOTUSDT',
    'chainlink': 'LINKUSDT',
    'matic-network': 'MATICUSDT',
    'shiba-inu': 'SHIBUSDT',
    'litecoin': 'LTCUSDT',
    'uniswap': 'UNIUSDT',
    'cosmos': 'ATOMUSDT',
    'ethereum-classic': 'ETCUSDT',
    'near': 'NEARUSDT',
    'aptos': 'APTUSDT',
    'arbitrum': 'ARBUSDT',
    'stellar': 'XLMUSDT',
    'pepe': 'PEPEUSDT'
  }
  if (manualMap[id]) return manualMap[id]
  // Fallback: Try to construct it from symbol (e.g. 'btc' -> 'BTCUSDT')
  return `${symbol.toUpperCase()}USDT`
}

export default function CoinDetailPage({ params }: CoinDetailPageProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d')
  const [resolvedParams, setResolvedParams] = useState<{ id: string } | null>(null)

  // Chart state
  const [candlestickData, setCandlestickData] = useState<BinanceKline[]>([])
  const [chartLoading, setChartLoading] = useState(true)
  const [wsConnected, setWsConnected] = useState(false)
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)

  const { coin, loading: coinLoading } = useCoinDetail(resolvedParams?.id || '')

  useEffect(() => {
    params.then(setResolvedParams)
  }, [params])

  // Fetch Binance Data & Connect WS
  useEffect(() => {
    if (!coin || !resolvedParams) return

    let isMounted = true
    let unsubscribe: (() => void) | null = null

    // Determine symbol
    const binanceSymbol = getBinanceSymbol(resolvedParams.id, coin.symbol)

    const initializeChart = async () => {
      if (isMounted) setChartLoading(true)
      try {
        // Fetch historical data - Request 1000 candles for max history
        // 1000 candles * 1d = ~3 years. 1w = ~19 years.
        const limit = 1000
        const historicalData = await getHistoricalKlines(binanceSymbol, selectedTimeframe, limit)

        if (isMounted) {
          if (historicalData.length > 0) {
            setCandlestickData(historicalData)
            setCurrentPrice(historicalData[historicalData.length - 1].close)
          } else {
            // If implicit mapping fails, maybe try without USDT if it's a stablecoin? 
            // But for now, empty state is better than crash.
          }
        }

        // Subscribe to real-time updates
        unsubscribe = binanceWS.subscribeKline(
          binanceSymbol,
          selectedTimeframe,
          (kline) => {
            if (!isMounted) return
            setWsConnected(true)
            setCurrentPrice(kline.close)

            setCandlestickData((prev) => {
              const newData = [...prev]
              const lastCandle = newData[newData.length - 1]

              if (lastCandle && lastCandle.time === kline.time) {
                newData[newData.length - 1] = kline
              } else if (kline.isClosed || !lastCandle || kline.time > lastCandle.time) {
                newData.push(kline)
                if (newData.length > 2000) newData.shift() // Keep chart performant
              }
              return newData
            })
          }
        )
      } catch (e) {
        console.error("Chart init error:", e)
      } finally {
        if (isMounted) setChartLoading(false)
      }
    }

    initializeChart()

    return () => {
      isMounted = false
      if (unsubscribe) unsubscribe()
      setWsConnected(false)
    }
  }, [coin, resolvedParams, selectedTimeframe])


  if (!resolvedParams || coinLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-background/80">
        <div className="container mx-auto px-4 py-8">
          <Link href="/crypto" className="flex items-center gap-2 mb-8 text-primary hover:text-primary/80">
            <ArrowLeft className="w-4 h-4" />
            Back to Coins
          </Link>
          <div className="space-y-6">
            <Skeleton className="h-40 rounded-lg" />
            <Skeleton className="h-96 rounded-lg" />
          </div>
        </div>
      </div>
    )
  }

  if (!coin) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-background/80">
        <div className="container mx-auto px-4 py-8">
          <Link href="/crypto" className="flex items-center gap-2 mb-8 text-primary hover:text-primary/80">
            <ArrowLeft className="w-4 h-4" />
            Back to Coins
          </Link>
          <Card>
            <CardContent className="p-12 text-center">
              <p className="text-muted-foreground">Coin not found</p>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const marketData = coin.market_data || {}

  // Use real-time price if available, otherwise CoinGecko price
  const displayPrice = currentPrice || marketData.current_price?.usd || 0

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Link href="/crypto" className="flex items-center gap-2 mb-8 text-muted-foreground hover:text-primary transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Back to Market
        </Link>

        {/* Coin Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
          <div className="flex items-center gap-4">
            {coin.image?.large ? (
              <img src={coin.image.large} alt={coin.name} className="w-16 h-16 rounded-full" />
            ) : (
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center text-2xl font-bold">
                {coin.symbol?.[0]?.toUpperCase()}
              </div>
            )}
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-2">
                {coin.name}
                <span className="text-muted-foreground text-xl">({coin.symbol?.toUpperCase()})</span>
              </h1>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="secondary" className="text-xs">Rank #{coin.market_cap_rank}</Badge>
                {wsConnected && (
                  <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30 text-xs">
                    <span className="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse" />
                    LIVE
                  </Badge>
                )}
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-4xl font-bold">
              ${displayPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
            </div>
            <div className={`flex items-center justify-end gap-1 text-lg font-medium ${(marketData.price_change_percentage_24h || 0) >= 0 ? "text-green-500" : "text-red-500"
              }`}>
              {(marketData.price_change_percentage_24h || 0) >= 0 ? '▲' : '▼'}
              {Math.abs(marketData.price_change_percentage_24h || 0).toFixed(2)}% (24h)
            </div>
          </div>
        </div>

        {/* Quick Search */}
        <div className="mb-6 max-w-md">
          <p className="text-sm text-muted-foreground mb-2">Switch Coin:</p>
          <CryptoSearch />
        </div>

        {/* Chart Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <Card className="lg:col-span-2 border-none shadow-none bg-background/50">
            <CardHeader className="px-0 pt-0 pb-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <CardTitle>Price Chart ({getBinanceSymbol(resolvedParams?.id || '', coin.symbol)})</CardTitle>
                <ToggleGroup type="single" value={selectedTimeframe} onValueChange={(val) => val && setSelectedTimeframe(val)}>
                  <ToggleGroupItem value="15m" size="sm">15m</ToggleGroupItem>
                  <ToggleGroupItem value="1h" size="sm">1H</ToggleGroupItem>
                  <ToggleGroupItem value="4h" size="sm">4H</ToggleGroupItem>
                  <ToggleGroupItem value="1d" size="sm">1D</ToggleGroupItem>
                  <ToggleGroupItem value="1w" size="sm">1W</ToggleGroupItem>
                </ToggleGroup>
              </div>
            </CardHeader>
            <CardContent className="px-0">
              {chartLoading ? (
                <div className="h-[500px] flex items-center justify-center bg-muted/10 rounded-lg">
                  Loading Chart Data...
                </div>
              ) : (
                <div className="h-[500px] w-full rounded-lg overflow-hidden border border-border/50 bg-card">
                  {candlestickData.length > 0 ? (
                    <FinancialChart data={candlestickData} />
                  ) : (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      Chart data not available for this pair on Binance.
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Market Stats */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Market Scenarios</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between py-2 border-b">
                  <span className="text-muted-foreground">Market Cap</span>
                  <span className="font-medium">${marketData.market_cap?.usd?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between py-2 border-b">
                  <span className="text-muted-foreground">Volume (24h)</span>
                  <span className="font-medium">${marketData.total_volume?.usd?.toLocaleString() || 'N/A'}</span>
                </div>
                <div className="flex justify-between py-2 border-b">
                  <span className="text-muted-foreground">Circulating Supply</span>
                  <span className="font-medium">{marketData.circulating_supply?.toLocaleString() || 'N/A'} {coin.symbol?.toUpperCase()}</span>
                </div>
                <div className="flex justify-between py-2 border-b">
                  <span className="text-muted-foreground">All Time High</span>
                  <span className="font-medium">${marketData.ath?.usd?.toLocaleString() || 'N/A'}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Project Info</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {coin.links?.homepage?.[0] && (
                  <a href={coin.links.homepage[0]} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-primary">
                    <Globe className="w-4 h-4" /> Website
                  </a>
                )}
                {coin.links?.repos_url?.github?.[0] && (
                  <a href={coin.links.repos_url.github[0]} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-primary">
                    <Github className="w-4 h-4" /> Source Code
                  </a>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Description */}
        <Card className="mb-8">
          <CardHeader><CardTitle>About {coin.name}</CardTitle></CardHeader>
          <CardContent>
            <div className="prose prose-invert max-w-none" dangerouslySetInnerHTML={{ __html: coin.description?.en || 'No description available.' }} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

