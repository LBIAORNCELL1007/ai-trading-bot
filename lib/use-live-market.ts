"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import type { CoinData } from "@/types/crypto"

// ─── Map CoinGecko `symbol` (uppercase) → Binance symbol & static metadata ───
const COIN_META: Record<string, {
  binanceSymbol: string; logo: string; rank: number; supply: string
}> = {
  BTC:  { binanceSymbol: "BTCUSDT",   logo: "₿",    rank: 1,  supply: "19.6M" },
  ETH:  { binanceSymbol: "ETHUSDT",   logo: "Ξ",    rank: 2,  supply: "120M"  },
  XRP:  { binanceSymbol: "XRPUSDT",   logo: "✕",    rank: 3,  supply: "54B"   },
  SOL:  { binanceSymbol: "SOLUSDT",   logo: "◎",    rank: 5,  supply: "443M"  },
  BNB:  { binanceSymbol: "BNBUSDT",   logo: "BNB",  rank: 4,  supply: "149M"  },
  ADA:  { binanceSymbol: "ADAUSDT",   logo: "₳",    rank: 9,  supply: "35B"   },
  DOGE: { binanceSymbol: "DOGEUSDT",  logo: "Ð",    rank: 8,  supply: "143B"  },
  AVAX: { binanceSymbol: "AVAXUSDT",  logo: "🔺",   rank: 11, supply: "377M"  },
  SHIB: { binanceSymbol: "SHIBUSDT",  logo: "🐕",   rank: 13, supply: "589T"  },
  LINK: { binanceSymbol: "LINKUSDT",  logo: "🔗",   rank: 14, supply: "587M"  },
  MATIC:{ binanceSymbol: "MATICUSDT", logo: "⬡",    rank: 16, supply: "9.2B"  },
  UNI:  { binanceSymbol: "UNIUSDT",   logo: "🦄",   rank: 18, supply: "600M"  },
}

// Ordered list of symbols we want to track (in display order)
const TRACK_SYMBOLS = ["BTC","ETH","XRP","SOL","BNB","ADA","DOGE","AVAX","SHIB","LINK","MATIC","UNI"]

export type LiveCoin = {
  symbol: string       // Binance symbol e.g. "BTCUSDT"
  name: string
  price: string
  change24h: string
  logo: string
  rank: number
  cap: string
  vol: string
  supply: string
  rawPrice: number
  rawChange: number
  rawVol: number
}

// ─── Formatting helpers ───────────────────────────────────────────────────────
function fmtPrice(p: number): string {
  if (p >= 10000) return `$${p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  if (p >= 1)     return `$${p.toFixed(2)}`
  if (p >= 0.01)  return `$${p.toFixed(4)}`
  return `$${p.toFixed(8)}`
}

function fmtChange(pct: number): string {
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`
}

function fmtVol(v: number): string {
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(0)}M`
  return `$${v.toFixed(0)}`
}

function fmtCap(cap: number | null): string {
  if (!cap) return "—"
  if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`
  if (cap >= 1e9)  return `$${(cap / 1e9).toFixed(0)}B`
  if (cap >= 1e6)  return `$${(cap / 1e6).toFixed(0)}M`
  return `$${cap.toFixed(0)}`
}

// ─── Static fallback data (shown before first fetch completes) ────────────────
const FALLBACK: LiveCoin[] = TRACK_SYMBOLS.map(sym => {
  const m = COIN_META[sym]
  const fallbackPrices: Record<string, number> = {
    BTC: 65432, ETH: 3456, XRP: 0.62, SOL: 145, BNB: 580,
    ADA: 0.48, DOGE: 0.15, AVAX: 35.6, SHIB: 0.000025, LINK: 18.2, MATIC: 0.95, UNI: 11.4
  }
  const price = fallbackPrices[sym] ?? 1
  return {
    symbol: m.binanceSymbol, name: sym, logo: m.logo, rank: m.rank,
    cap: "—", vol: "—", supply: m.supply,
    price: fmtPrice(price), change24h: "+0.00%",
    rawPrice: price, rawChange: 0, rawVol: 0,
  }
})

// ─── Convert CoinGecko CoinData → LiveCoin ────────────────────────────────────
function fromCoinData(coin: CoinData): LiveCoin | null {
  const sym = coin.symbol.toUpperCase()
  const meta = COIN_META[sym]
  if (!meta) return null
  const rawPrice  = coin.currentPrice ?? 0
  const rawChange = coin.priceChangePercentage24h ?? 0
  const rawVol    = coin.volume24h ?? 0
  return {
    symbol:   meta.binanceSymbol,
    name:     coin.name,
    logo:     meta.logo,
    rank:     coin.marketCapRank ?? meta.rank,
    supply:   meta.supply,
    cap:      fmtCap(coin.marketCap),
    vol:      fmtVol(rawVol),
    price:    fmtPrice(rawPrice),
    change24h: fmtChange(rawChange),
    rawPrice, rawChange, rawVol,
  }
}

const POLL_INTERVAL_S = 30   // poll CoinGecko every 30 seconds

// ─── The Hook ─────────────────────────────────────────────────────────────────
export function useLiveMarketData() {
  const [coins, setCoins]         = useState<LiveCoin[]>(FALLBACK)
  const [isLive, setIsLive]       = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [nextRefresh, setNextRefresh] = useState(POLL_INTERVAL_S)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const countdownRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined)

  const fetchCoins = useCallback(async () => {
    try {
      // Call our own Next.js API route — no CORS issues, uses CoinGecko service
      const res = await fetch("/api/coins?perPage=250")
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const json = await res.json()
      if (!json.success || !Array.isArray(json.data)) throw new Error("Bad payload")

      const liveCoins: LiveCoin[] = []
      for (const sym of TRACK_SYMBOLS) {
        const meta = COIN_META[sym]
        const coinData: CoinData | undefined = json.data.find(
          (c: CoinData) => c.symbol.toUpperCase() === sym
        )
        if (coinData) {
          const live = fromCoinData(coinData)
          if (live) liveCoins.push(live)
        } else {
          // keep fallback for this symbol
          const fb = FALLBACK.find(f => f.symbol === meta.binanceSymbol)
          if (fb) liveCoins.push(fb)
        }
      }

      if (liveCoins.length > 0) {
        setCoins(liveCoins)
        setIsLive(true)
      }
    } catch (err) {
      console.warn("[LiveMarket] CoinGecko fetch failed, keeping previous data:", err)
      setIsLive(false)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // ── Schedule repeating polls ───────────────────────────────────────────────
  const scheduleNext = useCallback(() => {
    // Reset countdown
    setNextRefresh(POLL_INTERVAL_S)
    clearInterval(countdownRef.current)
    countdownRef.current = setInterval(() => {
      setNextRefresh(s => {
        if (s <= 1) {
          clearInterval(countdownRef.current)
          return 0
        }
        return s - 1
      })
    }, 1000)

    // Schedule the actual fetch
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(async () => {
      await fetchCoins()
      scheduleNext()
    }, POLL_INTERVAL_S * 1000)
  }, [fetchCoins])

  useEffect(() => {
    // Fetch immediately on mount
    fetchCoins().then(() => scheduleNext())

    return () => {
      clearTimeout(timerRef.current)
      clearInterval(countdownRef.current)
    }
  }, [fetchCoins, scheduleNext])

  return { liveCoins: coins, isLive, isLoading, nextRefresh }
}
