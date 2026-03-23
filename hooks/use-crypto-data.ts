'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { CoinData, PriceHistory } from '@/types/crypto'

interface UseCoinsOptions {
  page?: number
  perPage?: number
  autoRefresh?: boolean // Enable auto-refresh for live prices
  refreshInterval?: number // Refresh interval in ms (default 10s)
}

export function useCoins(options: UseCoinsOptions = {}) {
  const [coins, setCoins] = useState<CoinData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const isMountedRef = useRef(false)

  const { page = 1, perPage = 50, autoRefresh = true, refreshInterval = 10000 } = options

  useEffect(() => {
    isMountedRef.current = true

    const fetchCoins = async (showLoading = true) => {
      try {
        if (showLoading && isMountedRef.current) setLoading(true)
        const response = await fetch(`/api/coins?page=${page}&perPage=${perPage}`)
        if (!response.ok) throw new Error('Failed to fetch coins')
        const data = await response.json()
        if (isMountedRef.current) {
          setCoins(data.data)
          setError(null)
          setLastUpdated(new Date())
        }
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setCoins([])
        }
      } finally {
        if (showLoading && isMountedRef.current) setLoading(false)
      }
    }

    fetchCoins(true)

    // Auto-refresh prices
    let interval: NodeJS.Timeout | null = null
    if (autoRefresh) {
      interval = setInterval(() => {
        if (isMountedRef.current) fetchCoins(false)
      }, refreshInterval)
    }

    return () => {
      isMountedRef.current = false
      if (interval) clearInterval(interval)
    }
  }, [page, perPage, autoRefresh, refreshInterval])

  return { coins, loading, error, lastUpdated }
}

export function useSearchCoins() {
  const [results, setResults] = useState<Array<{ id: string; name: string; symbol: string; image: string }>>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const search = useCallback(async (query: string) => {
    if (!query.trim()) {
      setResults([])
      return
    }

    try {
      setLoading(true)
      const response = await fetch(`/api/coins/search?q=${encodeURIComponent(query)}`)
      if (!response.ok) throw new Error('Search failed')
      const data = await response.json()
      setResults(data.data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  return { results, loading, error, search }
}

export function useCoinDetail(coinId: string | null) {
  const [coin, setCoin] = useState<any>(null)
  const [loading, setLoading] = useState(!!coinId)
  const [error, setError] = useState<string | null>(null)
  const isMountedRef = useRef(false)

  useEffect(() => {
    isMountedRef.current = true

    if (!coinId) {
      if (isMountedRef.current) {
        setCoin(null)
        setLoading(false)
      }
      return
    }

    const fetchCoin = async () => {
      try {
        if (isMountedRef.current) setLoading(true)
        const response = await fetch(`/api/coins/${coinId}`)
        if (!response.ok) throw new Error('Failed to fetch coin details')
        const data = await response.json()
        if (isMountedRef.current) {
          setCoin(data.data)
          setError(null)
        }
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setCoin(null)
        }
      } finally {
        if (isMountedRef.current) setLoading(false)
      }
    }

    fetchCoin()

    return () => { isMountedRef.current = false }
  }, [coinId])

  return { coin, loading, error }
}

export function usePriceHistory(coinId: string | null, days: string | number = 'max') {
  const [history, setHistory] = useState<PriceHistory[]>([])
  const [loading, setLoading] = useState(!!coinId)
  const [error, setError] = useState<string | null>(null)
  const isMountedRef = useRef(false)

  // Convert time range to days
  const convertTimeRangeToDays = (timeRange: string | number): string | number => {
    if (typeof timeRange === 'number') return timeRange

    const today = new Date()
    const yearStart = new Date(today.getFullYear(), 0, 1)
    const daysSinceYearStart = Math.floor((today.getTime() - yearStart.getTime()) / (1000 * 60 * 60 * 24))

    switch (timeRange) {
      case '1d': return 1
      case '7d': return 7
      case '1m': return 30
      case '3m': return 90
      case '1y': return 365
      case 'ytd': return daysSinceYearStart
      default: return 'max'
    }
  }

  useEffect(() => {
    isMountedRef.current = true

    if (!coinId) {
      if (isMountedRef.current) {
        setHistory([])
        setLoading(false)
      }
      return
    }

    const fetchHistory = async () => {
      try {
        if (isMountedRef.current) setLoading(true)
        const convertedDays = convertTimeRangeToDays(days)
        const response = await fetch(`/api/coins/${coinId}/history?days=${convertedDays}`)
        if (!response.ok) throw new Error('Failed to fetch price history')
        const data = await response.json()
        if (isMountedRef.current) {
          setHistory(data.data)
          setError(null)
        }
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setHistory([])
        }
      } finally {
        if (isMountedRef.current) setLoading(false)
      }
    }

    fetchHistory()

    return () => { isMountedRef.current = false }
  }, [coinId, days])

  return { history, loading, error }
}

export function useTrendingCoins(autoRefresh = true, refreshInterval = 10000) {
  const [coins, setCoins] = useState<CoinData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const isMountedRef = useRef(false)

  useEffect(() => {
    isMountedRef.current = true

    const fetchTrending = async (showLoading = true) => {
      try {
        if (showLoading && isMountedRef.current) setLoading(true)
        const response = await fetch('/api/coins/trending')
        if (!response.ok) throw new Error('Failed to fetch trending coins')
        const data = await response.json()
        if (isMountedRef.current) {
          setCoins(data.data)
          setError(null)
        }
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setCoins([])
        }
      } finally {
        if (showLoading && isMountedRef.current) setLoading(false)
      }
    }

    fetchTrending(true)

    // Auto-refresh prices
    let interval: NodeJS.Timeout | null = null
    if (autoRefresh) {
      interval = setInterval(() => {
        if (isMountedRef.current) fetchTrending(false)
      }, refreshInterval)
    }

    return () => {
      isMountedRef.current = false
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  return { coins, loading, error }
}
