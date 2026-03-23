'use client'

import { useState, useEffect, useCallback } from 'react'
import { loadCoinCache, searchCoinsLocally, isCacheReady } from '@/lib/coin-cache'

interface SearchResult {
  id: string
  name: string
  symbol: string
  image: string
  current_price?: number
}

export function useOptimizedSearch() {
  const [cacheReady, setCacheReady] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)

  // Initialize cache on component mount
  useEffect(() => {
    let isMounted = true

    const initCache = async () => {
      if (!isMounted) return
      setLoading(true)
      await loadCoinCache()
      if (isMounted) {
        setCacheReady(true)
        setLoading(false)
      }
    }

    if (!isCacheReady()) {
      initCache()
    } else {
      setCacheReady(true)
    }

    return () => {
      isMounted = false
    }
  }, [])

  const search = useCallback((q: string) => {
    setQuery(q)
    if (!q.trim()) {
      setResults([])
      return
    }

    // Use local search instead of API
    const localResults = searchCoinsLocally(q)
    setResults(localResults)
  }, [])

  return {
    query,
    results,
    loading,
    cacheReady,
    search,
  }
}
