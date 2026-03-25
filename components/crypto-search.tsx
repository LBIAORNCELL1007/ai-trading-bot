'use client'

import React from "react"


import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { Search, X, Loader } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { useOptimizedSearch } from '@/hooks/use-optimized-search'

export function CryptoSearch() {
  const [isOpen, setIsOpen] = useState(false)
  const [isFocused, setIsFocused] = useState(false)
  const { query, results: searchResults, loading: searchLoading, cacheReady, search } = useOptimizedSearch()
  const containerRef = useRef<HTMLDivElement>(null)

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value

    if (value.trim().length >= 1) {
      search(value)
      setIsOpen(true)
    } else {
      setIsOpen(false)
    }
  }

  const handleClear = () => {
    search('')
    setIsOpen(false)
  }

  const handleSelectCoin = () => {
    search('')
    setIsOpen(false)
  }

  return (
    <div ref={containerRef} className="w-full max-w-2xl mx-auto relative">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground pointer-events-none" />
        <Input
          placeholder={cacheReady ? "Search coins: Bitcoin, ETH, DOGE, SOL..." : "Loading coin data..."}
          disabled={!cacheReady}
          value={query}
          onChange={handleSearch}
          onFocus={() => {
            setIsFocused(true)
            if (query.trim().length >= 1) setIsOpen(true)
          }}
          onBlur={() => setIsFocused(false)}
          className="pl-12 pr-10 h-12 text-base bg-[#1DB954]/5 border-2 border-[#1DB954]/20 hover:border-[#1DB954]/40 focus:border-[#1DB954] focus:bg-[#1DB954]/10 transition-all text-foreground placeholder:text-muted-foreground/60"
        />
        {query && (
          <button
            onClick={handleClear}
            className="absolute right-4 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        )}
        {!cacheReady && (
          <Loader className="absolute right-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-primary animate-spin" />
        )}
      </div>

      {/* Search Results Dropdown */}
      {isOpen && query.trim().length >= 1 && cacheReady && (
        <Card className="absolute top-full left-0 right-0 mt-2 z-50 shadow-lg">
          <CardContent className="p-0">
            {searchResults.length > 0 ? (
              <div className="max-h-96 overflow-y-auto">
                {searchResults
                  .filter((v, i, a) => a.findIndex(t => (t.id === v.id)) === i) // Deduplicate results
                  .slice(0, 10)
                  .map((result, index) => (
                    <Link
                      key={result.id}
                      href={`/coins/${result.id}`}
                      onClick={handleSelectCoin}
                      className="flex items-center gap-3 p-4 hover:bg-muted transition-colors border-b last:border-0 cursor-pointer"
                    >
                      <img
                        src={result.image || '/placeholder.svg'}
                        alt={result.name}
                        className="w-8 h-8 rounded-full flex-shrink-0"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                        }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-sm text-foreground">{result.name}</p>
                        <p className="text-xs text-muted-foreground uppercase">{result.symbol}</p>
                      </div>
                      {result.current_price && (
                        <div className="text-right flex-shrink-0">
                          <p className="font-semibold text-sm">
                            ${result.current_price < 0.01 ? result.current_price.toFixed(6) : result.current_price.toFixed(2)}
                          </p>
                        </div>
                      )}
                    </Link>
                  ))}
              </div>
            ) : (
              <div className="p-4 text-center text-muted-foreground text-sm">
                No coins found matching "{query}"
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
