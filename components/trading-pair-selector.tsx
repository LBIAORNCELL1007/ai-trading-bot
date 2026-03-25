'use client'

import React from "react"

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Search, Loader } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useOptimizedSearch } from '@/hooks/use-optimized-search'

export function TradingPairSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const { query, results: searchResults, cacheReady, search, loading: searchLoading } = useOptimizedSearch()

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value

    if (value.trim().length >= 1) {
      search(value)
      setIsOpen(true)
    } else {
      setIsOpen(false)
    }
  }

  const handleSelectCoin = () => {
    search('')
    setIsOpen(false)
  }

  return (
    <Card className="bg-[#1A1A1A] border-[#333]/50 shadow-xl backdrop-blur-sm overflow-hidden">
      <CardHeader>
        <CardTitle>Trading Pair Selector</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Search and select a cryptocurrency to view its price chart and market data.
        </p>

        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground pointer-events-none" />
          <Input
            disabled={!cacheReady}
            placeholder={cacheReady ? "Search: Bitcoin, ETH, DOGE, SHIB..." : "Loading coin data..."}
            value={query}
            onChange={handleSearch}
            onFocus={() => {
              if (query.trim().length >= 1 && cacheReady) setIsOpen(true)
            }}
            onBlur={() => {
              setTimeout(() => setIsOpen(false), 200)
            }}
            className="pl-12 pr-10 h-12 bg-[#121212] border-2 border-[#1DB954]/20 focus:border-[#1DB954] focus:bg-[#1DB954]/5 transition-all text-foreground placeholder:text-muted-foreground/60"
          />
          {!cacheReady && (
            <Loader className="absolute right-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-primary animate-spin" />
          )}
        </div>

        {isOpen && query.trim().length >= 1 && cacheReady && (
          <div className="border border-[#333]/50 rounded-lg overflow-hidden bg-[#121212] shadow-2xl">
            {searchResults.length > 0 ? (
              <div className="max-h-64 overflow-y-auto">
                {searchResults.slice(0, 15).map((coin) => (
                  <Link
                    key={coin.id}
                    href={`/trade/${coin.symbol.toUpperCase()}`}
                    onClick={handleSelectCoin}
                    className="flex items-center gap-3 p-3 hover:bg-[#1DB954]/5 transition-colors border-b border-[#333]/30 last:border-0 cursor-pointer group"
                  >
                    <img
                      src={coin.image || '/placeholder.svg'}
                      alt={coin.name}
                      className="w-8 h-8 rounded-full flex-shrink-0"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm">{coin.name}</p>
                      <p className="text-xs text-muted-foreground uppercase">{coin.symbol}</p>
                    </div>
                    {coin.current_price && (
                      <div className="text-right flex-shrink-0">
                        <p className="text-sm font-semibold">
                          ${coin.current_price < 0.01 ? coin.current_price.toFixed(6) : coin.current_price.toFixed(2)}
                        </p>
                      </div>
                    )}
                  </Link>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center text-muted-foreground text-sm">
                No coins found for "{query}"
              </div>
            )}
          </div>
        )}

        <div className="bg-[#1DB954]/5 border border-[#1DB954]/20 rounded-xl p-5 text-sm">
          <p className="text-[#1DB954] font-medium leading-relaxed">
            <span className="mr-2">💡</span>
            <span className="opacity-80">Tip: Start typing any coin name or symbol. Results include Bitcoin, Ethereum, Cardano, Dogecoin, Shiba Inu, and thousands more including altcoins and memecoins.</span>
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
