'use client'

import Link from "next/link"
import React from "react"
import { useState } from 'react'
import { TrendingUp, RotateCcw } from 'lucide-react'
import { useCoins, useTrendingCoins } from '@/hooks/use-crypto-data'
import { CoinTable } from '@/components/coin-table'
import { CryptoSearch } from '@/components/crypto-search'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

export default function CryptoPage() {
  const [page, setPage] = useState(1)
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [searchResults, setSearchResults] = useState([])
  const [searchLoading, setSearchLoading] = useState(false)

  // Auto-refresh enabled by default - prices update every 10 seconds
  const { coins, loading: coinsLoading, lastUpdated } = useCoins({ page, perPage: 100 })
  const { coins: trendingCoins, loading: trendingLoading } = useTrendingCoins()

  return (
    <div className="min-h-screen bg-[#121212] text-foreground">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <Link href="/dashboard">
              <Button variant="ghost" className="hover:bg-[#1DB954]/10 text-muted-foreground hover:text-[#1DB954] gap-2">
                <RotateCcw className="w-4 h-4" />
                Back to Market
              </Button>
            </Link>
          </div>
          <div className="flex items-center gap-3 mb-3">
            <h1 className="text-4xl font-bold text-balance">
              Cryptocurrency Market Data
            </h1>
            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
              <span className="w-2 h-2 rounded-full bg-green-500 mr-1 animate-pulse" />
              LIVE
            </Badge>
          </div>
          <p className="text-lg text-muted-foreground text-balance mb-2">
            Track real-time prices, market cap, volume, and supply data for thousands of cryptocurrencies
          </p>
          {lastUpdated && (
            <p className="text-sm text-muted-foreground">
              Last updated: {lastUpdated.toLocaleTimeString()} • Auto-refreshing every 10s
            </p>
          )}

          {/* Featured Search Bar */}
          <div className="mt-6">
            <CryptoSearch />
          </div>
        </div>

        {/* Trending Section */}
        {!showSearchResults && (
          <div className="mb-12">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="h-5 w-5 text-primary" />
              <h2 className="text-2xl font-semibold">Trending Coins</h2>
            </div>

            {trendingLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[...Array(6)].map((_, i) => (
                  <Skeleton key={i} className="h-24 rounded-lg" />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {trendingCoins.slice(0, 6).map((coin) => (
                  <Link
                    key={coin.id}
                    href={`/trade/${coin.symbol.toUpperCase()}`}
                    className="group"
                  >
                    <Card className="bg-[#1A1A1A] border-[#333]/50 hover:border-[#1DB954]/50 transition-colors h-full">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <div className="w-10 h-10 rounded-full bg-muted flex-shrink-0 overflow-hidden">
                              {coin.image && (
                                <img
                                  src={coin.image || "/placeholder.svg"}
                                  alt={coin.name}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    e.currentTarget.style.display = 'none'
                                  }}
                                />
                              )}
                            </div>
                            <div className="min-w-0">
                              <p className="font-semibold text-sm truncate">{coin.name}</p>
                              <p className="text-xs text-muted-foreground">{coin.symbol}</p>
                            </div>
                          </div>
                          <Badge variant="outline">{coin.marketCapRank}</Badge>
                        </div>
                        <p className="text-2xl font-bold mb-2">
                          ${coin.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}
                        </p>
                        <p className={`text-sm font-semibold ${(coin.priceChangePercentage24h || 0) > 0 ? 'text-green-500' : 'text-red-500'
                          }`}>
                          {(coin.priceChangePercentage24h || 0) > 0 ? '+' : ''}
                          {coin.priceChangePercentage24h?.toFixed(2) || 'N/A'}% (24h)
                        </p>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Main Coin List */}
        <div>
          <h2 className="text-2xl font-semibold mb-6">
            {showSearchResults && searchResults.length > 0
              ? `Search Results: ${searchResults.length} coins`
              : 'All Cryptocurrencies'}
          </h2>

          {searchLoading || coinsLoading ? (
            <div className="space-y-2">
              {[...Array(10)].map((_, i) => (
                <Skeleton key={i} className="h-12 rounded-lg" />
              ))}
            </div>
          ) : (
            <>
              <CoinTable
                coins={coins}
                loading={coinsLoading}
              />

              {/* Pagination */}
              {!showSearchResults && (
                <div className="flex justify-center gap-2 mt-8">
                  <Button
                    variant="outline"
                    onClick={() => setPage(Math.max(1, page - 1))}
                    disabled={page === 1}
                  >
                    Previous
                  </Button>
                  <div className="flex items-center px-4 text-sm text-muted-foreground">
                    Page {page}
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => setPage(page + 1)}
                  >
                    Next
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
