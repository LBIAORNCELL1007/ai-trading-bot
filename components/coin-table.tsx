'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowUpRight, ArrowDownRight } from 'lucide-react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import type { CoinData } from '@/types/crypto'

interface CoinTableProps {
  coins: CoinData[]
  loading?: boolean
}

export function CoinTable({ coins, loading }: CoinTableProps) {
  const [sortBy, setSortBy] = useState<'marketCap' | 'priceChangePercentage24h'>('marketCap')

  const sortedCoins = [...coins].sort((a, b) => {
    if (sortBy === 'marketCap') {
      return (b.marketCap || 0) - (a.marketCap || 0)
    }
    return (b.priceChangePercentage24h || 0) - (a.priceChangePercentage24h || 0)
  })

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A'
    if (price < 0.01) return `$${price.toFixed(6)}`
    return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }

  const formatMarketCap = (marketCap: number | null) => {
    if (marketCap === null) return 'N/A'
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(1)}B`
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(1)}M`
    return `$${marketCap.toLocaleString()}`
  }

  const formatVolume = (volume: number | null) => {
    if (volume === null) return 'N/A'
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(1)}B`
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(1)}M`
    return `$${volume.toLocaleString()}`
  }

  const formatSupply = (supply: number | null) => {
    if (supply === null) return 'N/A'
    if (supply >= 1e9) return `${(supply / 1e9).toFixed(1)}B`
    if (supply >= 1e6) return `${(supply / 1e6).toFixed(1)}M`
    return supply.toLocaleString()
  }

  const getPriceChangeColor = (change: number | null) => {
    if (change === null) return 'text-muted-foreground'
    if (change > 0) return 'text-green-500'
    if (change < 0) return 'text-red-500'
    return 'text-muted-foreground'
  }

  if (loading) {
    return (
      <div className="space-y-2 animate-pulse">
        {[...Array(10)].map((_, i) => (
          <div key={i} className="h-12 bg-muted rounded" />
        ))}
      </div>
    )
  }

  return (
    <div className="w-full overflow-x-auto rounded-lg border border-border">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/50 hover:bg-muted/50">
            <TableHead className="w-12">#</TableHead>
            <TableHead className="min-w-40">Coin</TableHead>
            <TableHead className="text-right">Price</TableHead>
            <TableHead className="text-right">24h Change</TableHead>
            <TableHead className="text-right">7d Change</TableHead>
            <TableHead
              className="text-right cursor-pointer hover:text-foreground"
              onClick={() => setSortBy(sortBy === 'marketCap' ? 'priceChangePercentage24h' : 'marketCap')}
            >
              Market Cap
            </TableHead>
            <TableHead className="text-right">Volume 24h</TableHead>
            <TableHead className="text-right">Circulating Supply</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedCoins.map((coin, index) => (
            <TableRow
              key={coin.id}
              className="hover:bg-muted/50 transition-colors border-b border-border/50"
            >
              <TableCell className="text-xs font-medium text-muted-foreground">{index + 1}</TableCell>
              <TableCell>
                <Link
                  href={`/coins/${coin.id}`}
                  className="flex items-center gap-3 font-medium hover:text-primary transition-colors"
                >
                  <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center flex-shrink-0 overflow-hidden">
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
                    <p className="text-sm truncate">{coin.name}</p>
                    <p className="text-xs text-muted-foreground">{coin.symbol}</p>
                  </div>
                </Link>
              </TableCell>
              <TableCell className="text-right font-semibold">{formatPrice(coin.currentPrice)}</TableCell>
              <TableCell className={`text-right ${getPriceChangeColor(coin.priceChangePercentage24h)}`}>
                <div className="flex items-center justify-end gap-1">
                  {coin.priceChangePercentage24h !== null && coin.priceChangePercentage24h > 0 ? (
                    <ArrowUpRight className="w-4 h-4" />
                  ) : coin.priceChangePercentage24h !== null && coin.priceChangePercentage24h < 0 ? (
                    <ArrowDownRight className="w-4 h-4" />
                  ) : null}
                  {coin.priceChangePercentage24h?.toFixed(2) || 'N/A'}%
                </div>
              </TableCell>
              <TableCell className={`text-right ${getPriceChangeColor(coin.priceChangePercentage7d)}`}>
                {coin.priceChangePercentage7d?.toFixed(2) || 'N/A'}%
              </TableCell>
              <TableCell className="text-right text-sm">{formatMarketCap(coin.marketCap)}</TableCell>
              <TableCell className="text-right text-sm">{formatVolume(coin.volume24h)}</TableCell>
              <TableCell className="text-right text-sm">{formatSupply(coin.circulatingSupply)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
