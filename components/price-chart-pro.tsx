'use client'

import { useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { PriceHistory } from '@/types/crypto'

interface PriceChartProProps {
  data: PriceHistory[]
  coin: {
    name: string
    symbol: string
    image?: string
  }
  currentPrice: number
  loading?: boolean
  selectedRange?: TimeRange
  onTimeRangeChange?: (range: TimeRange) => void
}

type TimeRange = '1d' | '7d' | '1m' | '3m' | '1y' | 'ytd'

const TIME_RANGES: { label: string; value: TimeRange }[] = [
  { label: '1D', value: '1d' },
  { label: '7D', value: '7d' },
  { label: '1M', value: '1m' },
  { label: '3M', value: '3m' },
  { label: '1Y', value: '1y' },
  { label: 'YTD', value: 'ytd' },
]

export function PriceChartPro({ data, coin, currentPrice, loading, selectedRange: externalRange, onTimeRangeChange }: PriceChartProProps) {
  const [selectedRange, setSelectedRange] = useState<TimeRange>(externalRange || '1d')

  const handleRangeChange = (range: TimeRange) => {
    setSelectedRange(range)
    onTimeRangeChange?.(range)
  }

  // Show all data without slicing to ensure full historical view is available
  const chartData = data.map((point) => ({
    timestamp: new Date(point.timestamp).getTime(),
    time: new Date(point.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: selectedRange === '1d' ? '2-digit' : undefined,
      minute: selectedRange === '1d' ? '2-digit' : undefined,
    }),
    price: point.price,
  }))

  if (chartData.length === 0) {
    return (
      <Card className="bg-background border border-border">
        <CardContent className="p-8">
          <div className="h-96 flex items-center justify-center text-muted-foreground">
            No price data available
          </div>
        </CardContent>
      </Card>
    )
  }

  const minPrice = Math.min(...chartData.map((d) => d.price))
  const maxPrice = Math.max(...chartData.map((d) => d.price))
  const startPrice = chartData[0].price
  const endPrice = chartData[chartData.length - 1].price
  const change = endPrice - startPrice
  const changePercent = (change / startPrice) * 100
  const isPositive = change >= 0

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload[0]) {
      return (
        <div className="bg-background/95 border border-border rounded-lg p-3 shadow-xl">
          <p className="font-bold text-foreground">
            ${payload[0].value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}
          </p>
          <p className="text-xs text-muted-foreground mt-1">{payload[0].payload.time}</p>
        </div>
      )
    }
    return null
  }

  return (
    <Card className="bg-gradient-to-br from-background to-background/80 border-border overflow-hidden">
      <CardHeader className="pb-4 border-b border-border">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-4">
            {coin.image && (
              <img
                src={coin.image || "/placeholder.svg"}
                alt={coin.name}
                className="w-12 h-12 rounded-full"
                onError={(e) => {
                  e.currentTarget.style.display = 'none'
                }}
              />
            )}
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h2 className="text-2xl font-bold text-foreground">
                  {coin.name} ({coin.symbol.toUpperCase()})
                </h2>
                <Badge className="bg-yellow-500/20 text-yellow-600 border-0">HOT</Badge>
              </div>
              <div className="flex items-baseline gap-2 mt-2">
                <p className="text-3xl font-bold text-foreground">
                  ${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}
                </p>
                <p className={`text-lg font-semibold flex items-center gap-1 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
                  {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
                </p>
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                {coin.symbol.toUpperCase()} to USD: 1 {coin.symbol.toUpperCase()} equals ${currentPrice.toFixed(2)} USD
              </p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6">
        {/* Time Range Buttons */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {TIME_RANGES.map((range) => (
            <Button
              key={range.value}
              onClick={() => handleRangeChange(range.value)}
              variant={selectedRange === range.value ? 'default' : 'outline'}
              size="sm"
              className={`whitespace-nowrap ${selectedRange === range.value
                ? 'bg-yellow-500 text-black hover:bg-yellow-600'
                : 'border-border hover:bg-muted'
                }`}
            >
              {range.label}
            </Button>
          ))}
        </div>

        {/* Chart */}
        <div className="bg-background/50 rounded-lg p-4 border border-border/50">
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#eab308" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                vertical={false}
                opacity={0.3}
              />
              <XAxis
                dataKey="time"
                stroke="var(--muted-foreground)"
                style={{ fontSize: '12px' }}
                tick={{ fill: '#000000' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="var(--muted-foreground)"
                style={{ fontSize: '12px' }}
                tick={{ fill: '#000000' }}
                tickLine={false}
                axisLine={false}
                label={{ value: 'USD', position: 'right', offset: 10, fill: '#000000' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="price"
                stroke="#eab308"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorPrice)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-muted/30 rounded-lg p-4 border border-border/30">
            <p className="text-xs text-muted-foreground mb-1">24H High</p>
            <p className="text-lg font-semibold text-foreground">
              ${maxPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-4 border border-border/30">
            <p className="text-xs text-muted-foreground mb-1">24H Low</p>
            <p className="text-lg font-semibold text-foreground">
              ${minPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-4 border border-border/30">
            <p className="text-xs text-muted-foreground mb-1">Change</p>
            <p className={`text-lg font-semibold ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
              {isPositive ? '+' : ''}${change.toFixed(2)}
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-4 border border-border/30">
            <p className="text-xs text-muted-foreground mb-1">% Change</p>
            <p className={`text-lg font-semibold ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
              {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
