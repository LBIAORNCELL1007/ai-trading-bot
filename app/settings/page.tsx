'use client'

import Link from 'next/link'
import { TradingPairSelector } from '@/components/trading-pair-selector'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowLeft } from 'lucide-react'

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-background/80">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-12">
          <Link href="/crypto" className="flex items-center gap-2 mb-6 text-primary hover:text-primary/80 transition-colors">
            <ArrowLeft className="w-4 h-4" />
            Back to Coins
          </Link>

          <h1 className="text-4xl font-bold mb-3 text-balance">
            Settings
          </h1>
          <p className="text-lg text-muted-foreground text-balance">
            Manage your cryptocurrency tracking preferences
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Settings */}
          <div className="lg:col-span-2 space-y-6">
            {/* Trading Pair Selector */}
            <TradingPairSelector />

            {/* Display Preferences */}
            <Card>
              <CardHeader>
                <CardTitle>Display Preferences</CardTitle>
                <CardDescription>
                  Customize how you view cryptocurrency data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="text-sm font-semibold mb-3 block">Price Chart Type</label>
                  <div className="flex gap-3">
                    <Button variant="outline" className="flex-1 bg-transparent">
                      Area Chart
                    </Button>
                    <Button variant="outline" className="flex-1 bg-transparent">
                      Line Chart
                    </Button>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-semibold mb-3 block">Currency</label>
                  <select className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground">
                    <option>USD (United States Dollar)</option>
                    <option>EUR (Euro)</option>
                    <option>GBP (British Pound)</option>
                    <option>JPY (Japanese Yen)</option>
                    <option>INR (Indian Rupee)</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm font-semibold mb-3 block">Default Time Range</label>
                  <select className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground">
                    <option>1D (1 Day)</option>
                    <option>7D (7 Days)</option>
                    <option>1M (1 Month)</option>
                    <option>3M (3 Months)</option>
                    <option>1Y (1 Year)</option>
                    <option>All Time</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            {/* Data Sources */}
            <Card>
              <CardHeader>
                <CardTitle>Data Sources</CardTitle>
                <CardDescription>
                  Information about where your data comes from
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start justify-between p-4 bg-muted/30 rounded-lg">
                  <div>
                    <p className="font-semibold text-sm">CoinGecko API</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Real-time cryptocurrency market data for over 10,000 coins
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse inline-block"></div>
                    <p className="text-xs text-green-600 mt-1">Connected</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar Info */}
          <div className="space-y-6">
            {/* About Section */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">About This Site</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm text-muted-foreground">
                <div>
                  <p className="font-semibold text-foreground mb-1">Live Price Charts</p>
                  <p>Real-time cryptocurrency price data with interactive charts across multiple timeframes.</p>
                </div>

                <div>
                  <p className="font-semibold text-foreground mb-1">Complete Market Data</p>
                  <p>View market cap, trading volume, supply information, and popularity rankings for all coins.</p>
                </div>

                <div>
                  <p className="font-semibold text-foreground mb-1">Comprehensive Coverage</p>
                  <p>Browse and track over 10,000 cryptocurrencies including Bitcoin, Ethereum, altcoins, and memecoins.</p>
                </div>
              </CardContent>
            </Card>

            {/* Quick Links */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Links</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Link
                  href="/crypto"
                  className="block px-4 py-2 rounded-lg hover:bg-muted transition-colors text-sm font-medium text-primary"
                >
                  Browse Coins
                </Link>
                <a
                  href="https://www.coingecko.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block px-4 py-2 rounded-lg hover:bg-muted transition-colors text-sm font-medium text-primary"
                >
                  CoinGecko
                </a>
              </CardContent>
            </Card>

            {/* Stats Card */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Data Coverage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Cryptocurrencies</p>
                  <p className="text-2xl font-bold">10,000+</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Market Pairs</p>
                  <p className="text-2xl font-bold">Unlimited</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Update Frequency</p>
                  <p className="text-sm font-semibold">Real-time</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
