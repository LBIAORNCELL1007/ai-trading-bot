"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { AlertCircle, CheckCircle, Info } from "lucide-react"

interface StrategyConfig {
  symbol: string
  strategy: string
  timeframe: string
  riskPerTrade: number
  rewardMultiplier: number
  maxPositions: number
  paperTrading: boolean
  gridCount?: number
  gridPercentage?: number
}

interface StrategyInfo {
  name: string
  description: string
  best_for: string
  indicators: string[]
  pros: string[]
  cons: string[]
}

const STRATEGIES: Record<string, StrategyInfo> = {
  "MA_CROSSOVER": {
    name: "Moving Average Crossover",
    description: "Trend-following strategy using SMA/EMA crossovers (20/50/200)",
    best_for: "Trending markets with clear direction",
    indicators: ["SMA 20", "SMA 50", "SMA 200"],
    pros: [
      "Simple and reliable",
      "Works well in strong trends",
      "Easy to understand and implement"
    ],
    cons: [
      "Whipsaws in ranging markets",
      "Lags price action",
      "May miss quick reversals"
    ]
  },
  "MACD": {
    name: "MACD Strategy",
    description: "Momentum-based strategy using MACD crossovers and histogram",
    best_for: "Momentum trading and trend changes",
    indicators: ["MACD", "Signal Line", "Histogram"],
    pros: [
      "Good for momentum detection",
      "Clear entry/exit signals",
      "Works in multiple timeframes"
    ],
    cons: [
      "Can be delayed",
      "False signals in ranging markets",
      "Requires filter for confirmation"
    ]
  },
  "MEAN_REVERSION": {
    name: "Mean Reversion",
    description: "Oversold/Overbought strategy using Bollinger Bands and RSI",
    best_for: "Range-bound and ranging markets",
    indicators: ["Bollinger Bands", "RSI 14"],
    pros: [
      "High win rate in ranging markets",
      "Clear support/resistance levels",
      "Good risk/reward setup"
    ],
    cons: [
      "Works poorly in trends",
      "Can get whipped around",
      "Requires tight stops"
    ]
  },
  "GRID": {
    name: "Grid Trading",
    description: "Places buy/sell orders at fixed intervals to profit from oscillations",
    best_for: "Volatile and range-bound markets",
    indicators: ["Price Levels", "Volume"],
    pros: [
      "Profits from volatility",
      "Automated position building",
      "Works in choppy markets"
    ],
    cons: [
      "Large drawdown potential",
      "Requires adequate capital",
      "Can hit stop loss frequently"
    ]
  },
  "MULTI_SIGNAL": {
    name: "Multi-Signal Ensemble",
    description: "Voting system combining all strategies for robust signals",
    best_for: "All market conditions",
    indicators: ["All of above"],
    pros: [
      "Robust across all conditions",
      "Reduced false signals",
      "Diversified approach"
    ],
    cons: [
      "More complex to tune",
      "Moderate trade frequency",
      "Requires more capital"
    ]
  }
}

const SYMBOLS = [
  "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLusdt",
  "DOGEUSDT", "MATICUSDT", "LTCUSDT", "XRPUSDT", "AAVEUSDT"
]

const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

interface StrategyConfiguratorProps {
  onSave?: (config: StrategyConfig) => void
}

export default function StrategyConfigurator({ onSave }: StrategyConfiguratorProps) {
  const [config, setConfig] = useState<StrategyConfig>({
    symbol: "BTCUSDT",
    strategy: "MULTI_SIGNAL",
    timeframe: "1h",
    riskPerTrade: 2,
    rewardMultiplier: 3,
    maxPositions: 5,
    paperTrading: true,
    gridCount: 10,
    gridPercentage: 0.02,
  })

  const selectedStrategy = STRATEGIES[config.strategy]

  return (
    <div className="space-y-6">
      {/* Strategy Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Trading Strategy</CardTitle>
          <CardDescription>Choose a strategy based on market conditions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(STRATEGIES).map(([key, strategy]) => (
              <button
                key={key}
                onClick={() => setConfig({ ...config, strategy: key })}
                className={`p-4 rounded-lg border-2 text-left transition-all ${
                  config.strategy === key
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <h3 className="font-semibold mb-1">{strategy.name}</h3>
                <p className="text-sm text-muted-foreground mb-2">{strategy.description}</p>
                <Badge variant="outline" className="text-xs">
                  {strategy.best_for}
                </Badge>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Strategy Details */}
      {selectedStrategy && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="w-4 h-4" />
              {selectedStrategy.name} Details
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Key Indicators</h4>
                <div className="flex flex-wrap gap-2">
                  {selectedStrategy.indicators.map((ind) => (
                    <Badge key={ind} variant="secondary">
                      {ind}
                    </Badge>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Best Used For</h4>
                <p className="text-sm text-muted-foreground">{selectedStrategy.best_for}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2 text-green-600 flex items-center">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  Pros
                </h4>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  {selectedStrategy.pros.map((pro, i) => (
                    <li key={i}>• {pro}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2 text-red-600 flex items-center">
                  <AlertCircle className="w-4 h-4 mr-1" />
                  Cons
                </h4>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  {selectedStrategy.cons.map((con, i) => (
                    <li key={i}>• {con}</li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration Parameters</CardTitle>
          <CardDescription>Fine-tune the trading parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Symbol Selection */}
            <div className="space-y-2">
              <Label htmlFor="symbol">Trading Pair</Label>
              <Select value={config.symbol} onValueChange={(value) => setConfig({ ...config, symbol: value })}>
                <SelectTrigger id="symbol">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SYMBOLS.map((sym) => (
                    <SelectItem key={sym} value={sym}>
                      {sym}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Timeframe Selection */}
            <div className="space-y-2">
              <Label htmlFor="timeframe">Timeframe</Label>
              <Select value={config.timeframe} onValueChange={(value) => setConfig({ ...config, timeframe: value })}>
                <SelectTrigger id="timeframe">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIMEFRAMES.map((tf) => (
                    <SelectItem key={tf} value={tf}>
                      {tf}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Risk Per Trade */}
            <div className="space-y-2">
              <Label htmlFor="risk">Risk Per Trade (%)</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="risk"
                  type="number"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={config.riskPerTrade}
                  onChange={(e) => setConfig({ ...config, riskPerTrade: parseFloat(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm text-muted-foreground">{config.riskPerTrade}%</span>
              </div>
              <p className="text-xs text-muted-foreground">Recommended: 1-3%</p>
            </div>

            {/* Reward Multiplier */}
            <div className="space-y-2">
              <Label htmlFor="reward">Reward Multiplier</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="reward"
                  type="number"
                  min="1"
                  max="10"
                  step="0.5"
                  value={config.rewardMultiplier}
                  onChange={(e) => setConfig({ ...config, rewardMultiplier: parseFloat(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm text-muted-foreground">:{config.rewardMultiplier}</span>
              </div>
              <p className="text-xs text-muted-foreground">Risk/Reward ratio (1:{config.rewardMultiplier})</p>
            </div>

            {/* Max Positions */}
            <div className="space-y-2">
              <Label htmlFor="positions">Max Open Positions</Label>
              <Input
                id="positions"
                type="number"
                min="1"
                max="20"
                value={config.maxPositions}
                onChange={(e) => setConfig({ ...config, maxPositions: parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Maximum concurrent open trades</p>
            </div>

            {/* Grid Settings (conditional) */}
            {(config.strategy === "GRID" || config.strategy === "MULTI_SIGNAL") && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="gridCount">Grid Count</Label>
                  <Input
                    id="gridCount"
                    type="number"
                    min="5"
                    max="50"
                    value={config.gridCount || 10}
                    onChange={(e) => setConfig({ ...config, gridCount: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-muted-foreground">Number of grid levels</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gridPercent">Grid Spacing (%)</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      id="gridPercent"
                      type="number"
                      min="0.1"
                      max="5"
                      step="0.1"
                      value={config.gridPercentage || 0.02}
                      onChange={(e) => setConfig({ ...config, gridPercentage: parseFloat(e.target.value) })}
                      className="flex-1"
                    />
                    <span className="text-sm text-muted-foreground">{((config.gridPercentage || 0.02) * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Trading Mode */}
          <div className="pt-4 border-t">
            <div className="flex items-center gap-2 mb-4">
              <input
                type="checkbox"
                id="paperTrade"
                checked={config.paperTrading}
                onChange={(e) => setConfig({ ...config, paperTrading: e.target.checked })}
                className="rounded"
              />
              <Label htmlFor="paperTrade" className="cursor-pointer">
                {config.paperTrading ? "Paper Trading (Demo Mode)" : "Live Trading with Real Money"}
              </Label>
            </div>
            <p className="text-sm text-muted-foreground">
              {config.paperTrading
                ? "Trading with simulated money - no real funds at risk"
                : "Trading with real Binance account - use with caution"}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="pt-4 border-t flex gap-2">
            <Button variant="outline" className="flex-1 bg-transparent">
              Reset to Defaults
            </Button>
            <Button
              onClick={() => onSave?.(config)}
              className="flex-1"
            >
              Save Configuration
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Quick Start Presets */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start Presets</CardTitle>
          <CardDescription>Load pre-configured settings for different market conditions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Button
              variant="outline"
              onClick={() => setConfig({
                ...config,
                strategy: "MA_CROSSOVER",
                riskPerTrade: 2,
                rewardMultiplier: 2.5,
                maxPositions: 3,
              })}
            >
              Trending Markets
            </Button>
            <Button
              variant="outline"
              onClick={() => setConfig({
                ...config,
                strategy: "MEAN_REVERSION",
                riskPerTrade: 1.5,
                rewardMultiplier: 2,
                maxPositions: 5,
              })}
            >
              Ranging Markets
            </Button>
            <Button
              variant="outline"
              onClick={() => setConfig({
                ...config,
                strategy: "MULTI_SIGNAL",
                riskPerTrade: 1,
                rewardMultiplier: 3,
                maxPositions: 5,
              })}
            >
              Conservative
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
