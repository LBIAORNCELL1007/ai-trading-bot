"use client"

import React, { useState, useEffect } from "react"
import { SlidersHorizontal } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"

interface TradingConfig {
  symbol: string
  timeframe: string
  strategy: string
  riskPerTrade: number
  maxPositions: number
  stopLoss: number
  takeProfit: number
  trailingStop: number
  leverage: number
}

const TRADING_PAIRS = [
  'BTCUSDT',
  'ETHUSDT',
  'BNBUSDT',
  'ADAUSDT',
  'SOLUSDT',
  'XRPUSDT',
  'DOGEUSDT',
  'MATICUSDT',
  'DOTUSDT',
  'LTCUSDT',
  'AVAXUSDT',
  'LINKUSDT',
  'UNIUSDT',
  'ATOMUSDT',
  'ETCUSDT'
]

const defaultConfig: TradingConfig = {
  symbol: 'BTCUSDT',
  timeframe: '15m',
  strategy: 'MACD',
  riskPerTrade: 2,
  maxPositions: 3,
  stopLoss: 2.0,
  takeProfit: 4.0,
  trailingStop: 1.0,
  leverage: 1
}

export function GlobalTradeConfig() {
  const [open, setOpen] = useState(false)
  const [config, setConfig] = useState<TradingConfig>(defaultConfig)
  const { toast } = useToast()

  useEffect(() => {
    if (open) {
      const savedConfig = localStorage.getItem('tradingConfig')
      if (savedConfig) {
        try {
          setConfig(JSON.parse(savedConfig))
        } catch (e) {
          console.error("Failed to parse tradingConfig:", e)
        }
      }
    }
  }, [open])

  const handleSave = () => {
    localStorage.setItem('tradingConfig', JSON.stringify(config))
    // Dispatch an event so other components (like binance-trading-dashboard) can update if they are listening
    window.dispatchEvent(new Event('tradingConfigUpdated'))
    toast({
      title: "Settings Saved",
      description: "Default global trading configuration has been updated.",
    })
    setOpen(false)
  }

  const handleChange = (field: keyof TradingConfig, value: string | number) => {
    setConfig(prev => ({ ...prev, [field]: value }))
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          className="gap-2 transition-all text-[#EAEAEA]/70 hover:text-[#1DB954] hover:bg-[#1DB954]/10"
        >
          <SlidersHorizontal className="w-4 h-4" />
          <span className="hidden sm:inline">Trade Config</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px] bg-[#1A1A1A]/90 border-[#333]/50 text-foreground backdrop-blur-md">
        <DialogHeader>
          <DialogTitle>Global Trading Configuration</DialogTitle>
          <DialogDescription>
            Set your default trading parameters. These will be applied across the trading platform.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid gap-4 py-4 max-h-[60vh] overflow-y-auto">
          <div className="grid gap-2">
            <label className="text-sm font-medium">Default Trading Pair</label>
            <select
              value={config.symbol}
              onChange={(e) => handleChange('symbol', e.target.value)}
              className="w-full px-3 py-2 border rounded-md bg-background"
            >
              {TRADING_PAIRS.map((pair) => (
                <option key={pair} value={pair}>
                  {pair}
                </option>
              ))}
            </select>
          </div>

          <div className="grid gap-2">
            <label className="text-sm font-medium">Default Strategy</label>
            <select
              value={config.strategy}
              onChange={(e) => handleChange('strategy', e.target.value)}
              className="w-full px-3 py-2 border rounded-md bg-background"
            >
              <option value="MA_CROSSOVER">Moving Average Crossover</option>
              <option value="MACD">MACD Strategy</option>
              <option value="MEAN_REVERSION">Mean Reversion</option>
              <option value="GRID">Grid Trading</option>
              <option value="MULTI_SIGNAL">Multi-Signal Ensemble</option>
              <option disabled>--- New Strategies ---</option>
              <option value="RSI_DIVERGENCE">RSI Divergence</option>
              <option value="BOLLINGER_BREAKOUT">Bollinger Breakout</option>
              <option value="VWAP_TREND">VWAP Trend</option>
              <option value="ICHIMOKU">Ichimoku Cloud</option>
              <option value="PIVOT_REVERSAL">Pivot Reversal</option>
            </select>
          </div>

          <div className="grid gap-2">
            <label className="text-sm font-medium">Default Timeframe</label>
            <select
              value={config.timeframe}
              onChange={(e) => handleChange('timeframe', e.target.value)}
              className="w-full px-3 py-2 border rounded-md bg-background"
            >
              <option value="1m">1m</option>
              <option value="3m">3m</option>
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="30m">30m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium">Risk per Trade (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={config.riskPerTrade}
                onChange={(e) => handleChange('riskPerTrade', parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">Leverage (x)</label>
              <input
                type="number"
                min="1"
                max="125"
                step="1"
                value={config.leverage}
                onChange={(e) => handleChange('leverage', parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium">Stop Loss (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={config.stopLoss}
                onChange={(e) => handleChange('stopLoss', parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">Take Profit (%)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={config.takeProfit}
                onChange={(e) => handleChange('takeProfit', parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium">Trailing Stop (%)</label>
              <input
                type="number"
                min="0"
                step="0.1"
                value={config.trailingStop}
                onChange={(e) => handleChange('trailingStop', parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium">Max Positions</label>
              <input
                type="number"
                min="1"
                step="1"
                value={config.maxPositions}
                onChange={(e) => handleChange('maxPositions', parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 border rounded-md bg-background"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleSave} 
            className="bg-[#1DB954] hover:bg-[#1AA34A] text-white"
          >
            Save Default Config
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
