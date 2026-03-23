"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts"
import { AlertCircle, TrendingUp, TrendingDown, Zap } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import type { ModelPrediction, BacktestResult } from "@/lib/ml-models"
import type { TradeStats, PortfolioMetrics } from "@/lib/strategy-engine"

interface MLAnalyticsProps {
  predictions: ModelPrediction[]
  backtestResults?: BacktestResult
  tradeStats?: TradeStats
  portfolioMetrics?: PortfolioMetrics
  historyReturns: number[]
}

export default function MLAnalytics({
  predictions,
  backtestResults,
  tradeStats,
  portfolioMetrics,
  historyReturns,
}: MLAnalyticsProps) {
  // Prepare chart data
  const predictionHistory = predictions.slice(-50).map((p, i) => ({
    time: i,
    probability: (p.probability * 100).toFixed(1),
    confidence: (p.confidence * 100).toFixed(1),
  }))

  const equityHistory = historyReturns.slice(-50).map((r, i) => ({
    time: i,
    value: 10000 * (1 + r),
  }))

  const signalDistribution = predictions.slice(-100).map((p, i) => ({
    time: i,
    tcn: (p.signals.tcn * 100).toFixed(1),
    tda: (p.signals.tda * 100).toFixed(1),
    technical: (p.signals.technicalIndicators * 100).toFixed(1),
  }))

  const currentPrediction = predictions[predictions.length - 1] || {
    probability: 0.5,
    confidence: 0,
    signals: { tcn: 0.5, tda: 0.5, technicalIndicators: 0.5 },
  }

  const radarData = [
    { name: "TCN", value: currentPrediction.signals.tcn * 100 },
    { name: "TDA", value: currentPrediction.signals.tda * 100 },
    { name: "Technical", value: currentPrediction.signals.technicalIndicators * 100 },
  ]

  return (
    <div className="space-y-4">
      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Probability
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(currentPrediction.probability * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {currentPrediction.probability > 0.55
                ? "Bullish"
                : currentPrediction.probability < 0.45
                  ? "Bearish"
                  : "Neutral"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Model Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(currentPrediction.confidence * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {currentPrediction.confidence > 0.7
                ? "High"
                : currentPrediction.confidence > 0.4
                  ? "Medium"
                  : "Low"}
            </p>
          </CardContent>
        </Card>

        {backtestResults && (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Win Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(backtestResults.winRate * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {backtestResults.totalTrades} trades
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Sharpe Ratio
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {backtestResults.sharpeRatio.toFixed(2)}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Risk-adjusted return
                </p>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      {/* Model Signals Radar */}
      <Card>
        <CardHeader>
          <CardTitle>Model Signal Components</CardTitle>
          <CardDescription>Individual model predictions (TCN, TDA, Technical)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar name="Signal Strength" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Prediction Trends */}
      <Tabs defaultValue="predictions" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="signals">Model Signals</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Probability & Confidence Trends</CardTitle>
              <CardDescription>Last 50 predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={predictionHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="probability"
                    stroke="#10b981"
                    strokeWidth={2}
                    name="Probability (%)"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    name="Confidence (%)"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="signals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Signal Comparison</CardTitle>
              <CardDescription>Individual model contributions</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={signalDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="tcn" stackId="a" fill="#8b5cf6" name="TCN" isAnimationActive={false} />
                  <Bar dataKey="tda" stackId="a" fill="#ec4899" name="TDA" isAnimationActive={false} />
                  <Bar dataKey="technical" stackId="a" fill="#06b6d4" name="Technical" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          {/* Performance Metrics */}
          {backtestResults && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Backtest Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Return:</span>
                    <Badge
                      variant={backtestResults.totalReturn > 0 ? "default" : "destructive"}
                    >
                      {(backtestResults.totalReturn * 100).toFixed(2)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sharpe Ratio:</span>
                    <span className="font-medium">{backtestResults.sharpeRatio.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Max Drawdown:</span>
                    <Badge variant="destructive">
                      {(backtestResults.maxDrawdown * 100).toFixed(2)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Win Rate:</span>
                    <span className="font-medium">
                      {(backtestResults.winRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Trades:</span>
                    <span className="font-medium">{backtestResults.totalTrades}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Trade Statistics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {tradeStats && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Trades:</span>
                        <span className="font-medium">{tradeStats.totalTrades}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Winning:</span>
                        <Badge variant="default">{tradeStats.winningTrades}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Losing:</span>
                        <Badge variant="destructive">{tradeStats.losingTrades}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Win:</span>
                        <span className="text-green-600 font-medium">
                          ${tradeStats.avgWinSize.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Avg Loss:</span>
                        <span className="text-red-600 font-medium">
                          ${tradeStats.avgLossSize.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Profit Factor:</span>
                        <span className="font-medium">{tradeStats.profitFactor.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Expectancy:</span>
                        <span className="font-medium">${tradeStats.expectancy.toFixed(2)}</span>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Portfolio Metrics */}
          {portfolioMetrics && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Portfolio Metrics</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-muted-foreground text-sm">Total Value</p>
                  <p className="text-2xl font-bold">
                    ${portfolioMetrics.totalValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Cash Available</p>
                  <p className="text-2xl font-bold">
                    ${portfolioMetrics.totalCash.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Open Positions</p>
                  <p className="text-2xl font-bold">{portfolioMetrics.totalPositions}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Annual Volatility</p>
                  <p className="text-2xl font-bold">
                    {(portfolioMetrics.portfolioVolatility * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Sharpe Ratio</p>
                  <p className="text-2xl font-bold">{portfolioMetrics.sharpeRatio.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Max Drawdown</p>
                  <p className="text-2xl font-bold text-red-600">
                    {(portfolioMetrics.maxDrawdown * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground text-sm">Calmar Ratio</p>
                  <p className="text-2xl font-bold">{portfolioMetrics.calmarRatio.toFixed(2)}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Model Health Alert */}
      {currentPrediction.confidence < 0.4 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Low Model Confidence:</strong> The ML models are uncertain about market direction.
            Consider reducing position sizes or waiting for clearer signals.
          </AlertDescription>
        </Alert>
      )}

      {predictions.length < 20 && (
        <Alert>
          <Zap className="h-4 w-4" />
          <AlertDescription>
            <strong>Warming up:</strong> Models are gathering historical data. Full functionality will be
            available after {20 - predictions.length} more candles.
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
