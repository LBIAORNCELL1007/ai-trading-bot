"use client"

import { useEffect, useRef, useState } from "react"
import {
    createChart,
    ColorType,
    CrosshairMode,
    IChartApi,
    ISeriesApi,
    Time,
    CandlestickSeries,
    HistogramSeries,
    LineSeries
} from "lightweight-charts"
import type { BinanceKline } from "@/lib/binance-websocket"

interface FinancialChartProps {
    data: BinanceKline[]
    colors?: {
        backgroundColor?: string
        lineColor?: string
        textColor?: string
        areaTopColor?: string
        areaBottomColor?: string
    }
}

export function FinancialChart({ data, colors: {
    backgroundColor = '#1e1e1e',
    textColor = '#d1d5db',
} = {} }: FinancialChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null)
    const chartRef = useRef<IChartApi | null>(null)
    const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null)
    const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null)
    const ma7SeriesRef = useRef<ISeriesApi<"Line"> | null>(null)
    const ma25SeriesRef = useRef<ISeriesApi<"Line"> | null>(null)
    const ma99SeriesRef = useRef<ISeriesApi<"Line"> | null>(null)
    const [currentData, setCurrentData] = useState<BinanceKline | null>(null)
    const [maValues, setMaValues] = useState<{ ma7: number, ma25: number, ma99: number } | null>(null)

    // Calculate Moving Averages
    const calculateMA = (data: BinanceKline[], count: number) => {
        const result = []
        for (let i = 0; i < data.length; i++) {
            if (i < count - 1) {
                continue
            }

            let sum = 0
            for (let j = 0; j < count; j++) {
                sum += data[i - j].close
            }

            result.push({
                time: data[i].time / 1000 as Time,
                value: sum / count
            })
        }
        return result
        return result
    }

    // Helper for single point MA
    const calculateSimpleMA = (data: BinanceKline[], count: number, index: number) => {
        if (index < count - 1) return 0
        let sum = 0
        for (let j = 0; j < count; j++) {
            sum += data[index - j].close
        }
        return sum / count
    }

    useEffect(() => {
        if (!chartContainerRef.current) return

        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
            }
        }

        // Initialize Chart
        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: textColor,
            },
            grid: {
                vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
                horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 600,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: '#2B2B43',
            },
            crosshair: {
                mode: CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: '#2B2B43',
            },
        })

        chartRef.current = chart

        // 1. Candlestick Series
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#0ecb81',
            downColor: '#f6465d',
            borderDownColor: '#f6465d',
            borderUpColor: '#0ecb81',
            wickDownColor: '#f6465d',
            wickUpColor: '#0ecb81',
        })
        candleSeriesRef.current = candleSeries

        // 2. Volume Series (Histogram)
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        })
        volumeSeries.priceScale().applyOptions({
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        })
        volumeSeriesRef.current = volumeSeries

        // 3. Moving Averages
        const ma7Series = chart.addSeries(LineSeries, { color: '#f59e0b', lineWidth: 1, crosshairMarkerVisible: false })
        ma7SeriesRef.current = ma7Series

        const ma25Series = chart.addSeries(LineSeries, { color: '#a855f7', lineWidth: 1, crosshairMarkerVisible: false })
        ma25SeriesRef.current = ma25Series

        const ma99Series = chart.addSeries(LineSeries, { color: '#06b6d4', lineWidth: 1, crosshairMarkerVisible: false })
        ma99SeriesRef.current = ma99Series

        // Crosshair Move Handler
        chart.subscribeCrosshairMove(param => {
            if (param.time) {
                const dataPoint = data.find(d => d.time / 1000 === param.time)
                if (dataPoint) {
                    setCurrentData(dataPoint)
                    // Calculate MAs for this point (or look up if pre-calculated)
                    // For simplicity, we can calculate on fly or look up if we stored them
                    // MAs are simple: sum of last N closes
                    // Let's find index
                    const index = data.indexOf(dataPoint)
                    if (index >= 0) {
                        const ma7 = calculateSimpleMA(data, 7, index)
                        const ma25 = calculateSimpleMA(data, 25, index)
                        const ma99 = calculateSimpleMA(data, 99, index)
                        setMaValues({ ma7, ma25, ma99 })
                    }
                }
            } else {
                // Revert to latest
                if (data.length > 0) {
                    const last = data[data.length - 1]
                    setCurrentData(last)
                    const index = data.length - 1
                    setMaValues({
                        ma7: calculateSimpleMA(data, 7, index),
                        ma25: calculateSimpleMA(data, 25, index),
                        ma99: calculateSimpleMA(data, 99, index)
                    })
                }
            }
        })

        // Initial Set
        if (data.length > 0) {
            const last = data[data.length - 1]
            setCurrentData(last)
            const index = data.length - 1
            setMaValues({
                ma7: calculateSimpleMA(data, 7, index),
                ma25: calculateSimpleMA(data, 25, index),
                ma99: calculateSimpleMA(data, 99, index)
            })
        }

        window.addEventListener('resize', handleResize)

        return () => {
            window.removeEventListener('resize', handleResize)
            chart.remove()
        }
    }, [backgroundColor, textColor, data]) // Re-bind when data changes to have fresh closure

    // Update Data
    useEffect(() => {
        if (!candleSeriesRef.current || !volumeSeriesRef.current || data.length === 0) return

        // Format data for lightweight-charts
        // Unique and sorted time check could be added here if needed, but Binance data is usually sorted
        const candleData = data.map(d => ({
            time: (d.time / 1000) as Time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }))

        const volumeData = data.map(d => ({
            time: (d.time / 1000) as Time,
            value: d.volume,
            color: d.close >= d.open ? 'rgba(14, 203, 129, 0.3)' : 'rgba(246, 70, 93, 0.3)',
        }))

        try {
            candleSeriesRef.current.setData(candleData)
            volumeSeriesRef.current.setData(volumeData)

            if (ma7SeriesRef.current) ma7SeriesRef.current.setData(calculateMA(data, 7))
            if (ma25SeriesRef.current) ma25SeriesRef.current.setData(calculateMA(data, 25))
            if (ma99SeriesRef.current) ma99SeriesRef.current.setData(calculateMA(data, 99))

        } catch (e) {
            console.error("Error updating chart data:", e)
        }

    }, [data])

    return (
        <div className="relative">
            {/* Legend Overlay */}
            <div className="absolute top-2 left-2 z-10 flex gap-4 text-xs font-mono select-none pointer-events-none">
                <div className="flex flex-col gap-1">
                    <div className="block">
                        {currentData && (
                            <div className="flex gap-3">
                                <span className="text-muted-foreground mr-2">
                                    <span className="text-yellow-500 mr-1">MA(7): {maValues?.ma7.toFixed(2)}</span>
                                    <span className="text-purple-500 mr-1">MA(25): {maValues?.ma25.toFixed(2)}</span>
                                    <span className="text-cyan-500">MA(99): {maValues?.ma99.toFixed(2)}</span>
                                </span>
                                <span className="text-muted-foreground">O: <span className={currentData.close >= currentData.open ? "text-green-500" : "text-red-500"}>{currentData.open.toFixed(2)}</span></span>
                                <span className="text-muted-foreground">H: <span className={currentData.close >= currentData.open ? "text-green-500" : "text-red-500"}>{currentData.high.toFixed(2)}</span></span>
                                <span className="text-muted-foreground">L: <span className={currentData.close >= currentData.open ? "text-green-500" : "text-red-500"}>{currentData.low.toFixed(2)}</span></span>
                                <span className="text-muted-foreground">C: <span className={currentData.close >= currentData.open ? "text-green-500" : "text-red-500"}>{currentData.close.toFixed(2)}</span></span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div ref={chartContainerRef} className="w-full h-[600px]" />
        </div>
    )
}
